
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import duckdb
import os
import time
import hashlib
import requests

from config import CUBEJS_API_URL, DUCKDB_PATH

class CubeClient:
    """
    If CUBEJS_API_URL is set, uses Cube.js REST API.
    Otherwise, falls back to DuckDB local queries.
    """
    def __init__(self):
        self.use_cube = bool(CUBEJS_API_URL)
        duckdb_path = r"C:\Users\jakob\tbm_demo\tbm_demo.duckdb"
        self.duckdb_path = duckdb_path
        if self.use_cube:
            print(f"[CubeClient] Using Cube.js API at {CUBEJS_API_URL}")
            self._connect = None  # disable local connect
        else:
            print("[CubeClient] No Cube.js API detected – fallback to local DuckDB.")
            self._ensure_duckdb()

    # ---------------- Cube.js path ----------------
    def query_cubejs(self, measures, dimensions, filters,
                     timeDimensions=None, limit=5000) -> pd.DataFrame:
        query = {
            "measures": measures,
            "dimensions": dimensions,
            "filters": filters,
            "timeDimensions": timeDimensions or [],
            "limit": limit
        }
        payload = {"query": query}  # <--- wichtig!
        t0 = time.time()
        resp = requests.post(f"{CUBEJS_API_URL.rstrip('/')}/load", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        df = pd.DataFrame(data)
        df.attrs["provenance"] = {
            "engine": "cubejs",
            "payload": payload,
            "elapsed_sec": round(time.time() - t0, 3),
            "query_hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        }
        return df

    # ---------------- DuckDB path -----------------
    def _ensure_duckdb(self):
        # disable demo data generation
        if not os.path.exists(self.duckdb_path):
            raise FileNotFoundError(
                f"DuckDB not found at {self.duckdb_path}. "
                "Please point DUCKDB_PATH to your real fact_it_costs database."
            )

    def _connect(self):
        return duckdb.connect(self.duckdb_path, read_only=False)

    def query_duckdb_sql(self, sql:str, params:Tuple=()) -> pd.DataFrame:
        t0 = time.time()
        with self._connect() as con:
            df = con.execute(sql, params).fetchdf()
        df.attrs["provenance"] = {
            "engine":"duckdb",
            "sql": sql,
            "params": params,
            "elapsed_sec": round(time.time()-t0,3),
            "query_hash": hashlib.sha256((sql+str(params)).encode()).hexdigest()[:16]
        }
        return df

    def total_cost_by_country(self, country: str, country_dim: str = "DimCountry.countryName") -> pd.DataFrame:
        if self.use_cube:
            print(f"[CubeClient] Querying total cost for {country} ({country_dim})")
            payload = {
                "query": {
                    "measures": ["FctItCosts.actualCost"],
                    "dimensions": [country_dim],
                    "filters": [
                        {"dimension": country_dim, "operator": "equals", "values": [country]}
                    ],
                    "limit": 500
                }
            }
            resp = requests.post(f"{CUBEJS_API_URL.rstrip('/')}/load", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return pd.DataFrame(data)

    # ---------------- Public helpers ----------------
    def total_cost_by_service_fy(self, org: str, service: str,
                                 service_dim: str = "DimService.serviceName") -> pd.DataFrame:
        """
        Query actual cost, price, and quantity by org/service/app/fiscal year.
        """
        if self.use_cube:
            print(f"[CubeClient] Querying FctItCosts.actualCost + price + quantity for {org} {service} ({service_dim})")
            payload = {
                "query": {
                    "measures": [
                        "FctItCosts.actualCost",
                        "FctItCosts.price",
                        "FctItCosts.quantity"
                    ],
                    "dimensions": [
                        "DimOrg.businessUnit",
                        service_dim,
                        "FctItCosts.fiscalYear"
                    ],
                    "filters": [
                        {"dimension": "DimOrg.businessUnit", "operator": "equals", "values": [org]},
                        {"dimension": service_dim, "operator": "equals", "values": [service]}
                    ],
                    "limit": 500,
                    "cacheMode": "stale-if-slow"
                }
            }
            resp = requests.post(f"{CUBEJS_API_URL.rstrip('/')}/load", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            df = pd.DataFrame(data)
            # normalize datatypes
            for col in ["FctItCosts.actualCost", "FctItCosts.price", "FctItCosts.quantity"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df

    def top_cost_drivers(self, org: str, fy_old: str, fy_new: str, top_n: int = 5) -> pd.DataFrame:
        """
        Retrieve Top-N cost drivers for a business unit, aggregated by service.
        Phase 1: Get top-N services for FY_new.
        Phase 2: Fetch the same services for FY_old.
        Then combine, remove duplicates, and return the full FY24/FY25 dataset.
        """
        if not self.use_cube:
            raise RuntimeError("Cube.js must be active")

        base_url = CUBEJS_API_URL.rstrip('/')

        # --- Phase 1: Top-N services for FY_new ---
        print(f"[CubeClient] Top {top_n} cost drivers for {org} in {fy_new}")
        payload_new = {
            "query": {
                "measures": ["FctItCosts.actualCost"],
                "dimensions": ["DimService.serviceName", "FctItCosts.fiscalYear"],
                "filters": [
                    {"dimension": "DimOrg.businessUnit", "operator": "equals", "values": [org]},
                    {"dimension": "FctItCosts.fiscalYear", "operator": "equals", "values": [fy_new]},
                ],
                "order": {"FctItCosts.actualCost": "desc"},
                "limit": top_n,
                "cacheMode": "stale-if-slow"
            }
        }

        r_new = requests.post(f"{base_url}/load", json=payload_new, timeout=60)
        r_new.raise_for_status()
        df_new = pd.DataFrame(r_new.json().get("data", []))

        if df_new.empty:
            print(f"[CubeClient] Warning: No FY_new data for {org} in {fy_new}")
            return df_new

        # ensure numeric and distinct
        df_new["FctItCosts.actualCost"] = pd.to_numeric(df_new["FctItCosts.actualCost"], errors="coerce")
        top_services = df_new["DimService.serviceName"].unique().tolist()

        # --- Phase 2: Fetch FY_old values for the same services ---
        print(f"[CubeClient] Fetching FY {fy_old} values for Top {len(top_services)} services")
        payload_old = {
            "query": {
                "measures": ["FctItCosts.actualCost"],
                "dimensions": ["DimService.serviceName", "FctItCosts.fiscalYear"],
                "filters": [
                    {"dimension": "DimOrg.businessUnit", "operator": "equals", "values": [org]},
                    {"dimension": "FctItCosts.fiscalYear", "operator": "equals", "values": [fy_old]},
                    {"dimension": "DimService.serviceName", "operator": "in", "values": top_services}
                ],
                "cacheMode": "stale-if-slow"
            }
        }

        r_old = requests.post(f"{base_url}/load", json=payload_old, timeout=60)
        r_old.raise_for_status()
        df_old = pd.DataFrame(r_old.json().get("data", []))

        if not df_old.empty:
            df_old["FctItCosts.actualCost"] = pd.to_numeric(df_old["FctItCosts.actualCost"], errors="coerce")

        # --- Combine and deduplicate ---
        df_combined = pd.concat([df_new, df_old], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["DimService.serviceName", "FctItCosts.fiscalYear"], keep="first"
        ).reset_index(drop=True)

        # --- Aggregate by service + FY (just in case of multiple cost types) ---
        df_combined = (
            df_combined.groupby(["DimService.serviceName", "FctItCosts.fiscalYear"])["FctItCosts.actualCost"]
            .sum()
            .reset_index()
        )

        # --- Total BU costs (for normalization) ---
        print(f"[CubeClient] Fetching total cost for {org} ({fy_old} & {fy_new})")
        payload_total = {
            "query": {
                "measures": ["FctItCosts.actualCost"],
                "dimensions": ["DimOrg.businessUnit", "FctItCosts.fiscalYear"],
                "filters": [
                    {"dimension": "DimOrg.businessUnit", "operator": "equals", "values": [org]},
                    {"dimension": "FctItCosts.fiscalYear", "operator": "in", "values": [fy_old, fy_new]}
                ],
                "cacheMode": "stale-if-slow"
            }
        }
        r_total = requests.post(f"{base_url}/load", json=payload_total, timeout=60)
        r_total.raise_for_status()
        df_total = pd.DataFrame(r_total.json().get("data", []))
        if not df_total.empty:
            df_total["FctItCosts.actualCost"] = pd.to_numeric(df_total["FctItCosts.actualCost"], errors="coerce")
            totals = {
                fy_old: df_total.loc[df_total["FctItCosts.fiscalYear"] == fy_old, "FctItCosts.actualCost"].sum(),
                fy_new: df_total.loc[df_total["FctItCosts.fiscalYear"] == fy_new, "FctItCosts.actualCost"].sum()
            }
            print(f"[CubeClient] Totals for {org}: {fy_old}={totals[fy_old]:,.2f}, {fy_new}={totals[fy_new]:,.2f}")
        else:
            totals = {fy_old: None, fy_new: None}
        df_combined.attrs["total_costs"] = {"org": org, **totals}

        print(f"[CubeClient] Retrieved {len(df_combined)} rows ({len(top_services)} distinct services)")

        return df_combined

    # cube_client.py

    def query(self, q: dict) -> pd.DataFrame:
        if not isinstance(q, dict):
            raise TypeError("CubeClient.query expects a dict")
        if not self.use_cube:
            raise RuntimeError("Cube.js must be active")

        # LLM benutzt 'member' – Cube.js will 'dimension'
        for f in q.get("filters", []) or []:
            if "member" in f and "dimension" not in f:
                f["dimension"] = f.pop("member")

        # Filter unzulässige timeDimensions (z. B. fiscalYear)
        valid_time_dims = []
        for td in q.get("timeDimensions", []) or []:
            if "fiscalYear" not in (td.get("dimension") or ""):
                valid_time_dims.append(td)
        q["timeDimensions"] = valid_time_dims

        return self.query_cubejs(
            measures=q.get("measures", []),
            dimensions=q.get("dimensions", []),
            filters=q.get("filters", []),
            timeDimensions=q.get("timeDimensions", []),
            limit=q.get("limit", 5000),
        )

    def cost_delta_summary(self, org: str, fy_old: str, fy_new: str, top_n: int = 5) -> pd.DataFrame:
        """
        Service-level Delta FY_old -> FY_new für eine BU.
        Phase 1: Top-N Services aus FY_new holen (aggregiert nach Service).
        Phase 2: Für genau diese Services FY_old-Werte holen.
        Dann pivotieren und Deltas berechnen.
        """
        if not self.use_cube:
            raise RuntimeError("Cube.js must be active")

        base_url = CUBEJS_API_URL.rstrip('/')

        # --- Phase 1: Top-N Services in FY_new ---
        print(f"[CubeClient] Delta summary: Top {top_n} services for {org} in {fy_new}")
        payload_new = {
            "query": {
                "measures": ["FctItCosts.actualCost"],
                "dimensions": ["DimService.serviceName", "FctItCosts.fiscalYear"],
                "filters": [
                    {"dimension": "DimOrg.businessUnit", "operator": "equals", "values": [org]},
                    {"dimension": "FctItCosts.fiscalYear", "operator": "equals", "values": [fy_new]}
                ],
                "order": {"FctItCosts.actualCost": "desc"},
                "limit": top_n,
                "cacheMode": "stale-if-slow"
            }
        }
        r_new = requests.post(f"{base_url}/load", json=payload_new, timeout=60)
        r_new.raise_for_status()
        df_new = pd.DataFrame(r_new.json().get("data", []))

        if df_new.empty:
            print(f"[CubeClient] Warning: no FY_new data for {org} in {fy_new}")
            return df_new

        df_new["FctItCosts.actualCost"] = pd.to_numeric(df_new["FctItCosts.actualCost"], errors="coerce")
        top_services = df_new["DimService.serviceName"].unique().tolist()

        # --- Phase 2: FY_old für die gleichen Services ---
        print(f"[CubeClient] Delta summary: fetch {fy_old} for {len(top_services)} services")
        payload_old = {
            "query": {
                "measures": ["FctItCosts.actualCost"],
                "dimensions": ["DimService.serviceName", "FctItCosts.fiscalYear"],
                "filters": [
                    {"dimension": "DimOrg.businessUnit", "operator": "equals", "values": [org]},
                    {"dimension": "FctItCosts.fiscalYear", "operator": "equals", "values": [fy_old]},
                    {"dimension": "DimService.serviceName", "operator": "in", "values": top_services}
                ],
                "cacheMode": "stale-if-slow"
            }
        }
        r_old = requests.post(f"{base_url}/load", json=payload_old, timeout=60)
        r_old.raise_for_status()
        df_old = pd.DataFrame(r_old.json().get("data", []))
        if not df_old.empty:
            df_old["FctItCosts.actualCost"] = pd.to_numeric(df_old["FctItCosts.actualCost"], errors="coerce")

        # --- Kombinieren ohne Duplikate, dann pivotieren ---
        df_comb = pd.concat([df_new, df_old], ignore_index=True)
        df_comb = df_comb.drop_duplicates(
            subset=["DimService.serviceName", "FctItCosts.fiscalYear"], keep="first"
        )

        pivot = (df_comb
                 .groupby(["DimService.serviceName", "FctItCosts.fiscalYear"])["FctItCosts.actualCost"]
                 .sum()
                 .unstack())  # Spalten = FY_old/FY_new
        # fehlende Spalten ggf. ergänzen
        for fy in [fy_old, fy_new]:
            if fy not in pivot.columns:
                pivot[fy] = 0.0

        pivot = pivot.reset_index().rename(columns={"DimService.serviceName": "service"})
        # numerisch sicherstellen
        pivot[fy_old] = pd.to_numeric(pivot[fy_old], errors="coerce").fillna(0.0)
        pivot[fy_new] = pd.to_numeric(pivot[fy_new], errors="coerce").fillna(0.0)

        # Deltas
        pivot["delta_abs"] = pivot[fy_new] - pivot[fy_old]
        pivot["delta_pct"] = pivot["delta_abs"] / pivot[fy_old].replace(0, pd.NA)

        # zur Sicherheit nach größter Änderung sortieren (optional top_n wieder anwenden)
        pivot = pivot.sort_values("delta_abs", ascending=False).reset_index(drop=True)

        print(f"[CubeClient] Delta rows: {len(pivot)} (services={len(top_services)})")
        # Einheitliche Spaltennamen für Downstream
        return pivot.rename(columns={fy_old: "cost_old", fy_new: "cost_new"})








