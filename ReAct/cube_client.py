
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

    # ---------------- Public helpers ----------------
    def total_cost_by_service_fy(self, org: str, service: str) -> pd.DataFrame:
        """Query actual cost by org/service/fiscal year from FctItCosts."""
        if self.use_cube:
            print("[CubeClient] Querying FctItCosts.actualCost for", org, service)
            payload = {
                "query": {
                    "measures": ["FctItCosts.actualCost"],
                    "dimensions": [
                        "DimOrg.businessUnit",
                        "DimService.serviceName",
                        "FctItCosts.fiscalYear"
                    ],
                    "filters": [
                        {"dimension": "DimOrg.businessUnit", "operator": "equals", "values": [org]},
                        {"dimension": "DimService.serviceName", "operator": "equals", "values": [service]}
                    ],
                    "limit": 500
                }
            }
            resp = requests.post(f"{CUBEJS_API_URL.rstrip('/')}/load", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return pd.DataFrame(data)

    def top_cost_drivers(self, org: str, fy_old: str, fy_new: str, top_n: int = 5) -> pd.DataFrame:
        if self.use_cube:
            print("[CubeClient] Querying top cost drivers for", org, fy_new)
            payload = {
                "query": {
                    "measures": ["FctItCosts.actualCost"],
                    "dimensions": [
                        "DimService.serviceName",
                        "FctItCosts.costType",
                        "FctItCosts.fiscalYear"
                    ],
                    "filters": [
                        {"dimension": "DimOrg.businessUnit", "operator": "equals", "values": [org]},
                        {"dimension": "FctItCosts.fiscalYear", "operator": "in", "values": [fy_old, fy_new]}
                    ],
                    "order": [["FctItCosts.actualCost", "desc"]],
                    "limit": top_n
                }
            }
            resp = requests.post(f"{CUBEJS_API_URL.rstrip('/')}/load", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return pd.DataFrame(data)

    def cost_delta_summary(self, org: str, fy_old: str, fy_new: str) -> pd.DataFrame:
        """Compare actual costs for a business unit between two fiscal years using Cube.js."""
        if self.use_cube:
            print(f"[CubeClient] Querying cost delta summary for {org}: {fy_old} → {fy_new}")
            payload = {
                "query": {
                    "measures": ["FctItCosts.actualCost"],
                    "dimensions": ["DimOrg.businessUnit", "FctItCosts.fiscalYear"],
                    "filters": [
                        {"dimension": "DimOrg.businessUnit", "operator": "equals", "values": [org]},
                        {"dimension": "FctItCosts.fiscalYear", "operator": "in", "values": [fy_old, fy_new]}
                    ],
                    "limit": 500
                }
            }
            resp = requests.post(f"{CUBEJS_API_URL.rstrip('/')}/load", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            df = pd.DataFrame(data)

            if df.empty:
                print(f"[CubeClient] Warning: No data returned for {org}, FY {fy_old}/{fy_new}")
                return df

            try:
                df_wide = df.pivot(index="DimOrg.businessUnit",
                                   columns="FctItCosts.fiscalYear",
                                   values="FctItCosts.actualCost").fillna(0)

                # Cast to numeric (Cube.js returns strings)
                fy_old_val = float(df_wide.get(fy_old, pd.Series([0])).iloc[0])
                fy_new_val = float(df_wide.get(fy_new, pd.Series([0])).iloc[0])

                delta_abs = fy_new_val - fy_old_val
                delta_pct = (delta_abs / fy_old_val) if fy_old_val else None

                df_delta = pd.DataFrame({
                    "org": [org],
                    "fy_old": [fy_old],
                    "fy_new": [fy_new],
                    "cost_old": [fy_old_val],
                    "cost_new": [fy_new_val],
                    "delta_abs": [delta_abs],
                    "delta_pct": [delta_pct]
                })
                print(
                    f"[CubeClient] FY{fy_old}: {fy_old_val:,.2f}, FY{fy_new}: {fy_new_val:,.2f}, Δ = {delta_abs:,.2f}")
                return df_delta

            except Exception as e:
                print("[CubeClient] Pivot or delta computation failed:", e)
                return df





