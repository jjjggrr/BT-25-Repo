
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
        self.duckdb_path = DUCKDB_PATH
        if not self.use_cube:
            self._ensure_duckdb()

    # ---------------- Cube.js path ----------------
    def query_cubejs(self, measures: List[str], dimensions: List[str], filters: List[Dict[str,Any]],
                     timeDimensions: Optional[List[Dict[str,Any]]] = None, limit: int = 5000) -> pd.DataFrame:
        payload = {
            "measures": measures,
            "dimensions": dimensions,
            "filters": filters,
            "timeDimensions": timeDimensions or [],
            "limit": limit
        }
        t0 = time.time()
        resp = requests.post(CUBEJS_API_URL.rstrip("/") + "/load", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        df = pd.DataFrame(data)
        df.attrs["provenance"] = {
            "engine":"cubejs",
            "payload": payload,
            "elapsed_sec": round(time.time()-t0,3),
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
    def total_cost_by_service_fy(self, org:str, service:str) -> pd.DataFrame:
        if self.use_cube:
            return self.query_cubejs(
                measures=["ItCosts.total_cost"],
                dimensions=["Org.name","Service.name","FY.name"],
                filters=[
                    {"dimension":"Org.name","operator":"equals","values":[org]},
                    {"dimension":"Service.name","operator":"equals","values":[service]}
                ],
                limit=500
            )
        sql = """
        with agg as (
            select org_bu as org, service, fy, sum(price*quantity) as cost
            from fact_it_costs
            where org_bu = ? and service = ?
            group by 1,2,3
        )
        select * from agg order by fy;
        """
        return self.query_duckdb_sql(sql, (org, service))

    def top_cost_drivers(self, org:str, fy_old:str, fy_new:str, top_n:int=5) -> pd.DataFrame:
        if self.use_cube:
            # This should be implemented with your Cube schema; here we fallback to DuckDB decomposition.
            pass
        sql = """
        with rows as (
            select service, fy,
                   sum(price) as p, sum(quantity) as q, sum(price*quantity) as c
            from fact_it_costs
            where org_bu = ? and fy in (?,?)
            group by 1,2
        ),
        fy0 as (select service, p p0, q q0, c c0 from rows where fy=?),
        fy1 as (select service, p p1, q q1, c c1 from rows where fy=?),
        joined as (
            select coalesce(fy1.service, fy0.service) service,
                   coalesce(c1,0)-coalesce(c0,0) as delta,
                   (coalesce(p1,p0)-coalesce(p0,0))*coalesce(q0,0) as price_effect,
                   coalesce(p0,0)*(coalesce(q1,q0)-coalesce(q0,0)) as quantity_effect,
                   (coalesce(p1,p0)-coalesce(p0,0))*(coalesce(q1,q0)-coalesce(q0,0)) as cross_effect,
                   coalesce(c0,0) as c0, coalesce(c1,0) as c1
            from fy0 full outer join fy1 using(service)
        )
        select service, c0 as cost_old, c1 as cost_new, delta, price_effect, quantity_effect, cross_effect
        from joined
        order by abs(delta) desc
        limit ?;
        """
        return self.query_duckdb_sql(sql, (org, fy_old, fy_new, fy_old, fy_new, top_n))

    def cost_delta_summary(self, org:str, fy_old:str, fy_new:str) -> pd.DataFrame:
        sql = """
        with base as (
          select fy, sum(price*quantity) as cost
          from fact_it_costs
          where org_bu = ? and fy in (?,?)
          group by 1
        )
        select
          (select cost from base where fy=?) as cost_old,
          (select cost from base where fy=?) as cost_new,
          (select cost from base where fy=?) - (select cost from base where fy=?) as delta_abs;
        """
        return self.query_duckdb_sql(sql, (org, fy_old, fy_new, fy_old, fy_new, fy_new, fy_old))
