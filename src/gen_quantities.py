from __future__ import annotations
import random
from typing import List, Dict
from .config import (
    FISCAL_YEARS, FISCAL_MONTHS, SERVICES, BUSINESS_UNITS,
    RUN_QUANTITY_BASE, RUN_MONTHLY_NOISE, RANDOM_SEED,
)

random.seed(RANDOM_SEED)


def gen_run_quantities() -> List[Dict]:
    rows: List[Dict] = []
    for fy in FISCAL_YEARS:
        for m in FISCAL_MONTHS:
            for s in SERVICES:
                sid = s["service_id"]
                tid = s["tower_id"]
                for bu in BUSINESS_UNITS:
                    org = bu["org_id"]
                    base = RUN_QUANTITY_BASE.get(org, 1.0)
                    # Small month noise ±RUN_MONTHLY_NOISE
                    noise = 1.0 + random.uniform(-RUN_MONTHLY_NOISE, RUN_MONTHLY_NOISE)
                    qty = max(0.0, round(1000 * base * noise, 2))
                    rows.append({
                        "fiscal_year": fy,
                        "fiscal_month_num": m,
                        "tower_id": tid,
                        "service_id": sid,
                        "org_id": org,
                        "quantity": qty,
                    })
    return rows