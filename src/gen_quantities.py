from __future__ import annotations
import random
from typing import List, Dict
from .config import (
    FISCAL_YEARS, FISCAL_MONTHS, SERVICES, BUSINESS_UNITS,
    RUN_QUANTITY_BASE, RUN_MONTHLY_NOISE, RANDOM_SEED,
    DIM_APPS, BU_COUNTRIES, DIM_COST_CENTERS
)

random.seed(RANDOM_SEED)


def gen_run_quantities() -> List[Dict]:
    rows: List[Dict] = []
    for fy in FISCAL_YEARS:
        for m in FISCAL_MONTHS:
            for s in SERVICES:
                sid = s["service_id"]
                tid = s["tower_id"]
                for buc in BU_COUNTRIES:
                    org = buc["org_id"]
                    country = buc["country_code"]
                    base = RUN_QUANTITY_BASE.get(org, 1.0)
                    for app in DIM_APPS:
                        app_id = app["app_id"]
                        for cc in DIM_COST_CENTERS:
                            cc_id = cc["cost_center_id"]
                            noise = 1.0 + random.uniform(-RUN_MONTHLY_NOISE, RUN_MONTHLY_NOISE)
                            qty = max(0.0, round(1_000 * base * noise, 2))
                            rows.append({
                                "fiscal_year": fy,
                                "fiscal_month_num": m,
                                "tower_id": tid,
                                "service_id": sid,
                                "org_id": org,  # = BU_03
                                "country_code": country,  # = DE
                                "app_id": app_id,
                                "cost_center_id": cc_id,
                                "quantity": qty,
                            })
    return rows