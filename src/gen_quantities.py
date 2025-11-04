from __future__ import annotations
import random
from typing import List, Dict
from .config import (
    FISCAL_YEARS, FISCAL_MONTHS, SERVICES, BUSINESS_UNITS,
    RUN_QUANTITY_BASE, RUN_MONTHLY_NOISE, RANDOM_SEED,
    DIM_APPS, BU_COUNTRIES,
    COST_CENTER_BY_COUNTRY_AND_ORG,
    GLOBAL_APPS, LOCAL_APP_COUNTRIES,
    SERVICE_ALLOWED_APPS, CUSTOM_QUANTITY_MULTIPLIERS,  # neu
)

random.seed(RANDOM_SEED)

def _apps_allowed_in_country(service_id: str, country: str) -> List[str]:
    # 1) nur Apps, die überhaupt zu diesem Service gehören
    candidates = set(SERVICE_ALLOWED_APPS.get(service_id, []))
    if not candidates:
        return []

    # 2) Country-Availability filtern
    allowed = []
    for app_id in candidates:
        if app_id in GLOBAL_APPS:
            allowed.append(app_id)
        else:
            if country in LOCAL_APP_COUNTRIES.get(app_id, []):
                allowed.append(app_id)
    return allowed

def gen_run_quantities() -> List[Dict]:
    """
    Generates synthetic monthly run quantities for all active Apps per BU/Country.
    - Base derived from RUN_QUANTITY_BASE by Business Unit
    - Includes stochastic noise (±RUN_MONTHLY_NOISE)
    - Supports FY-specific app overrides (e.g., +10% for Microsoft 365 / Teams in FY25)
    """
    rows: List[Dict] = []

    for fy in FISCAL_YEARS:
        for m in FISCAL_MONTHS:
            for s in SERVICES:
                sid = s["service_id"]
                tid = s["tower_id"]

                for buc in BU_COUNTRIES:
                    org = buc["org_id"]
                    country = buc["country_code"]

                    # skip if BU not active in this country
                    cc_id = COST_CENTER_BY_COUNTRY_AND_ORG.get((country, org))
                    if not cc_id:
                        continue

                    allowed_apps = _apps_allowed_in_country(sid, country)
                    if not allowed_apps:
                        continue

                    # --- Baseline per BU ---
                    base_qty = RUN_QUANTITY_BASE.get(org, 1.0) * 1000  # absolute scale factor

                    for app_id in allowed_apps:
                        # --- Random monthly fluctuation ---
                        noise = 1.0 + random.uniform(-RUN_MONTHLY_NOISE, RUN_MONTHLY_NOISE)

                        # --- FY-specific multiplier (e.g., +10% in FY25) ---
                        mult = CUSTOM_QUANTITY_MULTIPLIERS.get(app_id, {}).get(fy, 1.0)

                        qty = max(0.0, round(base_qty * noise * mult, 2))

                        rows.append({
                            "fiscal_year": fy,
                            "fiscal_month_num": m,
                            "tower_id": tid,
                            "service_id": sid,
                            "org_id": org,
                            "country_code": country,
                            "app_id": app_id,
                            "cost_center_id": cc_id,
                            "quantity": qty,
                        })

    return rows
