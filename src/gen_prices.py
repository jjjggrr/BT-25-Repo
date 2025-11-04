from __future__ import annotations
import random
from typing import Dict, List
from .config import SERVICES, FISCAL_YEARS, DEFAULT_PRICE_DELTAS, RANDOM_SEED, CURRENCY, CUSTOM_PRICE_DELTAS, app_id, \
    APP_SERVICE_MAP
from .models import ServicePrice

random.seed(RANDOM_SEED)

# Base list price per service for FY24 (seeded for deterministic output)
BASE_PRICES_FY24 = {
    s["service_id"]: round(random.uniform(4.0, 8.0), 2) for s in SERVICES
}


def gen_service_prices() -> List[ServicePrice]:
    """
    Generates synthetic service-level pricing for FY24/FY25.
    FY25 prices are derived from FY24 using configurable deltas.
    Supports both service-level and app-level overrides.
    """
    prices: List[ServicePrice] = []

    for s in SERVICES:
        sid = s["service_id"]
        unit = f"{CURRENCY}/unit"

        # --- FY24 Base Price ---
        p24 = BASE_PRICES_FY24[sid]
        prices.append(ServicePrice(
            fiscal_year="FY24",
            service_id=sid,
            price=p24,
            unit=unit,
        ))

        # --- Determine applicable delta ---
        # 1. Check if any app mapped to this service has a custom delta
        app_overrides = [
            aid for aid, mapped_sid in APP_SERVICE_MAP.items()
            if mapped_sid == sid and aid in CUSTOM_PRICE_DELTAS
        ]

        if app_overrides:
            delta = CUSTOM_PRICE_DELTAS[app_overrides[0]]  # take first matching app delta
        else:
            delta = DEFAULT_PRICE_DELTAS.get(sid, 0.0)

        # --- FY25 Derived Price ---
        p25 = round(p24 * (1.0 + delta), 4)
        prices.append(ServicePrice(
            fiscal_year="FY25",
            service_id=sid,
            price=p25,
            unit=unit,
        ))

    return prices