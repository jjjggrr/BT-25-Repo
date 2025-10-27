from __future__ import annotations
import random
from typing import Dict, List
from .config import SERVICES, FISCAL_YEARS, DEFAULT_PRICE_DELTAS, RANDOM_SEED, CURRENCY
from .models import ServicePrice

random.seed(RANDOM_SEED)

# Base list price per service for FY24 (seeded for deterministic output)
BASE_PRICES_FY24 = {
    s["service_id"]: round(random.uniform(4.0, 8.0), 2) for s in SERVICES
}


def gen_service_prices() -> List[ServicePrice]:
    prices: List[ServicePrice] = []
    for s in SERVICES:
        sid = s["service_id"]
        unit = f"{CURRENCY}/unit"
        # FY24
        p24 = BASE_PRICES_FY24[sid]
        prices.append(ServicePrice(fiscal_year="FY24", service_id=sid, price=p24, unit=unit))
        # FY25 based on delta
        delta = DEFAULT_PRICE_DELTAS.get(sid, 0.0)
        p25 = round(p24 * (1.0 + delta), 4)
        prices.append(ServicePrice(fiscal_year="FY25", service_id=sid, price=p25, unit=unit))
    # Future FYs can be added similarly
    return prices