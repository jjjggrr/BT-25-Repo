from __future__ import annotations
from typing import List, Dict, Tuple
from collections import defaultdict
from .models import FactRunRow, FactChangeRow, ServicePrice, ProjectDef
from .config import FISCAL_YEARS, FISCAL_MONTHS, PROJECT_MONTHLY_DISTRIBUTION


def _price_lookup(prices: List[ServicePrice]) -> Dict[Tuple[str, str], ServicePrice]:
    return {(p.service_id, p.fiscal_year): p for p in prices}


def build_run_facts(quantities: List[Dict], prices: List[ServicePrice]) -> List[FactRunRow]:
    pmap = _price_lookup(prices)
    rows: List[FactRunRow] = []
    for q in quantities:
        key = (q["service_id"], q["fiscal_year"])
        if key not in pmap:
            raise KeyError(f"Missing price for {key}")
        price = pmap[key].price
        run_cost = round(price * float(q["quantity"]), 4)
        rows.append(FactRunRow(
            fiscal_year=q["fiscal_year"],
            fiscal_month_num=int(q["fiscal_month_num"]),
            tower_id=q["tower_id"],
            service_id=q["service_id"],
            org_id=q["org_id"],
            price=float(price),
            quantity=float(q["quantity"]),
            runCost=run_cost,
        ))
    return rows


def _monthly_weights(mode: str) -> List[float]:
    if mode == "flat":
        return [1.0/12.0]*12
    if mode == "frontloaded":
        base = [13-i for i in range(1,13)]
    elif mode == "backloaded":
        base = [i for i in range(1,13)]
    else:
        base = [1]*12
    s = float(sum(base))
    return [b/s for b in base]


def build_change_facts(projects: List[ProjectDef]) -> List[FactChangeRow]:
    weights = _monthly_weights(PROJECT_MONTHLY_DISTRIBUTION)
    rows: List[FactChangeRow] = []
    for p in projects:
        for fy in FISCAL_YEARS:
            exists = (fy == "FY24" and p.exists_fy24) or (fy == "FY25" and p.exists_fy25)
            if not exists:
                continue
            total = p.cost_fy24 if fy == "FY24" else p.cost_fy25
            # Distribute to BUs (stable allocation)
            for alloc in p.allocation:
                bu_amount = total * alloc.share
                # Spread over months
                for m, w in enumerate(weights, start=1):
                    amt = round(bu_amount * w, 4)
                    rows.append(FactChangeRow(
                        fiscal_year=fy,
                        fiscal_month_num=m,
                        project_id=p.project_id,
                        tower_id="",  # optional: map project->tower/service if desired
                        service_id=None,
                        org_id=alloc.org_id,
                        project_cost=amt,
                    ))
    return rows