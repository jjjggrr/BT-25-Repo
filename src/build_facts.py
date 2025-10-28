from __future__ import annotations
from collections import defaultdict
from typing import List, Dict
from .models import FactRunRow, FactChangeRow, ServicePrice, ProjectDef
from .config import FISCAL_YEARS, FISCAL_MONTHS, PROJECT_MONTHLY_DISTRIBUTION, COST_CENTER_BY_COUNTRY_AND_ORG


def _price_lookup(prices: List[ServicePrice]):
    return {(p.service_id, p.fiscal_year): p for p in prices}

def _aggregate_run_qty_by_fy_org_country(run_rows: List) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Aggregiert RUN-Quantities zu: agg[fy][org_id][country_code] = total_quantity
    Erwartet run_rows-Objekte / Dicts mit keys: fiscal_year, org_id, country_code, quantity.
    """
    agg: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for r in run_rows:
        fy = r.fiscal_year
        org = r.org_id
        ctry = r.country_code
        qty = float(r.quantity or 0.0)
        if qty <= 0:
            continue
        agg[fy][org][ctry] += qty
    return agg

def build_run_facts(quantities: List[Dict], prices: List[ServicePrice]) -> List[FactRunRow]:
    pmap = _price_lookup(prices)
    rows: List[FactRunRow] = []
    for q in quantities:
        key = (q["service_id"], q["fiscal_year"])
        if key not in pmap:
            raise KeyError(f"Missing price for {key}")
        price = float(pmap[key].price)
        qty = float(q["quantity"])
        run_cost = round(price * qty, 4)
        rows.append(FactRunRow(
            fiscal_year=q["fiscal_year"],
            fiscal_month_num=int(q["fiscal_month_num"]),
            tower_id=q["tower_id"],
            service_id=q["service_id"],
            org_id=q["org_id"],
            country_code=q["country_code"],
            app_id=q["app_id"],
            cost_center_id=q["cost_center_id"],
            price=price,
            quantity=qty,
            runCost=run_cost,
        ))
    return rows


def _monthly_weights(fiscal_year: str, mode: str) -> List[float]:
    """
    Returns list of 12 weights for distributing yearly budget across months.
    Now fiscal_year-aware:
      - FY24: heavy backloaded (70% last 4 months, final month peak)
      - FY25+: moderate backloaded (40% last 4 months)
    """
    if mode == "late-heavy":
        if fiscal_year == "FY24":
            # Heavy backloaded: 30% in first 8, 70% in last 4 (10%, 15%, 20%, 25%)
            first8 = [0.30 / 8.0] * 8
            last4 = [0.10, 0.15, 0.20, 0.25]
            return first8 + last4
        else:
            # FY25 and later: moderate backloaded (60/40)
            first8 = [0.60 / 8.0] * 8  # = 7.5% each
            last4 = [0.10, 0.10, 0.10, 0.10]  # 40%
            return first8 + last4

    # Fallback (old behavior)
    if mode == "flat":
        return [1.0 / 12.0] * 12

    raise ValueError(f"Unknown PROJECT_MONTHLY_DISTRIBUTION mode: {mode}")


def build_change_facts(projects: List[ProjectDef], run_rows: List[FactRunRow]) -> List[FactChangeRow]:
    """
    Builds CHANGE facts (project costs) with TBM-correct logic:
      - Tower = TWR-06 (Change)
      - Service = SRV-CHG
      - CHANGE is not App-based (app_id=None)
      - CHANGE has no unit-model (quantity=None)
      - Project costs are allocated per FY proportional to RUN consumption in the *same* FY
      - Only countries where the BU (org_id) actually has RUN in that FY receive costs
      - CostCenter is derived exactly as in RUN via (country, org) mapping
      - FY24: if a BU has no RUN in FY24 → no share of FY24 project costs
      - FY25: if the BU has RUN in FY25 → gets share in FY25
      - Monthly weighting is FY-dependent (late-heavy FY24, moderate FY25+)
    """

    from .config import FISCAL_YEARS, PROJECT_MONTHLY_DISTRIBUTION, COST_CENTER_BY_COUNTRY_AND_ORG
    from .models import FactChangeRow

    # --- 1) aggregate RUN consumption by FY / ORG / COUNTRY ---
    run_qty = _aggregate_run_qty_by_fy_org_country(run_rows)

    change_rows: List[FactChangeRow] = []

    # --- 2) iterate projects ---
    for p in projects:
        for fy in FISCAL_YEARS:

            # project exists in this FY?
            exists = (fy == "FY24" and p.exists_fy24) or (fy == "FY25" and p.exists_fy25)
            if not exists:
                continue

            # total capex for this FY
            total_fy = p.cost_fy24 if fy == "FY24" else p.cost_fy25
            if total_fy <= 0:
                continue

            # monthly weights (late-heavy in FY24, moderate in FY25+)
            weights = _monthly_weights(fy, PROJECT_MONTHLY_DISTRIBUTION)

            # --- 3) each Org allocation (BU share) ---
            for alloc in p.allocation:  # allocation is still list[dict]
                org_id = alloc.org_id
                bu_share = float(alloc.share)
                if bu_share <= 0:
                    continue

                bu_year_amount = total_fy * bu_share

                # --- 4) does this BU have ANY consumption in this FY? ---
                org_ctry_qty = run_qty.get(fy, {}).get(org_id, {})
                if not org_ctry_qty:
                    # no consumption → no share of project in this FY
                    continue

                qty_total = sum(org_ctry_qty.values())
                if qty_total <= 0:
                    continue

                # --- 5) allocate per month ---
                for m, w in enumerate(weights, start=1):
                    month_amount = bu_year_amount * w
                    if month_amount <= 0:
                        continue

                    # --- 6) split per country proportional to qty share ---
                    for country_code, qty in org_ctry_qty.items():
                        if qty <= 0:
                            continue

                        share = qty / qty_total
                        amt = round(month_amount * share, 2)
                        if amt <= 0:
                            continue

                        cc_id = COST_CENTER_BY_COUNTRY_AND_ORG.get((country_code, org_id))
                        if not cc_id:
                            continue  # safety, should not happen

                        # --- 7) create FactChangeRow ---
                        change_rows.append(
                            FactChangeRow(
                                fiscal_year=fy,
                                fiscal_month_num=m,
                                project_id=p.project_id,
                                tower_id="TWR-06",
                                service_id="SRV-CHG",
                                org_id=org_id,
                                country_code=country_code,
                                cost_center_id=cc_id,
                                app_id=None,     # change != app
                                quantity=None,   # change != unit-based
                                project_cost=amt,
                            )
                        )

    return change_rows


