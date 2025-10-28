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
    """Build yearly CHANGE (project) facts, allocate per BU and country proportional to RUN usage,
    and apply rounding correction once per FY (in month 12) to the country with the highest usage."""

    weights = [1 / 12.0] * 12  # flat monthly distribution
    change_rows: List[FactChangeRow] = []

    # --- RUN usage lookup per (BU, country) ---
    run_usage: Dict[tuple, float] = {}
    for r in run_rows:
        key = (r.org_id, r.country_code)
        run_usage[key] = run_usage.get(key, 0.0) + float(r.runCost or 0.0)

    # yearly accumulator for rounding correction
    yearly_accumulator: Dict[tuple, float] = {}

    for p in projects:
        for fy in ["FY24", "FY25"]:
            if not ((fy == "FY24" and p.exists_fy24) or (fy == "FY25" and p.exists_fy25)):
                continue

            total_year_cost = p.cost_fy24 if fy == "FY24" else p.cost_fy25
            if total_year_cost == 0:
                continue

            for alloc in p.allocation:
                org_id = alloc.org_id
                share_bu = alloc.share
                # Anteil des Jahresbudgets für diese BU
                bu_budget = total_year_cost * share_bu

                # --- Länderanteile pro BU aus RUN ableiten ---
                # filtere alle RUN-Zeilen dieser BU
                bu_countries = {cc: cost for (org, cc), cost in run_usage.items() if org == org_id}
                total_bu_run = sum(bu_countries.values())

                if total_bu_run <= 0:
                    continue

                # Land mit höchstem Verbrauch (für Korrektur am Jahresende)
                max_country = max(bu_countries.items(), key=lambda kv: kv[1])[0]

                # Monatsverteilung
                for month_idx, weight in enumerate(weights, start=1):
                    month_amount = bu_budget * weight
                    month_split = {}
                    sum_month = 0.0

                    for country_code, run_cost in bu_countries.items():
                        share_ctry = run_cost / total_bu_run
                        amt = round(month_amount * share_ctry, 2)
                        month_split[country_code] = amt
                        sum_month += amt

                    # Sammle Jahressumme pro BU/Land
                    for country_code, amt in month_split.items():
                        key = (p.project_id, fy, org_id, country_code)
                        yearly_accumulator[key] = yearly_accumulator.get(key, 0.0) + amt

                        change_rows.append(
                            FactChangeRow(
                                fiscal_year=fy,
                                fiscal_month_num=month_idx,
                                project_id=p.project_id,
                                tower_id="TWR-06",
                                service_id="SRV-CHG",
                                org_id=org_id,
                                country_code=country_code,
                                cost_center_id="N/A",
                                app_id=None,
                                quantity=None,
                                project_cost=amt,
                            )
                        )

                    # --- End-of-year rounding correction (month 12 only) ---
                    if month_idx == 12:
                        # berechne geplante Jahressumme für diese BU
                        planned_bu = round(bu_budget, 2)
                        actual_bu = round(sum(
                            v for (pid, fyy, org, _), v in yearly_accumulator.items()
                            if pid == p.project_id and fyy == fy and org == org_id
                        ), 2)
                        delta = round(planned_bu - actual_bu, 2)

                        if abs(delta) >= 0.01:
                            print(
                                f"[Rounding correction] Project {p.project_id} {fy} | "
                                f"BU={org_id} | Country={max_country} | Delta={delta:+.2f}"
                            )
                            # Füge Delta dem Land mit dem höchsten Verbrauch hinzu
                            change_rows.append(
                                FactChangeRow(
                                    fiscal_year=fy,
                                    fiscal_month_num=12,
                                    project_id=p.project_id,
                                    tower_id="TWR-06",
                                    service_id="SRV-CHG",
                                    org_id=org_id,
                                    country_code=max_country,
                                    cost_center_id="N/A",
                                    app_id=None,
                                    quantity=None,
                                    project_cost=delta,
                                )
                            )

    return change_rows



