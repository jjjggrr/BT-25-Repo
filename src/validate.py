from __future__ import annotations
from typing import List, Dict, Tuple
from collections import defaultdict
from rich import print as rprint
from .models import FactRunRow, FactChangeRow, ProjectDef


class ValidationError(Exception):
    pass


def validate_price_constancy(run_rows: List[FactRunRow]):
    # Ensure for a given (service_id, fiscal_year) there is only one distinct price
    seen: Dict[Tuple[str, str], float] = {}
    for r in run_rows:
        key = (r.service_id, r.fiscal_year)
        if key in seen and abs(seen[key] - r.price) > 1e-9:
            raise ValidationError(f"Price drift within FY for {key}: {seen[key]} vs {r.price}")
        seen.setdefault(key, r.price)
    rprint("[green]OK[/green] Price constancy per (service, FY)")


def validate_pxq(run_rows: List[FactRunRow]):
    for r in run_rows:
        expected = round(r.price * r.quantity, 4)
        if abs(expected - r.runCost) > 1e-6:
            raise ValidationError(f"PxQ mismatch: {r.service_id} {r.fiscal_year} m{r.fiscal_month_num} got {r.runCost} expected {expected}")
    rprint("[green]OK[/green] PxQ holds for RUN rows")


def validate_project_totals(change_rows: List[FactChangeRow], projects: List[ProjectDef]):
    # Sum per (project_id, FY)
    agg: Dict[Tuple[str, str], float] = defaultdict(float)
    for c in change_rows:
        agg[(c.project_id, c.fiscal_year)] += c.project_cost

    for p in projects:
        # dynamisch über alle FYs, die in agg für dieses Projekt existieren
        for (proj_id, fy), total in agg.items():
            if proj_id != p.project_id:
                continue

            # hole die geplante kosten für dieses FY vom Projekt
            planned = 0.0
            if fy == "FY24":
                planned = p.cost_fy24
            elif fy == "FY25":
                planned = p.cost_fy25
            else:
                # falls später FY26/FY27 etc. ins model kommen → ignorieren wir hier erstmal statt crash
                continue

            total = round(total, 2)
            planned = round(planned, 2)
            if abs(total - planned) > 0.02:
                raise ValidationError(
                    f"Project {p.project_id} {fy} total {total} != {planned}"
                )

    rprint("[green]OK[/green] Project yearly totals match CHANGE facts (within tolerance)")
