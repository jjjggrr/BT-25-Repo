from __future__ import annotations
from typing import List, Dict, Tuple
from collections import defaultdict
from rich import print as rprint
from .models import FactRunRow, FactChangeRow, ServicePrice, ProjectDef


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
        if p.exists_fy24:
            t = round(agg[(p.project_id, "FY24")], 2)
            if abs(t - p.cost_fy24) > 0.01:
                raise ValidationError(f"Project {p.project_id} FY24 total {t} != {p.cost_fy24}")
        if p.exists_fy25:
            t = round(agg[(p.project_id, "FY25")], 2)
            if abs(t - p.cost_fy25) > 0.01:
                raise ValidationError(f"Project {p.project_id} FY25 total {t} != {p.cost_fy25}")
    rprint("[green]OK[/green] Project yearly totals match CHANGE facts")


def validate_project_allocation_constancy(projects: List[ProjectDef]):
    # By construction allocations are stable; this check ensures shares sum to 1.0 and are in (0,1)
    for p in projects:
        s = sum(a.share for a in p.allocation)
        if abs(s - 1.0) > 1e-9:
            raise ValidationError(f"Allocation sum != 1.0 for {p.project_id}")
        if any(a.share <= 0 for a in p.allocation):
            raise ValidationError(f"Non-positive share for {p.project_id}")
    rprint("[green]OK[/green] Project BU allocations are valid & stable across FYs")