from __future__ import annotations
import random
from typing import List, Dict
from .config import (
    PROJECTS, PROJECT_BUDGETS, BUSINESS_UNITS, DEFAULT_PROJECT_ALLOCATIONS,
    RANDOM_SEED,
)
from .models import ProjectDef, ProjectAllocation

random.seed(RANDOM_SEED)


def _stable_allocation_for_project(project_id: str) -> List[ProjectAllocation]:
    # Use provided default if exists
    if project_id in DEFAULT_PROJECT_ALLOCATIONS:
        alloc_map = DEFAULT_PROJECT_ALLOCATIONS[project_id]
        shares = [ProjectAllocation(org_id=k, share=float(v)) for k, v in alloc_map.items()]
        s = sum(a.share for a in shares)
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"Allocation shares must sum to 1.0 for {project_id}")
        return shares

    # Otherwise draw a stable random vector across BUs
    weights = [random.random() for _ in BUSINESS_UNITS]
    total = sum(weights)
    shares = []
    for w, bu in zip(weights, BUSINESS_UNITS):
        shares.append(ProjectAllocation(org_id=bu["org_id"], share=w/total))
    return shares


def gen_projects() -> List[ProjectDef]:
    out: List[ProjectDef] = []
    for p in PROJECTS:
        pid = p["project_id"]
        budgets = PROJECT_BUDGETS.get(pid, {"FY24": 0.0, "FY25": 0.0})
        alloc = _stable_allocation_for_project(pid)
        out.append(ProjectDef(
            project_id=pid,
            name=p["project_name"],
            exists_fy24=p["exists_fy24"],
            exists_fy25=p["exists_fy25"],
            cost_fy24=float(budgets.get("FY24", 0.0)),
            cost_fy25=float(budgets.get("FY25", 0.0)),
            allocation=alloc,
        ))
    return out