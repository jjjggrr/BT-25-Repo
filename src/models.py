from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Literal


# === Data Records ===

class ServicePrice(BaseModel):
    fiscal_year: str
    service_id: str
    price: float
    unit: str  # e.g., "EUR/unit"


class ProjectAllocation(BaseModel):
    org_id: str
    share: float


class ProjectDef(BaseModel):
    project_id: str
    name: str
    exists_fy24: bool
    exists_fy25: bool
    cost_fy24: float
    cost_fy25: float
    allocation: List[ProjectAllocation]

    @field_validator("allocation")
    @classmethod
    def _shares_sum_to_one(cls, v: List[ProjectAllocation]):
        s = sum(a.share for a in v)
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"Allocation shares must sum to 1.0, got {s}")
        return v


class FactRunRow(BaseModel):
    fiscal_year: str
    fiscal_month_num: int
    tower_id: str
    service_id: str
    org_id: str
    country_code: str  # NEU
    app_id: str
    cost_center_id: str
    price: float
    quantity: float
    runCost: float


class FactChangeRow(BaseModel):
    fiscal_year: str
    fiscal_month_num: int
    project_id: str
    tower_id: str
    service_id: Optional[str] = None
    org_id: str
    country_code: str  # <-- NEU
    cost_center_id: str  # <-- NEU
    app_id: Optional[str] = None  # <-- NEU
    quantity: Optional[float] = None  # <-- NEU
    project_cost: float


# === Embedding Metadata ===
DocType = Literal["service_agreement", "project_brief"]
Granularity = Literal["tower", "service", "bu", "project"]


class ServiceDocMeta(BaseModel):
    doc_type: Literal["service_agreement"] = "service_agreement"
    fiscal_year: str
    tower_id: str
    service_id: str
    org_id: Optional[str] = None
    granularity: Granularity = "service"
    # Claims
    price_fy: Optional[float] = None
    price_unit: Optional[str] = None
    price_delta_pct_vs_prev_fy: Optional[float] = None
    # Tracing
    source_pdf_name: Optional[str] = None
    source_pdf_version: Optional[str] = None
    generated_at_utc: Optional[str] = None
    validated_against_cube: Optional[bool] = None
    validation_report_id: Optional[str] = None


class ProjectDocMeta(BaseModel):
    doc_type: Literal["project_brief"] = "project_brief"
    fiscal_year: str
    project_id: str
    granularity: Granularity = "project"
    # Claims
    project_cost_fy24: Optional[float] = None
    project_cost_fy25: Optional[float] = None
    is_new_in_fy25: Optional[bool] = None
    allocation_vector: Optional[List[ProjectAllocation]] = None
    # Tracing
    source_pdf_name: Optional[str] = None
    source_pdf_version: Optional[str] = None
    generated_at_utc: Optional[str] = None
    validated_against_cube: Optional[bool] = None
    validation_report_id: Optional[str] = None
