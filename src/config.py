from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import random

# === Paths (Windows-friendly) ===
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "out"
PDF_DIR = DATA_DIR / "pdf"
EMB_DIR = DATA_DIR / "embeddings"  # optional (Chroma will create its own .chromadb)

OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)

# === Random seed for reproducibility ===
RANDOM_SEED = 42

# === Fiscal Years (FY starts in October) ===
FISCAL_YEARS: List[str] = ["FY24", "FY25"]
FISCAL_MONTHS = list(range(1, 13))  # 1..12 TBM month index (Oct=1, Nov=2, ..., Sep=12)

CURRENCY = "EUR"

# Price is constant per (service_id, FY) – critical requirement
FY_PRICES_LOCK = True

# === Minimal master data (MVP) ===
TOWERS = [
    {"tower_id": "TWR-01", "tower_name": "Workplace"},
    {"tower_id": "TWR-02", "tower_name": "Hosting"},
]

SERVICES = [
    {"service_id": "SRV-DEV", "tower_id": "TWR-01", "service_name": "Device Management", "unit": "device/month"},
    {"service_id": "SRV-CLB", "tower_id": "TWR-01", "service_name": "Collaboration", "unit": "user/month"},
    {"service_id": "SRV-HOS", "tower_id": "TWR-02", "service_name": "Hosting", "unit": "vm/month"},
]

# Define 15 business units
BUSINESS_UNITS = [
    {"org_id": f"BU_{i:02d}", "business_unit_name": f"Business Unit {i:02d}"}
    for i in range(1, 11)
]

# Countries per BU
COUNTRIES = [
    "DE", "FR", "IT", "ES", "NL", "PL", "US", "CA", "MX", "BR"
]

# Expanded BU-country combinations
BU_COUNTRIES = []
for bu in BUSINESS_UNITS:
    # jede BU in 3–10 Ländern aktiv
    active = random.sample(COUNTRIES, k=random.randint(3, 10))
    for c in active:
        BU_COUNTRIES.append({
            "org_id": bu["org_id"],
            "country_code": c,
            "business_unit_name": bu["business_unit_name"],
        })

# Projects – some exist across FYs, some are new in FY25
PROJECTS = [
    {"project_id": "PRJ-A", "project_name": "Modernize Endpoint", "exists_fy24": True, "exists_fy25": True},
    {"project_id": "PRJ-B", "project_name": "Data Center Refresh", "exists_fy24": True, "exists_fy25": True},
    {"project_id": "PRJ-C", "project_name": "New Collaboration Suite", "exists_fy24": False, "exists_fy25": True},
]

# Default price deltas YOY (per service) for FY25 vs FY24 (can be pos/neg)
DEFAULT_PRICE_DELTAS = {
    "SRV-DEV": -0.03,  # -3% productivity
    "SRV-CLB": 0.00,  # flat
    "SRV-HOS": 0.02,  # +2%
}

# RUN quantity scale per BU to shape relative sizes
RUN_QUANTITY_BASE = {
    "BU-EMEA": 1.0,
    "BU-AMER": 0.7,
}

# Monthly variability amplitude (± fraction around BU base)
RUN_MONTHLY_NOISE = 0.05

# CHANGE (projects) yearly budgets (EUR) – generator will split to BUs via fixed shares
PROJECT_BUDGETS = {
    "PRJ-A": {"FY24": 800_000.0, "FY25": 650_000.0},
    "PRJ-B": {"FY24": 500_000.0, "FY25": 700_000.0},
    "PRJ-C": {"FY24": 0.0, "FY25": 450_000.0},
}

# === SCALE-UP PARAMETERS ===
NUM_COST_CENTERS = 50
NUM_APPS = 20

DIM_COST_CENTERS = [
    {"cost_center_id": f"CC-{i:04d}", "cost_center_name": f"Cost Center {i:04d}"}
    for i in range(1, NUM_COST_CENTERS + 1)
]

DIM_APPS = [
    {"app_id": f"APP-{i:04d}", "app_name": f"Application {i:04d}"}
    for i in range(1, NUM_APPS + 1)
]

# Allocation policy across BUs (stable across FYs for projects that exist across years)
# If not provided for a project, generator will draw a stable random vector.
DEFAULT_PROJECT_ALLOCATIONS: Dict[str, Dict[str, float]] = {
    # Example fixed split for PRJ-A
    # "PRJ-A": {"BU-EMEA": 0.6, "BU-AMER": 0.4}
}

# Distribution policy for project costs over months (flat monthly by default)
PROJECT_MONTHLY_DISTRIBUTION = "flat"  # ["flat" | "frontloaded" | "backloaded"]
