from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

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
random.seed(RANDOM_SEED)

# === Fiscal Years (FY starts in October) ===
FISCAL_YEARS: List[str] = ["FY24", "FY25"]
FISCAL_MONTHS = list(range(1, 13))  # 1..12 TBM month index (Oct=1, Nov=2, ..., Sep=12)

CURRENCY = "EUR"

# Price is constant per (service_id, FY) – critical requirement
FY_PRICES_LOCK = True

# === Minimal master data (keine Änderung an Services/Towers) ===
TOWERS = [
    {"tower_id": "TWR-01", "tower_name": "Workplace"},
    {"tower_id": "TWR-02", "tower_name": "Hosting / Compute"},
    {"tower_id": "TWR-03", "tower_name": "Network"},
    {"tower_id": "TWR-04", "tower_name": "Security"},
    {"tower_id": "TWR-05", "tower_name": "Applications"},
    {"tower_id": "TWR-06", "tower_name": "Change"},
]


SERVICES = [
    # Workplace
    {"service_id": "SRV-DEV",  "tower_id": "TWR-01", "service_name": "Device Management",           "unit": "device/month"},
    {"service_id": "SRV-CLB",  "tower_id": "TWR-01", "service_name": "Collaboration",               "unit": "user/month"},
    {"service_id": "SRV-DWP",  "tower_id": "TWR-01", "service_name": "Digital Workplace",           "unit": "user/month"},
    {"service_id": "SRV-MDM",  "tower_id": "TWR-01", "service_name": "Mobile Device Management",    "unit": "device/month"},

    # Hosting / Compute
    {"service_id": "SRV-HOS",  "tower_id": "TWR-02", "service_name": "Compute Hosting (VM)",        "unit": "vm/month"},
    {"service_id": "SRV-DBH",  "tower_id": "TWR-02", "service_name": "Database Hosting",            "unit": "db/month"},
    {"service_id": "SRV-CSP",  "tower_id": "TWR-02", "service_name": "Cloud Platform",              "unit": "acct/month"},
    {"service_id": "SRV-INT",  "tower_id": "TWR-02", "service_name": "Integration Platform",        "unit": "flow/month"},

    # Network
    {"service_id": "SRV-NET",  "tower_id": "TWR-03", "service_name": "LAN/WAN Connectivity",        "unit": "site/month"},
    {"service_id": "SRV-NMS",  "tower_id": "TWR-03", "service_name": "Network Management",          "unit": "device/month"},

    # Security
    {"service_id": "SRV-IAM",  "tower_id": "TWR-04", "service_name": "Identity & Access Mgmt",      "unit": "user/month"},
    {"service_id": "SRV-SEC",  "tower_id": "TWR-04", "service_name": "Security Platform",           "unit": "asset/month"},

    # Applications
    {"service_id": "SRV-ERP-FIN", "tower_id": "TWR-05", "service_name": "ERP Finance Platform",     "unit": "instance/month"},
    {"service_id": "SRV-ERP-SLS", "tower_id": "TWR-05", "service_name": "ERP Sales Platform",       "unit": "instance/month"},
    {"service_id": "SRV-ERP-PRC", "tower_id": "TWR-05", "service_name": "ERP Procurement Platform", "unit": "instance/month"},
    {"service_id": "SRV-HCM",     "tower_id": "TWR-05", "service_name": "HCM Platform",             "unit": "instance/month"},
    {"service_id": "SRV-CRM",     "tower_id": "TWR-05", "service_name": "CRM Platform",             "unit": "instance/month"},
    {"service_id": "SRV-ANL",     "tower_id": "TWR-05", "service_name": "Analytics / BI Platform",  "unit": "instance/month"},
    {"service_id": "SRV-DATA",    "tower_id": "TWR-05", "service_name": "Enterprise Data Platform", "unit": "instance/month"},
    {"service_id": "SRV-ITSM",    "tower_id": "TWR-05", "service_name": "ITSM Platform",            "unit": "instance/month"},
    {"service_id": "SRV-INTR",    "tower_id": "TWR-05", "service_name": "Intranet / Portal",        "unit": "instance/month"},
    {"service_id": "SRV-MES",     "tower_id": "TWR-05", "service_name": "Manufacturing Execution",  "unit": "instance/month"},
    {"service_id": "SRV-ALM",     "tower_id": "TWR-05", "service_name": "ALM / Dev Collaboration",  "unit": "instance/month"},

    # Change
    {"service_id": "SRV-CHG", "tower_id": "TWR-06", "service_name": "Change", "unit": ""}
]

APP_SERVICE_MAP: Dict[str, str] = {
    "APP-0001": "SRV-ERP-FIN",   # SAP S/4HANA Finance
    "APP-0002": "SRV-ERP-SLS",   # SAP S/4HANA Sales
    "APP-0003": "SRV-ERP-PRC",   # SAP S/4HANA Procurement
    "APP-0004": "SRV-HCM",       # SAP SuccessFactors
    "APP-0005": "SRV-CLB",       # Microsoft 365
    "APP-0006": "SRV-CLB",       # Microsoft Teams
    "APP-0007": "SRV-INTR",      # SharePoint Online (Intranet/Portal)
    "APP-0008": "SRV-IAM",       # Azure AD
    "APP-0009": "SRV-ITSM",      # ServiceNow
    "APP-0010": "SRV-CRM",       # Salesforce
    "APP-0011": "SRV-HCM",       # Workday HCM
    "APP-0012": "SRV-ALM",       # Atlassian Jira
    "APP-0013": "SRV-ALM",       # Atlassian Confluence
    "APP-0014": "SRV-ALM",       # GitLab Enterprise
    "APP-0015": "SRV-DBH",       # Oracle Database
    "APP-0016": "SRV-INT",       # IBM MQ / Integration
    "APP-0017": "SRV-NMS",       # Cisco NMS
    "APP-0018": "SRV-SEC",       # Palo Alto Security
    "APP-0019": "SRV-DATA",      # Snowflake
    "APP-0020": "SRV-ANL",       # Tableau
    "APP-0021": "SRV-DATA",      # Enterprise Data Lake
    "APP-0022": "SRV-IAM",       # IAM Core
    "APP-0023": "SRV-INT",       # API Gateway
    "APP-0024": "SRV-MES",       # Manufacturing Execution
    "APP-0025": "SRV-ERP-PRC",   # Procurement Hub
    "APP-0026": "SRV-ANL",       # Analytics & Reporting Hub
    "APP-0027": "SRV-DWP",       # DWP
    "APP-0028": "SRV-MDM",       # Mobile Device Mgmt
    "APP-0029": "SRV-INTR",      # Corporate Intranet Portal
    "APP-0030": "SRV-INT",       # Data Integration Hub
}

SERVICE_ALLOWED_APPS: Dict[str, List[str]] = defaultdict(list)
for app_id, srv in APP_SERVICE_MAP.items():
    SERVICE_ALLOWED_APPS[srv].append(app_id)
SERVICE_ALLOWED_APPS = dict(SERVICE_ALLOWED_APPS)




# === Business Units (ORG_001 … ORG_010 mit Realnamen) ===
BUSINESS_UNITS = [
    {"org_id": "ORG_001", "business_unit_name": "Corporate"},
    {"org_id": "ORG_002", "business_unit_name": "Finance & Controlling"},
    {"org_id": "ORG_003", "business_unit_name": "Human Resources"},
    {"org_id": "ORG_004", "business_unit_name": "IT & Digital"},
    {"org_id": "ORG_005", "business_unit_name": "Procurement"},
    {"org_id": "ORG_006", "business_unit_name": "Supply Chain & Logistics"},
    {"org_id": "ORG_007", "business_unit_name": "Manufacturing Operations"},
    {"org_id": "ORG_008", "business_unit_name": "Sales & Marketing"},
    {"org_id": "ORG_009", "business_unit_name": "Research & Development"},
    {"org_id": "ORG_010", "business_unit_name": "Customer Services"},
]

# === Countries (fixe 10er-Liste) ===
COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "PL", "US", "CA", "MX", "BR", "CN"]

COUNTRY_NAMES_EN = {
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "NL": "Netherlands",
    "PL": "Poland",
    "US": "United States",
    "CA": "Canada",
    "MX": "Mexico",
    "BR": "Brazil",
    "CN": "China",
    "AR": "Argentina",
}

COUNTRY_REGIONS = {
    "DE": "EMEA",
    "FR": "EMEA",
    "IT": "EMEA",
    "ES": "EMEA",
    "NL": "EMEA",
    "PL": "EMEA",
    "US": "AMERICAS",
    "CA": "AMERICAS",
    "MX": "AMERICAS",
    "BR": "AMERICAS",
    "AR": "AMERICAS",
    "CN": "APAC"
}


# === BU → Country Verfügbarkeit (C2: realistisch, deterministisch)
BU_COUNTRY_MATRIX: Dict[str, List[str]] = {
    "ORG_001": ["DE","FR","IT","ES","NL","PL","US","CA","MX","BR"],  # Corporate
    "ORG_002": ["DE","FR","IT","ES","NL","PL","US","CA","MX","BR"],  # Finance
    "ORG_003": ["DE","FR","IT","ES","NL","PL","US","CA","MX","BR"],  # HR
    "ORG_004": ["DE","FR","IT","ES","NL","PL","US","CA","MX","BR"],  # IT & Digital
    "ORG_005": ["DE","FR","IT","ES","NL","PL","US","CA","MX","BR"],  # Procurement
    "ORG_006": ["DE","IT","PL","MX","BR","NL","ES","CN"],                 # Supply Chain & Logistics
    "ORG_007": ["DE","IT","PL","MX","BR", "CN", "AR"],                           # Manufacturing
    "ORG_008": ["DE","FR","ES","NL","US","CA","MX","BR"],            # Sales & Marketing
    "ORG_009": ["DE","US","IT","NL"],                                # R&D (kein PL/ES/FR/CA/MX/BR)
    "ORG_010": ["DE","FR","ES","IT","US","CA","MX","BR"],            # Customer Services
}

# Für Abwärtskompatibilität: expandierte BU-Land-Zuordnung als Liste von Dicts
BU_COUNTRIES: List[Dict[str, str]] = [
    {"org_id": org, "country_code": c, "business_unit_name": next(bu["business_unit_name"] for bu in BUSINESS_UNITS if bu["org_id"] == org)}
    for org, clist in BU_COUNTRY_MATRIX.items()
    for c in clist
]

# === Projekte (unverändert, aber deterministisch) ===
PROJECTS = [
    {"project_id": "PRJ-A", "project_name": "Modernize Endpoint", "exists_fy24": True, "exists_fy25": True},
    {"project_id": "PRJ-B", "project_name": "Data Center Refresh", "exists_fy24": True, "exists_fy25": True},
    {"project_id": "PRJ-C", "project_name": "New Collaboration Suite", "exists_fy24": False, "exists_fy25": True},
]

DEFAULT_PRICE_DELTAS = {
    "SRV-DEV": -0.03,  # -3%
    "SRV-CLB": 0.00,   # flat
    "SRV-HOS": 0.02,   # +2%
}

# === Custom Overrides (per App) ===

# Preisänderungen (z. B. pro App statt pro Service)
CUSTOM_PRICE_DELTAS: Dict[str, float] = {
    "APP-0005": 0.05,  # Microsoft 365 +5 %
    "APP-0006": 0.05,  # Microsoft Teams +5 %
}

# Mengenmultiplikatoren (FY25-spezifisch)
CUSTOM_QUANTITY_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "APP-0005": {"FY25": 1.10},  # Microsoft 365 +10 % Volumen
    "APP-0006": {"FY25": 1.10},  # Microsoft Teams +10 % Volumen
}


# RUN quantity baseline: per org optional; default = 1.0
RUN_QUANTITY_BASE: Dict[str, float] = {}  # leer → alle ORGs = 1.0
RUN_MONTHLY_NOISE = 0.05

PROJECT_BUDGETS = {
    "PRJ-A": {"FY24": 800_000.0, "FY25": 650_000.0},
    "PRJ-B": {"FY24": 500_000.0, "FY25": 700_000.0},
    "PRJ-C": {"FY24": 0.0, "FY25": 450_000.0},
}

DEFAULT_PROJECT_ALLOCATIONS: Dict[str, Dict[str, float]] = {
    # Optional: feste Splits, sonst random-stabil über BUSINESS_UNITS
}

PROJECT_MONTHLY_DISTRIBUTION = "flat"  # ["flat" | "frontloaded" | "backloaded"]

# =====================================================================================
# === Applications (30 realistische Apps mit Vendor; IDs APP-0001 … APP-0030)
# =====================================================================================

DIM_APPS: List[Dict[str, str]] = [
    {"app_id": "APP-0001", "app_name": "SAP S/4HANA Finance", "vendor": "SAP"},
    {"app_id": "APP-0002", "app_name": "SAP S/4HANA Sales", "vendor": "SAP"},
    {"app_id": "APP-0003", "app_name": "SAP S/4HANA Procurement", "vendor": "SAP"},
    {"app_id": "APP-0004", "app_name": "SAP SuccessFactors", "vendor": "SAP"},
    {"app_id": "APP-0005", "app_name": "Microsoft 365", "vendor": "Microsoft"},
    {"app_id": "APP-0006", "app_name": "Microsoft Teams", "vendor": "Microsoft"},
    {"app_id": "APP-0007", "app_name": "Microsoft SharePoint Online", "vendor": "Microsoft"},
    {"app_id": "APP-0008", "app_name": "Azure Active Directory", "vendor": "Microsoft"},
    {"app_id": "APP-0009", "app_name": "ServiceNow Platform", "vendor": "ServiceNow"},
    {"app_id": "APP-0010", "app_name": "Salesforce CRM", "vendor": "Salesforce"},
    {"app_id": "APP-0011", "app_name": "Workday HCM", "vendor": "Workday"},
    {"app_id": "APP-0012", "app_name": "Atlassian Jira", "vendor": "Atlassian"},
    {"app_id": "APP-0013", "app_name": "Atlassian Confluence", "vendor": "Atlassian"},
    {"app_id": "APP-0014", "app_name": "GitLab Enterprise", "vendor": "GitLab"},
    {"app_id": "APP-0015", "app_name": "Oracle Database Platform", "vendor": "Oracle"},
    {"app_id": "APP-0016", "app_name": "IBM MQ / Integration Platform", "vendor": "IBM"},
    {"app_id": "APP-0017", "app_name": "Cisco Network Management Suite", "vendor": "Cisco"},
    {"app_id": "APP-0018", "app_name": "Palo Alto Security Platform", "vendor": "Palo Alto"},
    {"app_id": "APP-0019", "app_name": "Snowflake Data Cloud", "vendor": "Snowflake"},

    {"app_id": "APP-0020", "app_name": "Tableau Analytics Platform", "vendor": "Tableau"},
    {"app_id": "APP-0021", "app_name": "Enterprise Data Lake", "vendor": "Internal"},
    {"app_id": "APP-0022", "app_name": "IAM Core Platform", "vendor": "Internal"},
    {"app_id": "APP-0023", "app_name": "API Management Gateway", "vendor": "Internal"},
    {"app_id": "APP-0024", "app_name": "Manufacturing Execution Platform", "vendor": "Internal"},
    {"app_id": "APP-0025", "app_name": "Procurement Hub", "vendor": "Internal"},
    {"app_id": "APP-0026", "app_name": "Analytics & Reporting Hub", "vendor": "Internal"},
    {"app_id": "APP-0027", "app_name": "DWP (Digital Workplace Platform)", "vendor": "Internal"},
    {"app_id": "APP-0028", "app_name": "Mobile Device Management Platform", "vendor": "Internal"},
    {"app_id": "APP-0029", "app_name": "Corporate Intranet Portal", "vendor": "Internal"},
    {"app_id": "APP-0030", "app_name": "Data Integration Hub", "vendor": "Internal"},
]

# === App-Verfügbarkeit: 21 global, 9 lokal
GLOBAL_APPS = {
    "APP-0001","APP-0002","APP-0003","APP-0004",
    "APP-0005","APP-0006","APP-0007","APP-0008",
    "APP-0009","APP-0010","APP-0011","APP-0012","APP-0013","APP-0014",
    "APP-0015","APP-0016","APP-0018","APP-0019","APP-0020","APP-0021","APP-0022",
}

LOCAL_APP_COUNTRIES: Dict[str, List[str]] = {
    "APP-0017": ["DE","IT","NL","US","CA"],           # Cisco NMS
    "APP-0023": ["DE","FR","NL","US"],                # API Gateway
    "APP-0024": ["DE","IT","PL","MX","BR"],           # MES
    "APP-0025": ["DE","IT","PL","MX","BR"],           # Procurement Hub (lokalisiert)
    "APP-0026": ["DE","US","CA"],                     # Analytics Hub
    "APP-0027": ["DE","FR","NL","US","CA"],           # DWP
    "APP-0028": ["DE","FR","ES","US","CA"],           # MDM
    "APP-0029": ["DE","FR","ES","NL","US","CA"],      # Intranet
    "APP-0030": ["DE","IT","US"],                     # Data Integration Hub
}

# =====================================================================================
# === Cost Center Architektur
# =====================================================================================

COUNTRY_COST_CENTERS = {
    "BR": ["CC-BR-HQ", "CC-BR-MFG", "CC-BR-COM"],
    "CA": ["CC-CA-HQ", "CC-CA-COM"],
    "DE": ["CC-DE-HQ", "CC-DE-MFG", "CC-DE-COM"],
    "ES": ["CC-ES-HQ", "CC-ES-COM"],
    "FR": ["CC-FR-HQ", "CC-FR-COM"],
    "IT": ["CC-IT-HQ", "CC-IT-MFG", "CC-IT-COM"],
    "MX": ["CC-MX-HQ", "CC-MX-MFG", "CC-MX-COM"],
    "NL": ["CC-NL-HQ", "CC-NL-COM"],
    "PL": ["CC-PL-HQ", "CC-PL-MFG"],
    "US": ["CC-US-HQ", "CC-US-COM"],
    "CN": ["CC-CN-HQ", "CC-CN-MFG"],
    "AR": ["CC-AR-HQ", "CC-AR-MFG"],
}  # alphabetisch nach Country-Key

ORG_CLUSTER = {
    "ORG_001": "HQ",
    "ORG_002": "HQ",
    "ORG_003": "HQ",
    "ORG_004": "HQ",
    "ORG_005": "HQ",
    "ORG_006": "MFG",
    "ORG_007": "MFG",
    "ORG_008": "COM",
    "ORG_009": "HQ",  # R&D unter HQ
    "ORG_010": "COM",
}

# Precomputed Mapping: (country, org_id) -> cost_center_id
COST_CENTER_BY_COUNTRY_AND_ORG: Dict[tuple[str, str], str] = {}
for country, cc_list in COUNTRY_COST_CENTERS.items():
    for org_id, cluster in ORG_CLUSTER.items():
        if cluster == "HQ":
            candidates = [c for c in cc_list if c.endswith("-HQ")]
        elif cluster == "MFG":
            candidates = [c for c in cc_list if c.endswith("-MFG")]
        elif cluster == "COM":
            candidates = [c for c in cc_list if c.endswith("-COM")]
        else:
            candidates = []
        if candidates:
            COST_CENTER_BY_COUNTRY_AND_ORG[(country, org_id)] = candidates[0]
        # sonst kein Eintrag → BU in diesem Land nicht aktiv

# Beschreibende Namen für Cost Center (englisch)
def _cc_name(cc_id: str) -> str:
    # Format: CC-<CC>-<SUFFIX>
    try:
        _, ctry, suffix = cc_id.split("-")
    except ValueError:
        return cc_id
    cname = COUNTRY_NAMES_EN.get(ctry, ctry)
    if suffix == "HQ":
        return f"{cname} – Corporate/HQ"
    if suffix == "MFG":
        return f"{cname} – Manufacturing"
    if suffix == "COM":
        return f"{cname} – Commercial"
    return f"{cname} – {suffix}"

# DIM_COST_CENTERS automatisch aus COUNTRY_COST_CENTERS ableiten (alphabetisch sortiert)
_cc_items = []
for ctry in sorted(COUNTRY_COST_CENTERS.keys()):
    for cc_id in COUNTRY_COST_CENTERS[ctry]:
        _cc_items.append({"cost_center_id": cc_id, "cost_center_name": _cc_name(cc_id)})
DIM_COST_CENTERS: List[Dict[str, str]] = _cc_items

# =====================================================================================
# === Ende config.py
# =====================================================================================
