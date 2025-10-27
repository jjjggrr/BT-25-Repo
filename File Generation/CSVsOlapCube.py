# Generate synthetic TBM-like dataset (FY24/FY25 with fiscal year Oct–Sep)
# and write CSVs plus a DuckDB SQL bootstrap script to load locally on macOS.
import os
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime
from dateutil.relativedelta import relativedelta

# =========================
# Config (determinism, PxQ)
# =========================
USE_DETERMINISTIC = True
SEED = 42  # Für D2 (nicht-deterministisch): USE_DETERMINISTIC=False setzen

# Baselinebereiche (anpassbar)
BASE_PRICE_RANGE = (5.0, 50.0)           # € pro Einheit je Service
BASE_QTY_RANGE   = (100.0, 5000.0)       # Einheiten je Org×Service×Monat
MONTHLY_PRICE_GROWTH_RANGE = (0.004, 0.012)  # ~0.4%–1.2%/Monat (leichter Uptrend)
PRICE_NOISE_SD   = 0.01                  # 1% Preisrauschen
QTY_NOISE_SD     = 0.08                  # 8% Mengennebel

# Projektbudget relativ zur Run-Kostenbasis je Org×FY
PROJECT_BUDGET_FACTOR_RANGE = (0.12, 0.35)   # 12–35% von run_cost_year als Projektbudget
# Anteil des Projektbudgets in den letzten 4 Monaten (FY-M9..M12)
PROJECT_BACKLOAD_SHARE_RANGE = (0.60, 0.80)  # 60–80% in M9..M12

def _rng():
    if USE_DETERMINISTIC:
        return np.random.RandomState(SEED)
    return np.random.RandomState()

def _stable_u01(x: str) -> float:
    """Stabiler Hash → [0,1) ohne externe Libs (für reproduzierbare Baselines)."""
    return (abs(hash(str(x))) % 10_000) / 10_000.0

# =========================
# Pfade & Basis
# =========================
import platform
if platform.system() == "Windows":
    base_dir = "C:/Users/jakob/tbm_demo"
else:
    base_dir = "/Users/jakob/tbm_demo"

os.makedirs(base_dir, exist_ok=True)

def fiscal_months(start_year=2023, start_month=10, n_months=24):
    start = datetime(start_year, start_month, 1)
    out = []
    for i in range(n_months):
        d = start + relativedelta(months=i)
        fy = d.year + 1 if d.month >= 10 else d.year
        fiscal_period = f"FY{str(fy)[-2:]}"
        fiscal_month = d.strftime("%b")
        fiscal_month_num = ((d.month - 10) % 12) + 1  # Oct=1 ... Sep=12
        out.append({
            "date_key": int(d.strftime("%Y%m01")),
            "calendar_year": d.year,
            "calendar_month": d.month,
            "calendar_month_name": d.strftime("%b"),
            "fiscal_year": fiscal_period,
            "fiscal_month_num": fiscal_month_num,
            "fiscal_month_name": fiscal_month,
            "month_start": d.date().isoformat()
        })
    return pd.DataFrame(out)

dim_date = fiscal_months()

# Setze deterministische Seeds für numpy (nur kosmetisch; wir nutzen _rng() separat)
if USE_DETERMINISTIC:
    np.random.seed(SEED)

# Basisdimensionen
business_units = ["Sales", "Marketing", "Finance", "HR", "Operations", "R&D"]
regions = ["EMEA", "AMER", "APAC"]
cost_centers = [f"CC{100+i}" for i in range(1, 9)]
apps = [f"APP_{i:03d}" for i in range(1, 21)]
projects = [f"PRJ_{i:03d}" for i in range(1, 11)]

towers = [
    ("T01", "End User Services"),
    ("T02", "Network"),
    ("T03", "Hosting"),
    ("T04", "Application Dev & Support"),
    ("T05", "IT Management"),
    ("T06", "Cyber Security")
]

services_by_tower = {
    "T01": ["VDI", "Device Mgmt", "Collaboration", "Email"],
    "T02": ["LAN", "WAN", "Internet", "Load Balancing"],
    "T3x": ["VM", "Containers", "DBaaS", "Backup"],  # temporary key, fix below
    "T04": ["DevOps", "CI/CD", "Issue Tracking", "Test Environments"],
    "T05": ["ITSM", "FinOps", "Monitoring", "Asset Mgmt"],
    "T06": ["IAM", "Threat Detection", "Vulnerability Mgmt", "SOC"]
}
# Fix typo in key to map to T03
services_by_tower["T03"] = services_by_tower.pop("T3x")

dim_tower = pd.DataFrame([{"tower_id": t[0], "tower_name": t[1]} for t in towers])

service_rows = []
service_id_counter = 1
for t_id, s_list in services_by_tower.items():
    for s in s_list:
        service_rows.append({
            "service_id": f"S{service_id_counter:03d}",
            "tower_id": t_id,
            "service_name": s
        })
        service_id_counter += 1
dim_service = pd.DataFrame(service_rows)

org_rows = []
org_id = 1
for bu in business_units:
    for r in regions:
        org_rows.append({
            "org_id": f"ORG{org_id:03d}",
            "business_unit": bu,
            "region": r
        })
        org_id += 1
dim_org = pd.DataFrame(org_rows)

dim_cost_center = pd.DataFrame([{"cost_center_id": cc, "cost_center_name": cc} for cc in cost_centers])
dim_app = pd.DataFrame([{"app_id": a, "app_name": a} for a in apps])
dim_project = pd.DataFrame([{"project_id": p, "project_name": p} for p in projects])

# Mapping „Service → Unit Model“ (nur Labeling)
unit_model_map = {
    "VDI": "seats", "Device Mgmt": "devices", "Collaboration": "seats", "Email": "mailboxes",
    "LAN": "ports", "WAN": "sites", "Internet": "gbps", "Load Balancing": "rules",
    "VM": "vms", "Containers": "containers", "DBaaS": "db_instances", "Backup": "tb",
    "DevOps": "users", "CI/CD": "pipelines", "Issue Tracking": "users", "Test Environments": "envs",
    "ITSM": "tickets", "FinOps": "accounts", "Monitoring": "agents", "Asset Mgmt": "assets",
    "IAM": "identities", "Threat Detection": "events", "Vulnerability Mgmt": "scans", "SOC": "cases"
}

# =========================
# Fakt-Grundgerüst (Keys)
# =========================
rs_global = _rng()

fact_rows = []
for _, drow in dim_date.iterrows():
    fiscal_month_num = int(drow["fiscal_month_num"])
    date_key = int(drow["date_key"])
    fy = drow["fiscal_year"]

    for _, srow in dim_service.iterrows():
        service_id = srow["service_id"]
        tower_id = srow["tower_id"]
        service_name = srow["service_name"]

        for _, org in dim_org.iterrows():
            org_id = org["org_id"]

            # zufällige, aber reproduzierbare Auswahl der Cost Center (gewichtete Verteilung)
            cc = rs_global.choice(cost_centers, p=np.array([0.16,0.14,0.12,0.10,0.16,0.12,0.10,0.10]))
            # optionale Zuordnung von App/Project (nur Keys; monetäre Wirkung kommt später aus PxQ/Project modelling)
            app_id = rs_global.choice(apps) if rs_global.rand() < 0.25 else None
            project_id = rs_global.choice(projects) if rs_global.rand() < 0.15 else None

            unit_model = unit_model_map[service_name]

            fact_rows.append({
                "date_key": date_key,
                "fiscal_year": fy,
                "fiscal_month_num": fiscal_month_num,
                "tower_id": tower_id,
                "service_id": service_id,
                "service_name": service_name,
                "org_id": org_id,
                "business_unit": org["business_unit"],
                "region": org["region"],
                "cost_center_id": cc,
                "app_id": app_id,
                "project_id": project_id,
                "unit_model": unit_model
            })

fact = pd.DataFrame(fact_rows)

# =========================
# PxQ + lumpy Seasonality + Project Backload
# =========================
def enrich_price_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt price, quantity, service_cost mit lumpy Seasonality (S3) und leichtem Preis-Uptrend."""
    rs = _rng()
    out = df.copy()

    # Baselines: deterministisch aus IDs
    svc_u = out['service_id'].map(_stable_u01)
    base_price = BASE_PRICE_RANGE[0] + svc_u * (BASE_PRICE_RANGE[1] - BASE_PRICE_RANGE[0])

    orgsvc = out['org_id'].astype(str) + '|' + out['service_id'].astype(str)
    orgsvc_u = orgsvc.map(_stable_u01)
    base_qty = BASE_QTY_RANGE[0] + orgsvc_u * (BASE_QTY_RANGE[1] - BASE_QTY_RANGE[0])

    # Monatlicher Price-Growth je Service
    monthly_growth = MONTHLY_PRICE_GROWTH_RANGE[0] + svc_u * (MONTHLY_PRICE_GROWTH_RANGE[1] - MONTHLY_PRICE_GROWTH_RANGE[0])
    m = out['fiscal_month_num'].astype(int).clip(1, 12)
    trend_factor = 1.0 + monthly_growth * (m - 1)  # näherungsweise linear – bei kleinen g ausreichend

    # Preisrauschen
    price_noise = rs.normal(loc=0.0, scale=PRICE_NOISE_SD, size=len(out))
    out['price'] = (base_price * trend_factor) * (1.0 + price_noise)
    out['price'] = out['price'].clip(lower=0.01).round(2)  # <<< 2 decimals

    # Lumpy quantity:
    qty = base_qty.copy()
    # Org-scale
    org_u = out['org_id'].map(_stable_u01)
    org_scale = 0.85 + org_u * (1.15 - 0.85)  # 0.85..1.15
    qty = qty * org_scale

    # Spike-Wahrscheinlichkeit & Stärke je Org×Service×Monat
    key_u = (out['org_id'].astype(str) + '|' + out['service_id'].astype(str) + '|' + m.astype(str)).map(_stable_u01)
    p_spike = 0.18 + (orgsvc_u * 0.10)  # ~18–28% Spike-Chance
    is_spike = key_u < p_spike
    spike_uplift = 0.10 + key_u * 0.30  # 10–40% wenn Spike
    season_mult = 1.0 + (is_spike * spike_uplift)

    # Mengennebel
    qty_noise = rs.normal(loc=0.0, scale=QTY_NOISE_SD, size=len(out))
    qty_noise_mult = np.maximum(0.0, 1.0 + qty_noise)

    out['quantity'] = (qty * season_mult * qty_noise_mult).clip(lower=0.0)

    # <<< HIER RUNDEN WIR FÜR GUTE SEMANTIK
    out['quantity'] = out['quantity'].round(0)  # integer-consumption (float repr)

    # Servicekosten aus PxQ
    out['service_cost'] = out['price'] * out['quantity']
    return out

def add_project_backload(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt project_cost je Org×FY hinzu; 60–80% des Jahresbudgets in M9..M12, Rest dünn in M1..M8."""
    rs = _rng()
    out = df.copy()

    out['fiscal_year_str'] = out['fiscal_year'].astype(str)

    # Jahres-Runbasis je Org×FY (Summe der Servicekosten)
    run_year = (
        out.groupby(['fiscal_year_str', 'org_id'], as_index=False)['service_cost']
        .sum()
        .rename(columns={'service_cost': 'run_cost_year'})
    )
    out = out.merge(run_year, on=['fiscal_year_str', 'org_id'], how='left')

    # Projektbudget je Org×FY als Anteil der Run-Kosten
    proj_factor_u = (out['org_id'].astype(str) + '|' + out['fiscal_year_str']).map(_stable_u01)
    proj_factor = PROJECT_BUDGET_FACTOR_RANGE[0] + proj_factor_u * (PROJECT_BUDGET_FACTOR_RANGE[1] - PROJECT_BUDGET_FACTOR_RANGE[0])
    out['project_budget_year'] = out['run_cost_year'] * proj_factor

    # Backload-Anteil 60–80% in Monaten 9..12
    backload_u = (out['org_id'].astype(str) + '|BL|' + out['fiscal_year_str']).map(_stable_u01)
    backload_share = PROJECT_BACKLOAD_SHARE_RANGE[0] + backload_u * (PROJECT_BACKLOAD_SHARE_RANGE[1] - PROJECT_BACKLOAD_SHARE_RANGE[0])

    out['project_cost'] = 0.0

    # Verteilung pro Org×FY
    for (fy, org), grp in out.groupby(['fiscal_year_str', 'org_id']):
        idx = grp.index
        budget = float(grp['project_budget_year'].iloc[0])
        if budget <= 0:
            continue

        months_series = grp['fiscal_month_num'].astype(int)
        idx_m9_12 = grp.loc[months_series >= 9].index
        idx_m1_8  = grp.loc[months_series <= 8].index

        bl_share = float(backload_share.loc[idx].iloc[0])
        bl_budget = budget * bl_share
        rest_budget = budget - bl_budget

        # M9..M12: verteile bl_budget per Dirichlet
        if len(idx_m9_12) > 0:
            w = rs.dirichlet(np.ones(len(idx_m9_12)))
            out.loc[idx_m9_12, 'project_cost'] = bl_budget * w

        # M1..M8: dünn – ~40% der Monate aktiv
        if len(idx_m1_8) > 0 and rest_budget > 0:
            active_mask = rs.rand(len(idx_m1_8)) < 0.40
            if active_mask.any():
                active_idx = idx_m1_8[active_mask]
                w2 = rs.dirichlet(np.ones(active_mask.sum()))
                out.loc[active_idx, 'project_cost'] = rest_budget * w2
            else:
                # Fallback: alles in einen zufälligen Monat
                j = rs.choice(idx_m1_8)
                out.loc[j, 'project_cost'] = rest_budget

    # Final actual_cost
    out['project_cost'] = out['project_cost'].fillna(0.0)
    out['actual_cost']  = out['service_cost'] + out['project_cost']

    # Aufräumen
    out = out.drop(columns=['fiscal_year_str', 'run_cost_year', 'project_budget_year'], errors='ignore')
    return out

def split_run_project_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-Split in RUN- und PROJECT-Zeilen:
      - RUN:  cost_type='RUN',  actual_cost = service_cost, project_cost = 0.0
      - PROJECT: cost_type='PROJECT', price = NaN, quantity = 0.0, units = 0.0,
                 service_cost = 0.0, actual_cost = project_cost, project_quantity = 0.0
    Sortiert am Ende nach (date_key, tower_id, service_id, org_id, cost_type).
    """
    run = df.copy()
    run['cost_type'] = 'RUN'
    run['project_quantity'] = 0.0
    run['project_cost'] = 0.0  # <<< WICHTIG: Projektanteil gehört NICHT in RUN-Row
    # RUN-actual ist nur der Verbrauchsteil
    run['actual_cost'] = run['service_cost']
    # Units/Unit-Cost spiegeln PxQ (RUN only)
    run['units'] = run['quantity'].round(2)
    run['unit_cost'] = run['price'].round(4)

    proj_mask = df['project_cost'] > 0
    proj = df.loc[proj_mask].copy()
    if not proj.empty:
        proj['cost_type'] = 'PROJECT'
        proj['project_quantity'] = 0.0
        proj['quantity'] = 0.0
        proj['units'] = 0.0
        proj['price'] = np.nan
        proj['unit_cost'] = np.nan
        proj['service_cost'] = 0.0
        # PROJECT-actual ist nur der Projektanteil
        proj['actual_cost'] = proj['project_cost']

        out = pd.concat([run, proj], ignore_index=True)
    else:
        out = run

    # deterministische Ausgabe-Reihenfolge
    out = out.sort_values(
        ['date_key', 'tower_id', 'service_id', 'org_id', 'cost_type']
    ).reset_index(drop=True)

    return out


# Anwenden auf das Fact-Grundgerüst
fact = enrich_price_quantity(fact)
fact = add_project_backload(fact)

# Row-Splitting (RUN / PROJECT) + Sortierung
fact = split_run_project_rows(fact)

# Forecast jetzt NACH dem Split berechnen, auf Basis des getrennten actual_cost
rs_fc = _rng()
trend_factor_fc = np.where(fact['fiscal_year'] == 'FY25', 1.005, 1.0)
fact['forecast_cost'] = (
    fact['actual_cost'] * trend_factor_fc * (1.0 + rs_fc.normal(0.0, 0.02, size=len(fact)))
).round(2)

# Runden/Format (falls nicht schon vorher gesetzt)
fact['actual_cost'] = fact['actual_cost'].round(2)
fact['service_cost'] = fact['service_cost'].round(2)
fact['project_cost'] = fact['project_cost'].round(2)


# =========================
# CSV-Export
# =========================
files = {
    "dim_date.csv": dim_date,
    "dim_tower.csv": pd.DataFrame([{"tower_id": t[0], "tower_name": t[1]} for t in towers]),
    "dim_service.csv": dim_service,
    "dim_org.csv": dim_org,
    "dim_cost_center.csv": dim_cost_center,
    "dim_app.csv": dim_app,
    "dim_project.csv": dim_project,
    "fact_it_costs.csv": fact[
    [
        "date_key", "fiscal_year", "fiscal_month_num",
        "tower_id", "service_id", "org_id",
        "cost_center_id", "app_id", "project_id",
        "cost_type", "unit_model",
        # PxQ + Komponenten
        "units", "unit_cost", "quantity", "price",
        "project_quantity",
        "service_cost", "project_cost",
        # Hauptkennzahlen
        "actual_cost", "forecast_cost"
    ]
]

}

# Write dataframes to CSV files
for filename, df in files.items():
    df.to_csv(os.path.join(base_dir, filename), index=False)

# =========================
# DuckDB Bootstrap
# =========================
sql_script_path = os.path.join(base_dir, "refresh_duckdb.sql")
db_path = os.path.join(base_dir, "tbm_demo.duckdb")

with open(sql_script_path, "w") as f:
    # Neues Custom-Schema (droppbar)
    f.write("DROP SCHEMA IF EXISTS tbm CASCADE;\n")
    f.write("CREATE SCHEMA tbm;\n")

    # Alle Tabellen neu erzeugen
    for table_name, df in files.items():
        duckdb_table_name = os.path.splitext(table_name)[0]
        csv_path = os.path.join(base_dir, table_name)
        f.write(
            f"CREATE TABLE tbm.{duckdb_table_name} "
            f"AS SELECT * FROM read_csv_auto('{csv_path}');\n"
        )

# Connect to DuckDB and run the generated script
try:
    with duckdb.connect(database=db_path, read_only=False) as con:
        with open(sql_script_path, "r") as f:
            sql_script = f.read()
            con.execute(sql_script)
    print(f"Successfully refreshed DuckDB database at '{db_path}'")
except Exception as e:
    print(f"An error occurred: {e}")






