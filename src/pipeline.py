from __future__ import annotations
import json
import os
import platform
import random
from datetime import date
from pathlib import Path
from typing import List, Dict, Optional

import duckdb
import pandas as pd
from rich import print as rprint

from .config import (
    SERVICES, BUSINESS_UNITS, FISCAL_YEARS, CURRENCY,
    TOWERS, PROJECTS, PROJECT_BUDGETS,
)
from .models import (
    ServicePrice, ProjectDef, FactRunRow, FactChangeRow,
    ServiceDocMeta, ProjectDocMeta, ProjectAllocation,
)
from .gen_prices import gen_service_prices
from .gen_projects import gen_projects
from .gen_quantities import gen_run_quantities
from .build_facts import build_run_facts, build_change_facts
from .validate import (
    validate_price_constancy, validate_pxq, validate_project_totals
)
from .pdf_service_agreement import render_service_agreement_pdf
from .pdf_project_brief import render_project_brief_pdf
from .embedder import Embedder


# =========================
# Helpers
# =========================

def _base_dir() -> Path:
    if platform.system() == "Windows":
        return Path("C:/Users/jakob/tbm_demo")
    else:
        return Path("/Users/jakob/tbm_demo")


def _to_df(records, cls):
    return pd.DataFrame([r.model_dump() for r in records], columns=list(cls.model_fields.keys()))


def _fy_to_start_year(fy: str) -> int:
    # FY24 starts Oct 2023; FY25 starts Oct 2024 → start_year = 2000 + (N-1)
    n = int(fy.replace("FY", ""))
    return 2000 + (n - 1)


def _date_parts(fiscal_year: str, fiscal_month_num: int) -> Dict[str, object]:
    # Map TBM fiscal month (Oct=1..Sep=12) to calendar YYYY-MM-01
    start_year = _fy_to_start_year(fiscal_year)
    cal_month = ((fiscal_month_num - 1) + 9) % 12 + 1  # 1..12, Oct=10
    year = start_year if cal_month >= 10 else (start_year + 1)
    month_start = date(year, cal_month, 1)
    calendar_month_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][
        cal_month - 1]
    fiscal_names = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    fiscal_month_name = fiscal_names[fiscal_month_num - 1]
    date_key = int(month_start.strftime("%Y%m01"))
    return {
        "calendar_year": year,
        "calendar_month": cal_month,
        "calendar_month_name": calendar_month_name,
        "fiscal_month_name": fiscal_month_name,
        "month_start": month_start.isoformat(),
        "date_key": date_key,
    }


def _build_dim_date() -> pd.DataFrame:
    rows = []
    for fy in FISCAL_YEARS:
        for m in range(1, 13):
            p = _date_parts(fy, m)
            rows.append({
                "date_key": p["date_key"],
                "calendar_year": p["calendar_year"],
                "calendar_month": p["calendar_month"],
                "calendar_month_name": p["calendar_month_name"],
                "fiscal_year": fy,
                "fiscal_month_num": m,
                "fiscal_month_name": p["fiscal_month_name"],
                "month_start": p["month_start"],
            })
    df = pd.DataFrame(rows, columns=[
        "date_key", "calendar_year", "calendar_month", "calendar_month_name",
        "fiscal_year", "fiscal_month_num", "fiscal_month_name", "month_start"
    ])
    return df


def _build_dim_tower() -> pd.DataFrame:
    return pd.DataFrame([
        {"tower_id": t["tower_id"], "tower_name": t["tower_name"]} for t in TOWERS
    ], columns=["tower_id", "tower_name"])


def _build_dim_service() -> pd.DataFrame:
    return pd.DataFrame([
        {"service_id": s["service_id"], "tower_id": s["tower_id"], "service_name": s["service_name"]}
        for s in SERVICES
    ], columns=["service_id", "tower_id", "service_name"])


def _build_dim_org() -> pd.DataFrame:
    return pd.DataFrame([
        {"org_id": o["org_id"], "business_unit_name": o["business_unit_name"]}
        for o in BUSINESS_UNITS
    ], columns=["org_id", "business_unit_name"])


def _build_dim_cost_center() -> pd.DataFrame:
    from .config import DIM_COST_CENTERS
    return pd.DataFrame(
        [{"cost_center_id": c["cost_center_id"], "cost_center_name": c["cost_center_name"]}
         for c in DIM_COST_CENTERS],
        columns=["cost_center_id", "cost_center_name"]
    )


def _build_dim_app() -> pd.DataFrame:
    from .config import DIM_APPS
    return pd.DataFrame(
        [{"app_id": a["app_id"], "app_name": a["app_name"], "vendor": a["vendor"]}
         for a in DIM_APPS],
        columns=["app_id", "app_name", "vendor"]
    )


def _build_dim_project(projects: List[ProjectDef]) -> pd.DataFrame:
    return pd.DataFrame([
        {"project_id": p.project_id, "project_name": p.name}
        for p in projects
    ], columns=["project_id", "project_name"])

def write_csv_excel_friendly(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, sep=';', decimal=',', encoding='utf-8-sig')


def _build_dim_country() -> pd.DataFrame:
    countries = [
        {"country_code": "DE", "country_name": "GERMANY"},
        {"country_code": "FR", "country_name": "FRANCE"},
        {"country_code": "IT", "country_name": "ITALY"},
        {"country_code": "ES", "country_name": "SPAIN"},
        {"country_code": "NL", "country_name": "NETHERLANDS"},
        {"country_code": "PL", "country_name": "POLAND"},
        {"country_code": "US", "country_name": "UNITED STATES"},
        {"country_code": "CA", "country_name": "CANADA"},
        {"country_code": "MX", "country_name": "MEXICO"},
        {"country_code": "BR", "country_name": "BRAZIL"},
        {"country_code": "AR", "country_name": "ARGENTINA"},
        {"country_code": "CN", "country_name": "CHINA"},
    ]
    return pd.DataFrame(countries, columns=["country_code", "country_name"])


def _final_fact_df(run_rows: List[FactRunRow], change_rows: List[FactChangeRow]) -> pd.DataFrame:
    # unified schema for tbm.fact_it_costs as CSV (your legacy column order)
    # RUN part
    run_df = _to_df(run_rows, FactRunRow).copy()
    run_df["cost_type"] = "RUN"
    run_df["service_cost"] = run_df["runCost"].astype(float)
    run_df["project_cost"] = 0.0
    run_df["actual_cost"] = run_df["service_cost"]
    run_df["quantity"] = run_df["quantity"].astype(int)
    run_df["units"] = run_df["quantity"].astype(float)
    run_df["unit_cost"] = run_df["price"].astype(float)  # legacy alias
    run_df["project_quantity"] = 0.0
    run_df["unit_model"] = "PXQ"
    run_df["project_id"] = ""
    # synthetic date_key
    run_df["date_key"] = run_df.apply(lambda r: _date_parts(r["fiscal_year"], int(r["fiscal_month_num"]))['date_key'],
                                      axis=1)

    # CHANGE part
    ch_df = _to_df(change_rows, FactChangeRow).copy()
    ch_df["cost_type"] = "CHANGE"
    ch_df["service_cost"] = 0.0
    ch_df["actual_cost"] = ch_df["project_cost"].astype(float)
    ch_df["price"] = 0.0
    # quantity/app_id/cost_center_id/tower_id/service_id bleiben aus FactChangeRow
    # KEINE Überschreibungen mehr!
    ch_df["unit_model"] = "N/A"
    ch_df["date_key"] = ch_df.apply(
        lambda r: _date_parts(r["fiscal_year"], int(r["fiscal_month_num"]))["date_key"],
        axis=1,
    )

    # Align columns and order to your legacy list
    col_order = [
        "date_key", "fiscal_year", "fiscal_month_num",
        "tower_id", "service_id", "org_id", "country_code",
        "cost_center_id", "app_id", "project_id",
        "cost_type", "unit_model",
        # PxQ + Komponenten
        "units", "unit_cost", "quantity", "price", "project_quantity",
        "service_cost", "project_cost",
        # Hauptkennzahlen
        "actual_cost", "forecast_cost",
    ]

    for df in (run_df, ch_df):
        for col in col_order:
            if col not in df.columns:
                df[col] = 0 if col in (
                "units", "unit_cost", "quantity", "price", "project_quantity", "service_cost", "project_cost",
                "actual_cost", "forecast_cost") else ""

    full = pd.concat([run_df, ch_df], ignore_index=True)[col_order]

    # Forecast = actual * (1 ± 5%), deterministic
    rnd = random.Random(12345)
    full["forecast_cost"] = full["actual_cost"].apply(lambda x: round(x * (1.0 + rnd.uniform(-0.05, 0.05)), 4))

    # Types
    num_cols = ["date_key", "fiscal_month_num", "units", "unit_cost", "quantity", "price", "project_quantity",
                "service_cost", "project_cost", "actual_cost", "forecast_cost"]
    full[num_cols] = full[num_cols].apply(pd.to_numeric)
    return full


def _write_csvs_and_bootstrap_duckdb(files: Dict[str, pd.DataFrame]):
    """Write DataFrames directly into DuckDB (in-memory) and
        export Excel-friendly CSVs into base_dir for inspection."""
    base_dir = _base_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    db_path = base_dir / "tbm_demo.duckdb"
    con = duckdb.connect(database=str(db_path), read_only=False)
    con.execute("CREATE SCHEMA IF NOT EXISTS tbm;")

    # Für jeden DataFrame
    for filename, df in files.items():
        table_name = Path(filename).stem
        print(f"[duckdb] Creating tbm.{table_name} from DataFrame ({len(df)} rows)")

        # --- 1) Direkt in-Memory nach DuckDB ---
        con.register('temp_df', df)
        con.execute(f"CREATE OR REPLACE TABLE tbm.{table_name} AS SELECT * FROM temp_df;")
        con.unregister('temp_df')

        # --- 2) Zusätzlich Excel-freundlich exportieren (im base_dir) ---
        out_path = base_dir / filename
        write_csv_excel_friendly(df, out_path)
        rprint(f"[cyan]CSV[/cyan] → {out_path}")

    con.close()
    rprint(f"[green]DuckDB refresh complete (in-memory, Excel-friendly CSVs in '{base_dir}')[/green]")


# =========================
# Main pipeline
# =========================

def run_pipeline():
    rprint("[bold]1) Generate prices[/bold]")
    prices: List[ServicePrice] = gen_service_prices()

    rprint("[bold]2) Generate projects + allocations[/bold]")
    projects: List[ProjectDef] = gen_projects()

    rprint("[bold]3) Generate RUN quantities[/bold]")
    qrows = gen_run_quantities()

    rprint("[bold]4) Build RUN facts (PxQ) and CHANGE facts (projects)[/bold]")
    run_rows: List[FactRunRow] = build_run_facts(qrows, prices)
    change_rows: List[FactChangeRow] = build_change_facts(projects, run_rows)

    rprint("[bold]5) Validate[/bold]")
    validate_price_constancy(run_rows)
    validate_pxq(run_rows)
    validate_project_totals(change_rows, projects)
    #validate_project_allocation_constancy(projects)

    # 6) Build dims
    rprint("[bold]6) Build dimension tables[/bold]")
    dim_date = _build_dim_date()
    dim_tower = _build_dim_tower()
    dim_service = _build_dim_service()
    dim_org = _build_dim_org()
    dim_cost_center = _build_dim_cost_center()
    dim_app = _build_dim_app()
    dim_project = _build_dim_project(projects)

    # 7) Build unified fact with your legacy column order
    rprint("[bold]7) Build unified fact_it_costs (legacy schema)[/bold]")
    fact = _final_fact_df(run_rows, change_rows)

    # 8) Assemble file map exactly like your previous code
    files = {
        "dim_date.csv": dim_date,
        "dim_tower.csv": dim_tower,
        "dim_service.csv": dim_service,
        "dim_org.csv": dim_org,
        "dim_cost_center.csv": dim_cost_center,
        "dim_app.csv": dim_app,
        "dim_project.csv": dim_project,
        "dim_country.csv": _build_dim_country(),
        "fact_it_costs.csv": fact[[
            "date_key", "fiscal_year", "fiscal_month_num",
            "tower_id", "service_id", "org_id", "country_code",
            "cost_center_id", "app_id", "project_id",
            "cost_type", "unit_model",
            # PxQ + Komponenten
            "units", "unit_cost", "quantity", "price",
            "project_quantity",
            "service_cost", "project_cost",
            # Hauptkennzahlen
            "actual_cost", "forecast_cost",
        ]],
    }

    # 9) Write CSVs and refresh DuckDB (like your original flow)
    _write_csvs_and_bootstrap_duckdb(files)

    # 10) Render PDFs → Embed → Delete (unchanged)
    rprint("[bold]10) Render PDFs per Service & Project + Embed + Delete[/bold]")
    emb = Embedder(_base_dir())  # persist near base_dir

    for s in SERVICES:
        sid = s["service_id"]
        for fy in FISCAL_YEARS:
            price_curr = next(p for p in prices if p.service_id == sid and p.fiscal_year == fy)
            price_prev: Optional[ServicePrice] = None
            if fy == "FY25":
                price_prev = next(p for p in prices if p.service_id == sid and p.fiscal_year == "FY24")

            price_delta = 0.0 if price_prev is None else (price_curr.price / price_prev.price - 1.0)
            meta = ServiceDocMeta(
                fiscal_year=fy,
                tower_id=s["tower_id"],
                service_id=sid,
                project_id="N/A",  # Services are not projects
                price_fy=price_curr.price,
                price_unit=f"{CURRENCY}/unit",
                price_delta_pct_vs_prev_fy=price_delta,
            )
            pdf_path = render_service_agreement_pdf(meta, price_curr, price_prev)
            chunks = [
                f"Overview & Scope: {s['service_name']} provides capability to BUs.",
                f"Pricing Model & Unit: Unit={s['unit']}, Billing=PxQ, price constant per FY.",
                (
                        f"Price Table (FY): price_fy={price_curr.price:.4f} {CURRENCY}/unit"
                        + (f", delta_vs_prev={price_delta:.4f}" if price_prev else "")
                ),
                "Change Log vs Previous FY: see price delta and scope notes.",
                "Operational Notes: SLAs unchanged.",
            ]
            emb.add_service_chunks(pdf_name=pdf_path.name, meta=meta, chunks=chunks)
            pdf_path.unlink(missing_ok=True)

    for p in projects:
        for fy in FISCAL_YEARS:
            exists = (fy == "FY24" and p.exists_fy24) or (fy == "FY25" and p.exists_fy25)
            if not exists:
                continue
            meta = ProjectDocMeta(
                fiscal_year=fy,
                tower_id="N/A",  # Projects are not services
                service_id="N/A",  # Projects are not services
                project_id=p.project_id,
                project_cost_fy24=p.cost_fy24 if p.exists_fy24 else 0.0,
                project_cost_fy25=p.cost_fy25 if p.exists_fy25 else 0.0,
                is_new_in_fy25=(p.exists_fy25 and not p.exists_fy24),
                allocation_vector=p.allocation,
            )
            pdf_path = render_project_brief_pdf(meta, p)
            chunks = [
                f"Project Summary: {p.name} addresses capability uplift.",
                f"Budget & Yearly Costs: FY24={p.cost_fy24:.2f} {CURRENCY}, FY25={p.cost_fy25:.2f} {CURRENCY}, NewInFY25={p.exists_fy25 and not p.exists_fy24}",
                "BU Allocation (stable across FYs): " + ", ".join([f"{a.org_id}:{a.share:.4f}" for a in p.allocation]),
                "Changes vs Previous FY: phased delivery; budget reflects roadmap.",
                "Dependencies & Risks: none material.",
            ]
            emb.add_project_chunks(pdf_name=pdf_path.name, meta=meta, chunks=chunks)
            pdf_path.unlink(missing_ok=True)

    rprint("[bold green]Pipeline complete.[/bold green]")


if __name__ == "__main__":
    run_pipeline()
