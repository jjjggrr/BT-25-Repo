from __future__ import annotations
import json
import platform
import random
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from rich import print as rprint
import duckdb

from .config import OUT_DIR, PDF_DIR, EMB_DIR, SERVICES, BUSINESS_UNITS, FISCAL_YEARS, CURRENCY
from .models import (
    ServicePrice, ProjectDef, FactRunRow, FactChangeRow,
    ServiceDocMeta, ProjectDocMeta,
)
from .gen_prices import gen_service_prices
from .gen_projects import gen_projects
from .gen_quantities import gen_run_quantities
from .build_facts import build_run_facts, build_change_facts
from .validate import (
    validate_price_constancy, validate_pxq, validate_project_totals, validate_project_allocation_constancy
)
from .pdf_service_agreement import render_service_agreement_pdf
from .pdf_project_brief import render_project_brief_pdf
from .embedder import Embedder


# ---- helpers for final fact and date_key ----

def _to_df(records, cls):
    return pd.DataFrame([r.model_dump() for r in records], columns=list(cls.model_fields.keys()))


def _fy_to_start_year(fy: str) -> int:
    # FY24 starts Oct 2023; FY25 starts Oct 2024 → start_year = 2000 + (N-1)
    n = int(fy.replace("FY", ""))
    return 2000 + (n - 1)


def _date_key(fiscal_year: str, fiscal_month_num: int) -> int:
    # Map TBM fiscal month (Oct=1..Sep=12) to calendar YYYYMM integer
    start_year = _fy_to_start_year(fiscal_year)
    cal_month = ((fiscal_month_num - 1) + 9) % 12 + 1  # 1..12, Oct=10
    year = start_year if cal_month >= 10 else (start_year + 1)
    return year * 100 + cal_month  # e.g., 202310


def _final_fact_df(run_rows: List[FactRunRow], change_rows: List[FactChangeRow]) -> pd.DataFrame:
    # Build unified schema for tbm.fact_it_costs
    # Column set aligned to your Cube: service_cost (RUN), project_cost (CHANGE), actual_cost, forecast_cost, price, quantity, units, cost_type, unit_model, ids, date_key
    # Note: cost_center_id/app_id not modeled → set empty strings for compatibility

    # RUN part
    run_df = _to_df(run_rows, FactRunRow)
    run_df["cost_type"] = "RUN"
    run_df["service_cost"] = run_df["runCost"].astype(float)
    run_df["project_cost"] = 0.0
    run_df["actual_cost"] = run_df["service_cost"]  # no change component on RUN rows
    run_df["units"] = run_df["quantity"].astype(float)
    run_df["unit_model"] = "PXQ"
    run_df["project_id"] = ""
    run_df["cost_center_id"] = ""
    run_df["app_id"] = ""
    run_df["date_key"] = run_df.apply(lambda r: _date_key(r["fiscal_year"], int(r["fiscal_month_num"])), axis=1)

    # CHANGE part
    ch_df = _to_df(change_rows, FactChangeRow)
    ch_df["cost_type"] = "CHANGE"
    ch_df["service_cost"] = 0.0
    ch_df["actual_cost"] = ch_df["project_cost"].astype(float)
    ch_df["price"] = 0.0
    ch_df["quantity"] = 0.0
    ch_df["units"] = 0.0
    ch_df["unit_model"] = "N/A"
    ch_df["service_id"] = ch_df["service_id"].fillna("")
    ch_df["tower_id"] = ch_df["tower_id"].fillna("")
    ch_df["cost_center_id"] = ""
    ch_df["app_id"] = ""
    ch_df["date_key"] = ch_df.apply(lambda r: _date_key(r["fiscal_year"], int(r["fiscal_month_num"])), axis=1)

    # Align columns
    common_cols = [
        "fiscal_year", "fiscal_month_num", "date_key",
        "tower_id", "service_id", "org_id", "cost_center_id", "app_id", "project_id",
        "cost_type", "unit_model",
        "service_cost", "project_cost", "actual_cost",
        "forecast_cost",  # to be filled below
        "quantity", "units", "price",
    ]

    # Ensure missing cols exist before concat
    for df in (run_df, ch_df):
        for col in common_cols:
            if col not in df.columns:
                df[col] = 0 if col in ("service_cost","project_cost","actual_cost","forecast_cost","quantity","units","price") else ""

    # Concat
    full = pd.concat([run_df, ch_df], ignore_index=True)[common_cols]

    # Compute forecast_cost = actual_cost * (1 + epsilon), epsilon ~ U(-0.05, 0.05)
    rnd = random.Random(12345)
    eps = full["actual_cost"].apply(lambda _: rnd.uniform(-0.05, 0.05))
    full["forecast_cost"] = (full["actual_cost"] * (1.0 + eps)).round(4)

    # Final tidy types
    num_cols = ["fiscal_month_num","date_key","service_cost","project_cost","actual_cost","forecast_cost","quantity","units","price"]
    full[num_cols] = full[num_cols].apply(pd.to_numeric)

    return full


def _write_duckdb(final_fact: pd.DataFrame):
    # honor user's base_dir (Windows vs macOS)
    if platform.system() == "Windows":
        base_dir = "C:/Users/jakob/tbm_demo"
    else:
        base_dir = "/Users/jakob/tbm_demo"

    db_path = Path(base_dir) / "tbm.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Write using DuckDB; create schema and replace table
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS tbm")

    # Register DataFrame and create table
    con.register("final_fact", final_fact)
    con.execute("DROP TABLE IF EXISTS tbm.fact_it_costs")
    con.execute("CREATE TABLE tbm.fact_it_costs AS SELECT * FROM final_fact")
    con.close()

    rprint(f"[green]Wrote[/green] tbm.fact_it_costs → {db_path}")


def run_pipeline():
    rprint("[bold]1) Generate prices[/bold]")
    prices: List[ServicePrice] = gen_service_prices()

    rprint("[bold]2) Generate projects + allocations[/bold]")
    projects: List[ProjectDef] = gen_projects()

    rprint("[bold]3) Generate RUN quantities[/bold]")
    qrows = gen_run_quantities()

    rprint("[bold]4) Build RUN facts (PxQ) and CHANGE facts (projects)[/bold]")
    run_rows: List[FactRunRow] = build_run_facts(qrows, prices)
    change_rows: List[FactChangeRow] = build_change_facts(projects)

    rprint("[bold]5) Validate[/bold]")
    validate_price_constancy(run_rows)
    validate_pxq(run_rows)
    validate_project_totals(change_rows, projects)
    validate_project_allocation_constancy(projects)

    rprint("[bold]6) Persist CSVs[/bold]")
    _to_df(run_rows, FactRunRow).to_csv(OUT_DIR / "fact_run.csv", index=False)
    _to_df(change_rows, FactChangeRow).to_csv(OUT_DIR / "fact_change.csv", index=False)

    # Optional: persist dimension CSVs for loader convenience
    pd.DataFrame(SERVICES).to_csv(OUT_DIR / "dim_service.csv", index=False)
    pd.DataFrame(BUSINESS_UNITS).to_csv(OUT_DIR / "dim_org.csv", index=False)

    rprint("[bold]7) Render PDFs per Service & Project + Embed + Delete[/bold]")
    emb = Embedder(EMB_DIR)

    for s in SERVICES:
        sid = s["service_id"]
        for fy in FISCAL_YEARS:
            price_curr = next(p for p in prices if p.service_id == sid and p.fiscal_year == fy)
            price_prev: Optional[ServicePrice] = None
            if fy == "FY25":
                price_prev = next(p for p in prices if p.service_id == sid and p.fiscal_year == "FY24")

            meta = ServiceDocMeta(
                fiscal_year=fy,
                tower_id=s["tower_id"],
                service_id=sid,
                price_fy=price_curr.price,
                price_unit=f"{CURRENCY}/unit",
                price_delta_pct_vs_prev_fy=(None if price_prev is None else (price_curr.price/price_prev.price - 1.0)),
            )
            pdf_path = render_service_agreement_pdf(meta, price_curr, price_prev)
            chunks = [
                f"Overview & Scope: {s['service_name']} provides capability to BUs.",
                f"Pricing Model & Unit: Unit={s['unit']}, Billing=PxQ, price constant per FY.",
                (
                    f"Price Table (FY): price_fy={price_curr.price:.4f} {CURRENCY}/unit"
                    + (f", delta_vs_prev={(price_curr.price/price_prev.price - 1.0):.4f}" if price_prev else "")
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
                project_id=p.project_id,
                project_cost_fy24=p.cost_fy24,
                project_cost_fy25=p.cost_fy25,
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

    rprint("[bold]8) Build unified fact + write to DuckDB[/bold]")
    final_fact = _final_fact_df(run_rows, change_rows)

    # Expose DataFrame to DuckDB and create table in target DB file
    _write_duckdb(final_fact)

    rprint("[bold green]Pipeline complete.[/bold green]")


if __name__ == "__main__":
    run_pipeline()