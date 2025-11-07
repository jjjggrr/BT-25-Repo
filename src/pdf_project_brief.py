from __future__ import annotations
from datetime import datetime
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from .models import ProjectDocMeta, ProjectDef
from .config import PDF_DIR, CURRENCY
from src.config import BUSINESS_UNITS

bu_name_map = {bu["org_id"]: bu["business_unit_name"] for bu in BUSINESS_UNITS}

def render_project_brief_pdf(meta: ProjectDocMeta, project: ProjectDef) -> Path:
    safe_name = project.name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    fname = f"PRJ_{safe_name}_{meta.fiscal_year}.pdf"
    fpath = PDF_DIR / fname

    c = canvas.Canvas(str(fpath), pagesize=A4)
    w, h = A4
    y = h - 2 * cm

    def section(marker: str, title: str, lines: list[str]):
        nonlocal y
        c.setFont("Helvetica", 9)
        c.drawString(2 * cm, y, f"###SECTION: {marker}")
        y -= 0.4 * cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, title)
        y -= 0.5 * cm
        c.setFont("Helvetica", 10)
        for line in lines:
            c.drawString(2 * cm, y, line)
            y -= 0.45 * cm
        y -= 0.3 * cm

    # === Header ===
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, f"Project Brief – {project.name} ({project.project_id}) {meta.fiscal_year}")
    y -= 1.2 * cm

    # === Meta ===
    meta_lines = [
        f"Meta:",
        f"ProjectID: {project.project_id}",
        f"ProjectName: {project.name}",
        f"FiscalYear: {meta.fiscal_year}",
        f"FundingType: CHANGE",
        f"Currency: {CURRENCY}",
        f"DocType: ProjectBrief",
    ]
    for line in meta_lines:
        c.setFont("Helvetica", 9)
        c.drawString(2 * cm, y, line)
        y -= 0.4 * cm
    y -= 0.3 * cm

    # 1) Contract Overview
    section("contract_overview", "1. Contract Overview", [
        f"This document outlines the project agreement between ExampleCompany AG and suppliers for '{project.name}'.",
        "The project is governed under internal Change Tower (TWR-06) and funded as a CHANGE initiative."
    ])

    # 2) Objectives
    section("objectives", "2. Objectives & Deliverables", [
        "Objectives: modernize, consolidate and automate relevant IT capabilities.",
        "Deliverables: implementation, documentation, knowledge transfer.",
        "Expected outcome: operational efficiency and cost reduction."
    ])

    # 3) Timeline
    section("timeline", "3. Timeline & Milestones", [
        "Design Phase: FY24 Q1",
        "Build & Rollout: FY24 Q2–Q3",
        "Transition to Operations: FY24 Q4",
    ])

    # 4) Budget
    section("budget", "4. Budget & Yearly Costs", [
        f"FY24Cost: {project.cost_fy24:,.2f} {CURRENCY}",
        f"FY25Cost: {project.cost_fy25:,.2f} {CURRENCY}",
        f"NewProjectFY25: {'Yes' if (project.exists_fy25 and not project.exists_fy24) else 'No'}",
    ])

    # 5) Business Unit Allocation
    lines = [f"{alloc.org_id} ({bu_name_map.get(alloc.org_id, 'Unknown')}): {alloc.share*100:.2f}%" for alloc in project.allocation]
    section("allocation", "5. Business Unit Allocation", lines or ["No allocations available."])

    # 6) Dependencies
    section("dependencies", "6. Dependencies & Risks", [
        "Depends on hosting and identity platforms.",
        "Risks: schedule slippage, vendor dependency, integration complexity."
    ])

    # 7) Benefits
    section("benefits", "7. Expected Benefits", [
        "Improved process automation (~10–15%).",
        "Reduced operating costs through consolidation.",
        "Enhanced transparency and reporting capabilities."
    ])

    c.setFont("Helvetica", 8)
    c.drawString(2 * cm, 1.5 * cm, f"Generated: {datetime.utcnow().isoformat()}Z | Source: {fname}")
    c.save()
    return fpath
