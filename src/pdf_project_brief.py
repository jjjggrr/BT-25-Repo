from __future__ import annotations
from datetime import datetime
from random import choice
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from .models import ProjectDocMeta, ProjectDef
from .config import PDF_DIR, CURRENCY

def render_project_brief_pdf(meta: ProjectDocMeta, project: ProjectDef) -> Path:
    fname = f"PRJ_{project.project_id}_{meta.fiscal_year}.pdf"
    fpath = PDF_DIR / fname

    c = canvas.Canvas(str(fpath), pagesize=A4)
    w, h = A4
    y = h - 2*cm

    def section(title: str, lines: list[str]):
        nonlocal y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, title)
        y -= 0.5*cm
        c.setFont("Helvetica", 10)
        for line in lines:
            c.drawString(2*cm, y, line)
            y -= 0.45*cm
        y -= 0.3*cm

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, y, f"Project Brief – {project.name} ({project.project_id}) {meta.fiscal_year}")
    y -= 1.2*cm

    # 1) Contract Overview
    section("1. Contract Overview", [
        f"This document outlines the project agreement between ExampleCompany AG and involved suppliers for '{project.name}'.",
        "The project is governed under internal Change Tower (TWR-06) and funded as a CHANGE initiative."
    ])

    # 2) Objectives & Deliverables
    section("2. Objectives & Deliverables", [
        "Objectives: modernize, consolidate and automate relevant IT capabilities.",
        "Deliverables: technical implementation, documentation, knowledge transfer.",
        "Expected outcome: operational efficiency and cost reduction across Business Units."
    ])

    # 3) Timeline & Milestones
    section("3. Timeline & Milestones", [
        "Design Phase: FY24 Q1",
        "Build & Rollout: FY24 Q2–Q3",
        "Transition to Operations: FY24 Q4",
    ])

    # 4) Budget & Yearly Costs
    section("4. Budget & Yearly Costs", [
        f"FY24: {project.cost_fy24:,.2f} {CURRENCY} | FY25: {project.cost_fy25:,.2f} {CURRENCY}",
        f"New in FY25: {'Yes' if (project.exists_fy25 and not project.exists_fy24) else 'No'}",
    ])

    # 5) Business Unit Allocation
    lines = [f"{alloc.org_id}: {alloc.share*100:.2f}%" for alloc in project.allocation]
    section("5. Business Unit Allocation", lines or ["No allocations available."])

    # 6) Dependencies & Risks
    section("6. Dependencies & Risks", [
        "Depends on underlying hosting and identity platforms.",
        "Risks: schedule slippage, dependency on vendor delivery, integration complexity."
    ])

    # 7) Expected Benefits
    section("7. Expected Benefits", [
        "Improved process automation and reduced manual work (~10–15%).",
        "Lower operating costs due to platform consolidation.",
        "Enhanced service transparency and reporting capabilities."
    ])

    c.setFont("Helvetica", 8)
    c.drawString(2*cm, 1.5*cm, f"Generated: {datetime.utcnow().isoformat()}Z | Source: {fname}")
    c.save()
    return fpath
