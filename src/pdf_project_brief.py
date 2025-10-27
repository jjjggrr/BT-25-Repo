from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import List
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from .models import ProjectDocMeta, ProjectDef
from .config import PDF_DIR, CURRENCY


def render_project_brief_pdf(meta: ProjectDocMeta, project: ProjectDef) -> Path:
    """Creates a 2-page Project Brief PDF with section-based content.
    Sections: Summary, Budget & Yearly Costs, BU Allocation, Changes vs Prev FY, Dependencies
    """
    fname = f"PRJ_{project.project_id}_{meta.fiscal_year}.pdf"
    fpath = PDF_DIR / fname

    c = canvas.Canvas(str(fpath), pagesize=A4)
    w, h = A4

    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, h-2*cm, f"Project Brief – {project.name} ({project.project_id}) {meta.fiscal_year}")

    # 1) Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, h-3.2*cm, "1. Project Summary")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, h-4.0*cm, "Purpose: Deliver scope to improve IT capability across BUs.")

    # 2) Budget & Yearly Costs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, h-5.5*cm, "2. Budget & Yearly Costs")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, h-6.2*cm, f"FY24: {project.cost_fy24:,.2f} {CURRENCY} | FY25: {project.cost_fy25:,.2f} {CURRENCY}")
    c.drawString(2*cm, h-7.0*cm, f"New in FY25: {'Yes' if (project.exists_fy25 and not project.exists_fy24) else 'No'}")

    # Page 2
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, h-2*cm, "3. BU Allocation (stable across FYs)")
    c.setFont("Helvetica", 10)
    y = h-3*cm
    for alloc in project.allocation:
        c.drawString(2*cm, y, f"{alloc.org_id}: {alloc.share*100:.2f}%")
        y -= 0.6*cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y-0.8*cm, "4. Changes vs Previous FY")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y-1.6*cm, "Scope continues with adjusted phasing; costs reflect project roadmap.")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y-3.2*cm, "5. Dependencies & Risks")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y-4.0*cm, "No material risks identified for the current fiscal year.")

    c.setFont("Helvetica", 8)
    c.drawString(2*cm, 1.5*cm, f"Generated: {datetime.utcnow().isoformat()}Z | Source: {fname}")
    c.save()

    return fpath