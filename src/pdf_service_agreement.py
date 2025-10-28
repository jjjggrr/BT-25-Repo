from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from .models import ServiceDocMeta, ServicePrice
from .config import PDF_DIR, SERVICES, DEFAULT_PRICE_DELTAS, CURRENCY


def _service_info(service_id: str):
    for s in SERVICES:
        if s["service_id"] == service_id:
            return s
    raise KeyError(service_id)


def render_service_agreement_pdf(meta: ServiceDocMeta, price_curr: ServicePrice, price_prev: Optional[ServicePrice]) -> Path:
    """Creates a 2-page Service Agreement PDF with section-based content.
    Sections: Overview & Scope, Pricing Model & Unit, Price Table (FY), Change Log vs Prev FY, Notes
    """
    s = _service_info(meta.service_id)
    fname = f"SA_{meta.service_id}_{meta.fiscal_year}.pdf"
    fpath = PDF_DIR / fname

    c = canvas.Canvas(str(fpath), pagesize=A4)
    w, h = A4

    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, h-2*cm, f"Service Agreement – {s['service_name']} ({meta.service_id}) {meta.fiscal_year}")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, h-3*cm, f"Tower: {meta.tower_id} | Unit: {s['unit']} | Currency: {CURRENCY}")

    # 1) Overview & Scope
    text = c.beginText(2*cm, h-4*cm)
    text.setFont("Helvetica-Bold", 12)
    text.textLine("1. Overview & Scope")
    text.setFont("Helvetica", 10)
    text.textLines([
        f"This Service provides {s['service_name']} capabilities to Business Units.",
        "It is charged as a RUN (PxQ) service with monthly measurement.",
    ])
    c.drawText(text)

    # 2) Pricing Model & Unit
    text = c.beginText(2*cm, h-8*cm)
    text.setFont("Helvetica-Bold", 12)
    text.textLine("2. Pricing Model & Unit")
    text.setFont("Helvetica", 10)
    text.textLines([
        f"Unit Definition: {s['unit']}",
        "Billing: Quantity (Q) times Price (P) – price is constant per fiscal year.",
    ])
    c.drawText(text)

    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, h-2*cm, "3. Price Table (FY)")
    c.setFont("Helvetica", 10)
    prev_txt = f" vs prev FY ({price_prev.fiscal_year}): Δ {(price_curr.price/price_prev.price - 1.0)*100:.2f}%" if price_prev else ""
    c.drawString(2*cm, h-3*cm, f"Price {meta.fiscal_year}: {price_curr.price:.4f} {CURRENCY}/unit{prev_txt}")

    # 4) Change Log vs Previous FY
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, h-5*cm, "4. Change Log vs Previous FY")
    c.setFont("Helvetica", 10)
    delta_pct = None
    if price_prev:
        delta_pct = price_curr.price/price_prev.price - 1.0
    reason = "Productivity improvement" if (delta_pct is not None and delta_pct < 0) else (
        "Vendor price increase" if (delta_pct is not None and delta_pct > 0) else "No change"
    )
    c.drawString(2*cm, h-6*cm, f"Reason: {reason}")

    # 5) Notes
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, h-8*cm, "5. Operational Notes")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, h-9*cm, "No minimum commitments. SLAs unchanged.")

    c.setFont("Helvetica", 8)
    c.drawString(2*cm, 1.5*cm, f"Generated: {datetime.utcnow().isoformat()}Z | Source: {fname}")
    c.save()

    return fpath