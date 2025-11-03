from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional
from random import uniform, choice
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from .models import ServicePrice
from .config import PDF_DIR, DIM_APPS, SERVICES, APP_SERVICE_MAP, CURRENCY


def _random_sla_text() -> list[str]:
    uptime = round(uniform(99.7, 99.95), 2)
    resp = choice(["P1 ≤ 1h / P2 ≤ 4h / P3 ≤ 8h", "Critical ≤ 1h, High ≤ 4h, Normal ≤ 8h"])
    reso = choice(["≥ 95% incidents resolved within SLA", "≥ 97% within SLA targets"])
    window = choice(["Sundays 02:00–04:00 CET", "Saturdays 01:00–03:00 CET"])
    return [
        f"Availability: {uptime}% monthly uptime excluding maintenance.",
        f"Incident Response: {resp}.",
        f"Resolution Compliance: {reso}.",
        f"Maintenance Window: {window}.",
    ]


def _service_name_for_app(app_id: str) -> str:
    service_id = APP_SERVICE_MAP.get(app_id)
    if not service_id:
        return "Generic IT Service"
    service = next((s for s in SERVICES if s["service_id"] == service_id), None)
    return service["service_name"] if service else "Unassigned Service"


def render_service_agreement_pdf(
    app: dict,
    fiscal_year: str,
    price_curr: Optional[ServicePrice] = None,
    price_prev: Optional[ServicePrice] = None,
) -> Path:
    app_id = app["app_id"]
    app_name = app["app_name"]
    vendor = app["vendor"]
    service_name = _service_name_for_app(app_id)
    fname = f"SLA_{app_id}_{fiscal_year}.pdf"
    fpath = PDF_DIR / fname

    doc = SimpleDocTemplate(
        str(fpath),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    normal = ParagraphStyle("NormalTight", parent=styles["Normal"], spaceAfter=6)
    section_title = styles["Heading2"]

    story = []

    # === Meta Block ===
    story.append(Paragraph(f"Service Level Agreement – {app_name} ({app_id}) {fiscal_year}", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))
    meta_text = (
        f"Meta:\n"
        f"ServiceID: {app_id}\n"
        f"ServiceName: {service_name}\n"
        f"Vendor: {vendor}\n"
        f"FiscalYear: {fiscal_year}\n"
        f"Currency: {CURRENCY}\n"
        f"Tower: {service_name.split()[0] if service_name else 'N/A'}\n"
        f"DocType: SLA\n"
    )
    story.append(Paragraph(f"<pre>{meta_text}</pre>", normal))
    story.append(Spacer(1, 0.5 * cm))

    # === 1) Contract Overview ===
    story.append(Paragraph("###SECTION: contract_overview", normal))
    story.append(Paragraph("1. Contract Overview", section_title))
    story.append(
        Paragraph(
            f"This Service Level Agreement (SLA) is a contract between <b>{vendor}</b> and ExampleCompany AG "
            f"regarding the delivery of <b>{app_name}</b> as part of the <b>{service_name}</b> service. "
            "The agreement defines performance obligations, service boundaries, and financial terms. "
            "All prices and metrics are valid for the stated fiscal year and may be revised annually.",
            normal,
        )
    )

    # === 2) Service Scope ===
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("###SECTION: service_scope", normal))
    story.append(Paragraph("2. Service Scope & Applicability", section_title))
    story.append(
        Paragraph(
            f"The {app_name} application is provided as part of the enterprise {service_name} offering. "
            "Scope includes operational support, maintenance, and minor enhancement activities. "
            "Major changes require separate Change Requests under Tower governance.",
            normal,
        )
    )

    # === 3) Service Level Objectives ===
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("###SECTION: sla", normal))
    story.append(Paragraph("3. Service Level Objectives", section_title))
    for line in _random_sla_text():
        story.append(Paragraph(line, normal))

    # === 4) Pricing & Financial Model ===
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("###SECTION: pricing", normal))
    story.append(Paragraph("4. Pricing & Financial Model", section_title))
    delta_txt = ""
    reason = "No change"
    if price_prev and price_curr:
        delta = price_curr.price / price_prev.price - 1.0
        delta_txt = f"Δ {(delta) * 100:+.2f}% vs {price_prev.fiscal_year}"
        if delta < 0:
            reason = "Productivity improvements and cost optimization."
        elif delta > 0:
            reason = "Vendor pricing adjustment or new license model."

    story.append(
        Paragraph(
            f"PricingBasis: per_user_month<br/>"
            f"PriceChange: {delta_txt or 'Stable pricing FY over FY'}<br/>"
            f"Rationale: {reason}<br/>"
            f"BillingCurrency: {CURRENCY}",
            normal,
        )
    )

    # === 5) Dependencies ===
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("###SECTION: dependencies", normal))
    story.append(Paragraph("5. Dependencies & Integration Points", section_title))
    story.append(
        Paragraph(
            "Depends on underlying identity, hosting and network services. "
            "Integrates with central ITSM and monitoring platforms for availability tracking. "
            "Requires active authentication within ExampleCompany’s enterprise directory.",
            normal,
        )
    )

    # === 6) Change History ===
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("###SECTION: changes", normal))
    story.append(Paragraph("6. Change History & FY Comparison", section_title))
    story.append(
        Paragraph(
            "No major feature additions; focus on stability and operational excellence. "
            "Minor performance optimizations introduced in Q2 of the fiscal year. "
            "Cost structure aligned with global vendor framework agreement.",
            normal,
        )
    )

    # === 7) Operational Notes ===
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("###SECTION: operations", normal))
    story.append(Paragraph("7. Operational & Governance Notes", section_title))
    story.append(
        Paragraph(
            "Support hours: Monday–Friday 08:00–18:00 CET (excluding public holidays). "
            "Incident escalation path: Tier 1 – Service Desk → Tier 2 – Application Ops → Tier 3 – Vendor. "
            "Monthly SLA and performance review held with service owner and vendor representatives.",
            normal,
        )
    )

    # === Footer ===
    story.append(Spacer(1, 0.8 * cm))
    story.append(
        Paragraph(
            f"<font size=8>Generated: {datetime.utcnow().isoformat()}Z | Source: {fname}</font>",
            styles["Normal"],
        )
    )

    doc.build(story)
    return fpath
