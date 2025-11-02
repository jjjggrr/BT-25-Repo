import argparse
import json
import os
import re
import time
from typing import Dict, Any, List, Tuple

from cube_client import CubeClient
from chroma_client import ChromaClient
from config import RESULTS_DIR
from cube_meta import load_schema_cache
from difflib import get_close_matches

SCHEMA_CACHE = load_schema_cache(force_refresh=False)
VALID_VALUES = SCHEMA_CACHE.get("valid_values", {})

def normalize_value(value, dimension_name):
    """Return closest valid value for a given dimension"""
    vals = VALID_VALUES.get(dimension_name, [])
    if not vals:
        return value
    match = get_close_matches(value, vals, n=1, cutoff=0.7)
    return match[0] if match else value


ORG_RE = re.compile(r"(ORG[_-]?\d{1,4})", re.IGNORECASE)
FY_RE  = re.compile(r"FY(\d{2})")
SERVICE_GUESSES = ["Microsoft 365","M365","IaaS Compute","Storage","CRM","SAP"]

def parse_question(q: str, valid_values: dict | None = None) -> Dict[str, Any]:
    """
    Parses a natural-language query and tries to identify:
      - organization (orgId or businessUnit)
      - fiscal years (FY)
      - service name
      - intent flags (top_n, ask_drivers)
    Uses fuzzy matching against schema_cache valid values.
    """
    import re
    from difflib import get_close_matches

    q_norm = q.strip().lower()

    # ---------------- FY extraction ----------------
    fys = sorted(set(re.findall(r"fy\s?(\d{2})", q_norm)))
    fy_old, fy_new = (None, None)
    if len(fys) >= 2:
        fy_old, fy_new = f"FY{fys[0]}", f"FY{fys[1]}"
    elif len(fys) == 1:
        fy_new = f"FY{fys[0]}"

    # ---------------- Service extraction ----------------
    SERVICE_GUESSES = ["Microsoft 365", "M365", "SAP", "CRM", "Storage", "IaaS Compute"]
    service = None
    for guess in SERVICE_GUESSES:
        if guess.lower() in q_norm:
            service = "Microsoft 365" if guess.lower() in ["m365", "microsoft 365"] else guess
            break
    matched_service_dim = "DimService.serviceName" if service else None

    # ---------------- Org extraction (ID or BU) ----------------
    org = None
    matched_org_dim = None
    ORG_PATTERN = re.compile(r"(org[_-]?\d{1,4})", re.IGNORECASE)

    # Direct pattern match (ORG_001 etc.)
    m = ORG_PATTERN.search(q_norm)
    if m:
        org = m.group(1).upper()
        matched_org_dim = "DimOrg.orgId"

    # Fuzzy match against valid values (if cache present)
    if valid_values and not org:
        org_dims = [d for d in valid_values.keys() if d.startswith("DimOrg.")]
        for dim in org_dims:
            values = valid_values.get(dim, [])
            match = get_close_matches(q_norm, [v.lower() for v in values], n=1, cutoff=0.6)
            if match:
                idx = [v.lower() for v in values].index(match[0])
                org = values[idx]
                matched_org_dim = dim
                break

    # ---------------- Intent extraction ----------------
    top_n = 5
    m_top = re.search(r"top\s+(\d+)", q_norm)
    if m_top:
        try:
            top_n = int(m_top.group(1))
        except ValueError:
            pass
    ask_drivers = "driver" in q_norm or "why" in q_norm or "reason" in q_norm

    # ---------------- Build result ----------------
    return {
        "org": org,
        "org_dimension": matched_org_dim,
        "service": service,
        "service_dimension": matched_service_dim,
        "fy_old": fy_old,
        "fy_new": fy_new,
        "intent": {"top_n": top_n, "ask_drivers": ask_drivers},
    }


def build_retrieval_queries(parsed:Dict[str,Any]) -> List[str]:
    org = parsed.get("org") or ""
    service = parsed.get("service") or ""
    fy_old = parsed.get("fy_old") or "FY24"
    fy_new = parsed.get("fy_new") or "FY25"
    base = [
        f"{service} pricing change {fy_new} {fy_old} {org}",
        f"{service} productivity change {fy_new} {org}",
        f"new project {fy_new} {org}",
        f"vendor renegotiation {fy_new} {org}",
        f"scope change {fy_new} {org}"
    ]
    return base

def orchestrate(question:str) -> Tuple[Dict[str, Any], str]:
        parsed = parse_question(question, VALID_VALUES)
        cube = CubeClient()
        chroma = ChromaClient()

        # Defaults if not provided
        org = parsed["org"] or "Org001"
        fy_old = parsed["fy_old"] or "FY24"
        fy_new = parsed["fy_new"] or "FY25"
        service = parsed["service"] or "Microsoft 365"
        top_n = parsed["intent"]["top_n"]

        # Numeric retrievals
        df_service = cube.total_cost_by_service_fy(org, service)
        df_drivers = cube.top_cost_drivers(org, fy_old, fy_new, top_n=top_n)
        df_delta = cube.cost_delta_summary(org, fy_old, fy_new)

        # Text retrievals
        queries = build_retrieval_queries(parsed)
        docs = chroma.query(queries, k=5)

        result = {
            "question": question,
            "parsed": parsed,
            "numeric": {
                "service_costs_fy": df_service.to_dict(orient="records"),
                "top_cost_drivers": df_drivers.to_dict(orient="records"),
                "delta_summary": df_delta.to_dict(orient="records")
            },
            "textual": docs,
            "provenance": {
                "service_costs_fy": getattr(df_service, "attrs", {}).get("provenance", {}),
                "top_cost_drivers": getattr(df_drivers, "attrs", {}).get("provenance", {}),
                "delta_summary": getattr(df_delta, "attrs", {}).get("provenance", {}),
                "queries": queries,
                "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        }
        # Persist
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"result_{int(time.time())}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result, out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="English sentence query")
    args = ap.parse_args()
    res, path = orchestrate(args.query)
    print(json.dumps(res, indent=2))
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
