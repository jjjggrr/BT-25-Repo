import argparse
import json
import os
import re
import time
from typing import Dict, Any, List, Tuple

from chroma_client import ChromaClient
from config import RESULTS_DIR
from cube_client import CubeClient

ORG_RE = re.compile(r"(Org|ORG)[-_ ]?(\d{1,4})")
FY_RE  = re.compile(r"FY(\d{2})")
SERVICE_GUESSES = ["Microsoft 365","M365","IaaS Compute","Storage","CRM","SAP"]

def parse_question(q:str) -> Dict[str,Any]:
    q_norm = q.strip()

    org = None
    m = ORG_RE.search(q_norm)
    if m:
        num = m.group(2).zfill(3)
        org = f"Org{num}"

    fys = FY_RE.findall(q_norm)
    fy_old, fy_new = None, None
    if len(fys) >= 2:
        fy_old = f"FY{fys[0]}"
        fy_new = f"FY{fys[1]}"
    elif len(fys) == 1:
        fy_new = f"FY{fys[0]}"

    service = None
    for guess in SERVICE_GUESSES:
        if guess.lower() in q_norm.lower():
            service = "Microsoft 365" if guess.lower() in ["m365","microsoft 365"] else guess
            break

    intent = {
        "top_n": 5 if "top 5" in q_norm.lower() else 5,
        "ask_drivers": "driver" in q_norm.lower() or "why" in q_norm.lower()
    }

    return {"org": org, "fy_old": fy_old, "fy_new": fy_new, "service": service, "intent": intent}

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
        parsed = parse_question(question)
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
