import argparse, json, re, os, time
from typing import Dict, Any, List, Tuple, Optional
from difflib import get_close_matches

from cube_client import CubeClient
from chroma_client import ChromaClient
from config import RESULTS_DIR
from cube_meta import load_schema_cache


# ---------------------------------------------------------------------------
#  GLOBALS
# ---------------------------------------------------------------------------

SCHEMA_CACHE = load_schema_cache(force_refresh=False)
VALID_VALUES = SCHEMA_CACHE.get("valid_values", {})


# ---------------------------------------------------------------------------
#  HELPERS
# ---------------------------------------------------------------------------

def normalize_value(value: str, dim_name: str, valid_values: dict) -> str:
    """Fuzzy-match a value to the closest known valid value for the dimension."""
    if not value or dim_name not in valid_values:
        return value
    vals = valid_values[dim_name]
    match = get_close_matches(value.lower(), [v.lower() for v in vals], n=1, cutoff=0.7)
    if not match:
        return value
    # get exact casing from schema values
    idx = [v.lower() for v in vals].index(match[0])
    return vals[idx]


# ---------------------------------------------------------------------------
#  PARSER
# ---------------------------------------------------------------------------

def parse_question(q: str, valid_values: dict | None = None) -> Dict[str, Any]:
    """
    Parses a natural-language query and identifies:
      - Organization (orgId or businessUnit)
      - Service (DimService.serviceName)
      - Country (DimCountry.countryName)
      - Project (DimProject.projectName)
      - Fiscal years (FY)
      - Intent flags
    Uses fuzzy matching against schema_cache valid values.
    """
    from difflib import get_close_matches
    import re

    q_norm = q.strip().lower()

    # ---------------- FY extraction ----------------
    fys = sorted(set(re.findall(r"fy\s?(\d{2})", q_norm)))
    fy_old, fy_new = (None, None)
    if len(fys) >= 2:
        fy_old, fy_new = f"FY{fys[0]}", f"FY{fys[1]}"
    elif len(fys) == 1:
        fy_new = f"FY{fys[0]}"

    # ---------------- Tokenize query ----------------
    tokens = re.findall(r"[a-zA-Z0-9_+]+", q_norm)
    tokens += [" ".join(pair) for pair in zip(tokens, tokens[1:])]  # bigrams

    detected: Dict[str, Dict[str, Optional[Any]]] = {
        "org": {"value": None, "dim": None},
        "service": {"value": None, "dim": None},
        "country": {"value": None, "dim": None},
        "project": {"value": None, "dim": None},
    }

    # ---------------- Exact pattern for ORG_xxx ----------------
    m = re.search(r"(org[_-]?\d{1,4})", q_norm, re.IGNORECASE)
    if m:
        detected["org"]["value"] = m.group(1).upper()
        detected["org"]["dim"] = "DimOrg.orgId"

    # ---------------- Fuzzy matching across dimensions ----------------
    # Build candidate ngrams once (outside the dim loop)
    tokens = re.findall(r"[a-zA-Z0-9_+]+", q_norm)
    ngrams = tokens + [" ".join(p) for p in zip(tokens, tokens[1:])]  # bigrams
    ngrams += [" ".join(p) for p in zip(tokens, tokens[1:], tokens[2:])]  # trigrams
    ngrams = [t.lower() for t in ngrams]

    if valid_values:
        for dim, vals in valid_values.items():
            if not vals:
                continue
            low_vals = [v.lower() for v in vals]
            for tok_l in ngrams:  # iterate over ngrams, not tokens
                if len(tok_l) < 4:
                    continue
                if tok_l in {"what", "the", "and", "for", "cost", "why", "high", "top", "unit", "are"}:
                    continue

                match = None
                # direct substring match
                for lv in low_vals:
                    if tok_l in lv:
                        match = lv
                        break

                # fuzzy match fallback
                if not match:
                    fuzzy = get_close_matches(tok_l, low_vals, n=1, cutoff=0.7)
                    if fuzzy:
                        match = fuzzy[0]

                if match:
                    idx = low_vals.index(match)
                    v = vals[idx]
                    if dim.startswith("DimOrg."):
                        detected["org"]["value"], detected["org"]["dim"] = v, dim
                    elif dim.startswith("DimService."):
                        detected["service"]["value"], detected["service"]["dim"] = v, dim
                    elif dim.startswith("DimCountry."):
                        detected["country"]["value"], detected["country"]["dim"] = v, dim
                    elif dim.startswith("DimProject."):
                        detected["project"]["value"], detected["project"]["dim"] = v, dim

    # ---------------- Intent ----------------
    m_top = re.search(r"top\s+(\d+)", q_norm)
    top_n = int(m_top.group(1)) if m_top else 5
    ask_drivers = any(w in q_norm for w in ["driver", "why", "reason", "cause"])

    return {
        "org": detected["org"]["value"],
        "org_dimension": detected["org"]["dim"],
        "service": detected["service"]["value"],
        "service_dimension": detected["service"]["dim"],
        "country": detected["country"]["value"],
        "country_dimension": detected["country"]["dim"],
        "project": detected["project"]["value"],
        "project_dimension": detected["project"]["dim"],
        "fy_old": fy_old,
        "fy_new": fy_new,
        "intent": {"top_n": top_n, "ask_drivers": ask_drivers},
    }



# ---------------------------------------------------------------------------
#  BUILD RETRIEVAL QUERIES
# ---------------------------------------------------------------------------

def build_retrieval_queries(parsed: Dict[str, Any]) -> List[str]:
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


# ---------------------------------------------------------------------------
#  MAIN ORCHESTRATION LOGIC
# ---------------------------------------------------------------------------

def orchestrate(question: str) -> Tuple[Dict[str, Any], str]:
    parsed = parse_question(question, VALID_VALUES)
    cube = CubeClient()
    chroma = ChromaClient()

    # --- Extract parameters ---
    org = parsed["org"]
    org_dim = parsed["org_dimension"] or "DimOrg.businessUnit"
    fy_old = parsed["fy_old"] or "FY24"
    fy_new = parsed["fy_new"] or "FY25"
    service = parsed["service"]
    service_dim = parsed["service_dimension"] or "DimService.serviceName"
    top_n = parsed["intent"]["top_n"]

    print(f"[Orchestrator] Parsed: org={org} ({org_dim}), service={service}, FYs={fy_old}->{fy_new}")

    # --- Numeric retrievals ---
    df_service = cube.total_cost_by_service_fy(org, service)
    df_drivers = cube.top_cost_drivers(org, fy_old, fy_new, top_n=top_n)
    df_delta = cube.cost_delta_summary(org, fy_old, fy_new)

    # --- Textual retrievals ---
    queries = build_retrieval_queries(parsed)
    docs = chroma.query(queries, k=5)

    # --- Combine results ---
    result = {
        "question": question,
        "parsed": parsed,
        "numeric": {
            "service_costs_fy": df_service.to_dict(orient="records"),
            "top_cost_drivers": df_drivers.to_dict(orient="records"),
            "delta_summary": df_delta.to_dict(orient="records"),
        },
        "textual": docs,
        "provenance": {
            "service_costs_fy": getattr(df_service, "attrs", {}).get("provenance", {}),
            "top_cost_drivers": getattr(df_drivers, "attrs", {}).get("provenance", {}),
            "delta_summary": getattr(df_delta, "attrs", {}).get("provenance", {}),
            "queries": queries,
            "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    # --- Persist ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"result_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[Orchestrator] Result written to {out_path}")
    return result, out_path


# ---------------------------------------------------------------------------
#  ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="English sentence query")
    args = ap.parse_args()
    res, path = orchestrate(args.query)
    print(json.dumps(res, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()