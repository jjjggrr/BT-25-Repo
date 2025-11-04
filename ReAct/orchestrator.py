import argparse, json, re, os, time
from typing import Dict, Any, List, Tuple, Optional
from difflib import get_close_matches
import pandas as pd
import numpy as np
from datetime import datetime

from sympy import false

from llm_client import GeminiClient

from cube_client import CubeClient
from chroma_client import ChromaClient
from config import RESULTS_DIR
from cube_meta import load_schema_cache
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
#  GLOBALS
# ---------------------------------------------------------------------------

SCHEMA_CACHE = load_schema_cache(force_refresh=False)
VALID_VALUES = SCHEMA_CACHE.get("valid_values", {})
load_dotenv()
useLLM = false


# ---------------------------------------------------------------------------
#  HELPERS
# ---------------------------------------------------------------------------
def _round_dataframe(df, cols=None, digits=2):
    """Rundet numerische Spalten eines pandas-DataFrames (in place)."""


    if df is None or df.empty:
        return df

    if cols is None:
        cols = df.select_dtypes(include=["float", "int"]).columns

    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: round(x, digits) if isinstance(x, (float, int)) else x
            )
    return df

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

def normalize_fiscal_year(fy_str: str | int | None) -> str:
    """
    Normalisiert Eingaben wie '2024', 2025, 'FY24', '24' → 'FY24'.
    Nimmt an, dass FY = Kalenderjahr (vereinfacht).
    """
    if not fy_str:
        return f"FY{datetime.now().year % 100}"  # z. B. FY25

    s = str(fy_str).strip().upper()
    if s.startswith("FY"):
        return s
    # numerisch z. B. 2024 -> 24
    if s.isdigit():
        year = int(s)
        if year > 2000:
            return f"FY{year % 100}"
        else:
            return f"FY{year:02d}"
    # z. B. "24"
    if len(s) == 2 and s.isdigit():
        return f"FY{s}"
    return s

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

    # Define stopwords
    STOPWORDS = {"what", "the", "and", "for", "cost", "why", "high", "top", "unit", "are", "in", "of", "to", "by"}

    if valid_values:
        for dim, vals in valid_values.items():
            if not vals:
                continue
            low_vals = [v.lower() for v in vals]

            for tok_l in ngrams:
                if len(tok_l) < 4 or tok_l in STOPWORDS:
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
                    elif dim.startswith("DimApp."):
                        # treat applications as service-equivalent
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

def split_into_subqueries(q: str, parsed: dict) -> List[Dict[str, Any]]:
    """
    Heuristically split a complex question into sub-queries.
    Returns structured tasks the orchestrator can execute separately.
    """
    subs = []
    q_lower = q.lower()

    # 1) top cost drivers
    if "driver" in q_lower or "cost driver" in q_lower:
        subs.append({
            "type": "drivers",
            "org": parsed.get("org"),
            "fy_old": parsed.get("fy_old"),
            "fy_new": parsed.get("fy_new")
        })

    # 2) why is <service/app> high?
    if "why" in q_lower or "reason" in q_lower:
        if parsed.get("service"):
            subs.append({
                "type": "why_service",
                "org": parsed.get("org"),
                "service": parsed.get("service"),
                "service_dim": parsed.get("service_dimension")
            })

    # 3) total costs for a country
    if "country" in parsed and parsed.get("country"):
        subs.append({
            "type": "totals",
            "country": parsed.get("country"),
            "country_dim": parsed.get("country_dimension")
        })

    return subs


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

def _json_safe(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if pd.isna(obj):
        return None
    return obj

# ---------------------------------------------------------------------------
#  MAIN ORCHESTRATION LOGIC
# ---------------------------------------------------------------------------

def orchestrate(question: str) -> Dict[str, Any]:
    """
    Core orchestrator logic:
    - Parses query
    - Detects and executes multiple subqueries (ReAct-style)
    - Falls back to single-query pipeline
    - Writes both raw and LLM-formatted results
    """
    parsed = parse_question(question, VALID_VALUES)
    subqueries = split_into_subqueries(question, parsed)

    cube = CubeClient()
    chroma = ChromaClient()
    subquery_results = []  # für Multi-query-Flow

    # === Multi-query mode ===
    if subqueries:
        print(f"[Orchestrator] Detected {len(subqueries)} subqueries: {[s.get('type') for s in subqueries]}")

        for sub in subqueries:
            t = str(sub.get("type", "")).strip().lower()

            # --- 1) Drivers ---
            if t == "drivers":
                print(f"[Orchestrator] Executing driver query for {sub.get('org')} FY {sub.get('fy_old')} → {sub.get('fy_new')}")
                df = cube.top_cost_drivers(sub.get("org"), sub.get("fy_old"), sub.get("fy_new"), top_n=5)
                _round_dataframe(df, digits=2)  # <<–– hier
                totals = getattr(df, "attrs", {}).get("total_costs", {})
                # totals-Dict ebenfalls runden
                totals = {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in totals.items()}

                subquery_results.append({
                    "type": t,
                    "data": df.to_dict(orient="records"),
                    "totals": totals
                })
                continue

            # --- 2) Why Service Costs High ---
            if t == "why_service":
                print(f"[Orchestrator] Executing why_service query for {sub.get('service')} ({sub.get('service_dim')})")

                # 1) Numerik aus Cube.js
                df = cube.total_cost_by_service_fy(sub.get("org"), sub.get("service"), sub.get("service_dim"))
                _round_dataframe(df, digits=2)
                numeric_data = df.to_dict(orient="records")

                # delta_pct separat als Prozent formatieren
                for r in numeric_data:
                    if "delta_pct" in r and isinstance(r["delta_pct"], (float, int)):
                        r["delta_pct"] = f"{round(r['delta_pct'] * 100, 1)}%"

                # 2) FYs robust setzen
                fy_new = normalize_fiscal_year(sub.get("fy_new"))
                fy_old = normalize_fiscal_year(sub.get("fy_old"))
                if not fy_new and fy_old:
                    fy_new = fy_old
                    fy_old = None
                if fy_new and not fy_old:
                    # heuristik: wenn nur FY25 gegeben ist, nimm das Vorjahr als FY24
                    try:
                        y = int(fy_new.replace("FY", ""))
                        fy_old = f"FY{y - 1:02d}"
                    except Exception:
                        fy_old = fy_new  # fallback, not ideal but avoids None

                # 3) RAG-Query-Text
                query_text = f"Reasons for {sub.get('service')} cost change {fy_old} vs {fy_new}"

                # 4) RAG-Kontext (über App-Name, nicht service_id)
                context_strings = chroma.query_expanded_for_service(
                    question=query_text,
                    app_name=sub.get("service"),  # <<--- WICHTIG: App-Name übergeben
                    fiscal_years=[fy_old, fy_new],  # z. B. ["FY24","FY25"]
                    section_filter=["pricing", "changes", "sla"],
                    top_k_init=5,
                    max_snippets=6,
                    max_chars_per_snippet=320
                )

                # 5) Für format_for_llm() in Dicts verpacken
                context_docs = [{"text": s} for s in context_strings]

                subquery_results.append({
                    "type": t,
                    "numeric": numeric_data,
                    "textual": context_docs
                })
                continue

            # --- 3) Totals (z. B. Country) ---
            if t == "totals":
                print(f"[Orchestrator] Executing totals query for {sub.get('country')} ({sub.get('country_dim')})")
                df = cube.total_cost_by_country(sub.get("country"), sub.get("country_dim"))
                _round_dataframe(df, digits=2)
                subquery_results.append({
                    "type": t,
                    "data": df.to_dict(orient="records")
                })
                continue

        # === Combine & persist ===
        result = {
            "question": question,
            "parsed": parsed,
            "subquery_results": subquery_results,
            "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        # Write full result
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"result_{int(time.time())}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[Orchestrator] Multi-query result written to {out_path}")

        # === LLM Formatting ===
        llm_input = format_for_llm(result)
        llm_path = out_path.replace(".json", "_llm.json")
        with open(llm_path, "w", encoding="utf-8") as f:
            json.dump(llm_input, f, indent=2, default=_json_safe)
        print(f"[Orchestrator] LLM-formatted data written to {llm_path}")

        # === LLM Query ===
        if useLLM:
            from llm_client import GeminiClient

            try:
                llm_client = GeminiClient(model="gemini-2.5-flash")
                prompt = f"Given the following analytical results and context, answer this question: {question}"
                answer = llm_client.generate_answer(prompt=prompt, context=llm_input)
                answer_path = llm_path.replace("_llm.json", "_llm_answer.txt")
                with open(answer_path, "w", encoding="utf-8") as f:
                    f.write(answer)
                print(f"[Orchestrator] LLM answer written to {answer_path}")
            except Exception as e:
                print(f"[Orchestrator] LLM generation failed: {e}")

        return result, out_path

    # === Single-query fallback ===
    print("[Orchestrator] No explicit subqueries detected, running standard pipeline.")
    df_service = cube.total_cost_by_service_fy(parsed["org"], parsed["service"], parsed["service_dimension"])
    df_drivers = cube.top_cost_drivers(parsed["org"], parsed["fy_old"], parsed["fy_new"], top_n=parsed["intent"]["top_n"])
    df_delta = cube.cost_delta_summary(parsed["org"], parsed["fy_old"], parsed["fy_new"])
    queries = build_retrieval_queries(parsed)
    docs = chroma.query(queries, top_k=5)

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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"result_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[Orchestrator] Result written to {out_path}")

    llm_input = format_for_llm(result)
    llm_path = out_path.replace(".json", "_llm.json")
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_input, f, indent=2)
    print(f"[Orchestrator] LLM-formatted data written to {llm_path}")

    return result, out_path



def format_for_llm(result: dict) -> dict:
    """
    Convert multi-query or fallback result into LLM-friendly structure.
    Includes Top-5 union logic, pivoted cost comparisons, and delta metrics.
    """
    parsed = result.get("parsed", {})
    subs = result.get("subquery_results", [])

    out = {
        "summary": {
            "org": parsed.get("org"),
            "fiscal_years": [parsed.get("fy_old"), parsed.get("fy_new")],
            "country": parsed.get("country")
        },
        "drivers": [],
        "app_costs": {},
        "country_totals": {},
        "context_docs": []
    }

    for sub in subs:
        t = sub.get("type")

        # --- Drivers (Top 5 per FY, union) ---
        if t == "drivers" and sub.get("data"):
            df = pd.DataFrame(sub["data"])
            df = df.rename(columns={
                "DimService.serviceName": "service",
                "FctItCosts.fiscalYear": "fy",
                "FctItCosts.actualCost": "cost"
            })
            df["cost"] = df["cost"].astype(float)

            # Aggregate duplicates
            df_grouped = df.groupby(["service", "fy"], as_index=False)["cost"].sum()

            # Top5 per FY
            top_24 = (
                df_grouped[df_grouped["fy"] == "FY24"]
                .sort_values("cost", ascending=False)
                .head(5)["service"]
                .tolist()
            )
            top_25 = (
                df_grouped[df_grouped["fy"] == "FY25"]
                .sort_values("cost", ascending=False)
                .head(5)["service"]
                .tolist()
            )

            top_union = sorted(set(top_24 + top_25))
            df_filtered = df_grouped[df_grouped["service"].isin(top_union)]

            # Pivot + delta
            df_pivot = df_filtered.pivot(
                index="service", columns="fy", values="cost"
            ).fillna(value=pd.NA).reset_index()

            if "FY24" in df_pivot.columns and "FY25" in df_pivot.columns:
                df_pivot["delta_abs"] = df_pivot["FY25"] - df_pivot["FY24"]
                df_pivot["delta_pct"] = df_pivot["delta_abs"] / df_pivot["FY24"]

            # --- Optional: include total BU costs ---
            totals = sub.get("totals", {})
            if totals:
                out["drivers_summary"] = {
                    "org": totals.get("org"),
                    "total_FY24": float(totals.get("FY24") or 0),
                    "total_FY25": float(totals.get("FY25") or 0)
                }

            out["drivers"] = [
                {
                    "service": row["service"],
                    "FY24": float(row.get("FY24")) if "FY24" in df_pivot.columns else None,
                    "FY25": float(row.get("FY25")) if "FY25" in df_pivot.columns else None,
                    "delta_abs": float(row["delta_abs"]) if "delta_abs" in df_pivot.columns else None,
                    "delta_pct": float(row["delta_pct"]) if "delta_pct" in df_pivot.columns else None,
                }
                for _, row in df_pivot.iterrows()
            ]
            for d in out["drivers"]:
                for k in ["FY24", "FY25", "delta_abs"]:
                    if d.get(k) is not None:
                        d[k] = round(d[k], 2)
                if d.get("delta_pct") is not None:
                    d["delta_pct"] = f"{round(d['delta_pct'] * 100, 1)}%"


        # --- App Costs (e.g. Microsoft 365) ---
        elif t == "why_service" and sub.get("numeric"):
            df = pd.DataFrame(sub["numeric"])
            df = df.rename(columns={
                "DimApp.appName": "app",
                "FctItCosts.fiscalYear": "fy",
                "FctItCosts.actualCost": "cost",
                "FctItCosts.price": "price",
                "FctItCosts.quantity": "quantity"
            })
            # ensure numeric
            for c in ["cost", "price", "quantity"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # pivot cost per FY for readability
            pivot = df.pivot(index="app", columns="fy", values="cost").fillna(value=pd.NA).reset_index()

            # compute deltas and join average price/total quantity
            if "FY24" in pivot.columns and "FY25" in pivot.columns:
                pivot["delta_abs"] = pivot["FY25"] - pivot["FY24"]
                pivot["delta_pct"] = pivot["delta_abs"] / pivot["FY24"]

            out["app_costs"] = [
                {
                    "app": row["app"],
                    "FY24_cost": float(row.get("FY24")) if "FY24" in pivot.columns else None,
                    "FY25_cost": float(row.get("FY25")) if "FY25" in pivot.columns else None,
                    "FY24_price": df[df["fy"] == "FY24"]["price"].mean(),
                    "FY25_price": df[df["fy"] == "FY25"]["price"].mean(),
                    "FY24_quantity": df[df["fy"] == "FY24"]["quantity"].sum(),
                    "FY25_quantity": df[df["fy"] == "FY25"]["quantity"].sum(),
                    "delta_abs": float(row["delta_abs"]) if "delta_abs" in pivot.columns else None,
                    "delta_pct": float(row["delta_pct"]) if "delta_pct" in pivot.columns else None,
                }
                for _, row in pivot.iterrows()
            ]
            for a in out["app_costs"]:
                for k in [
                    "FY24_cost", "FY25_cost", "FY24_price", "FY25_price",
                    "FY24_quantity", "FY25_quantity", "delta_abs"
                ]:
                    if a.get(k) is not None:
                        a[k] = round(a[k], 2)
                if a.get("delta_pct") is not None:
                    a["delta_pct"] = f"{round(a['delta_pct'] * 100, 1)}%"

            if sub.get("textual"):
                out["context_docs"].extend([d["text"] for d in sub["textual"]])


        # --- Country Totals ---
        elif t == "totals" and sub.get("data"):
            for row in sub["data"]:
                name = row.get("DimCountry.countryName")
                if name:
                    val = float(row.get("FctItCosts.actualCost", 0))
                    out["country_totals"][name] = round(val, 2)

    return out


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