import argparse, json, re, os, time
from typing import Dict, Any, List, Optional
from difflib import get_close_matches
from datetime import datetime
import time

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from cube_client import CubeClient
from chroma_client import ChromaClient
from config import RESULTS_DIR
from cube_meta import load_schema_cache, build_llm_schema

# Optional: LLM-Client (nur genutzt, wenn USE_LLM=true)
try:
    from llm_client import GeminiClient
except Exception:
    GeminiClient = None

# ---------------------------------------------------------------------------
#  GLOBALS
# ---------------------------------------------------------------------------

load_dotenv()
SCHEMA_CACHE = load_schema_cache(force_refresh=False)
VALID_VALUES = SCHEMA_CACHE.get("valid_values", {})

# Schalter für LLM-Modus (default: False)
useLLM = os.getenv("USE_LLM", "false").lower() == "true"
llm_mode = os.getenv("LLM_MODE", "full")  # "generate_queries", "interpret_results", "full"

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
            df[c] = df[c].apply(lambda x: round(x, digits) if isinstance(x, (float, int)) else x)
    return df


def normalize_value(value: str, dim_name: str, valid_values: dict) -> str:
    """Fuzzy-match a value to the closest known valid value for the dimension."""
    if not value or dim_name not in valid_values:
        return value
    vals = valid_values[dim_name]
    match = get_close_matches(value.lower(), [v.lower() for v in vals], n=1, cutoff=0.7)
    if not match:
        return value
    idx = [v.lower() for v in vals].index(match[0])
    return vals[idx]

def normalize_fiscal_year(fy_str: str | int | None) -> str:
    """
    Normalisiert Eingaben wie '2024', 2025, 'FY24', '24' → 'FY24'.
    Nimmt an, dass FY = Kalenderjahr (vereinfacht).
    """
    if not fy_str:
        return f"FY{datetime.now().year % 100}"
    s = str(fy_str).strip().upper()
    if s.startswith("FY"):
        return s
    if s.isdigit():
        year = int(s)
        if year > 2000:
            return f"FY{year % 100}"
        else:
            return f"FY{year:02d}"
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
      - Service/App (DimService.serviceName / DimApp.appName)
      - Country (DimCountry.countryName)
      - Project (DimProject.projectName)
      - Fiscal years (FY)
      - Intent flags
    Uses fuzzy matching against schema_cache valid values.
    """
    q_norm = q.strip().lower()

    # FY extraction
    fys = sorted(set(re.findall(r"fy\s?(\d{2})", q_norm)))
    fy_old, fy_new = (None, None)
    if len(fys) >= 2:
        fy_old, fy_new = f"FY{fys[0]}", f"FY{fys[1]}"
    elif len(fys) == 1:
        fy_new = f"FY{fys[0]}"

    detected: Dict[str, Dict[str, Optional[Any]]] = {
        "org": {"value": None, "dim": None},
        "service": {"value": None, "dim": None},
        "country": {"value": None, "dim": None},
        "project": {"value": None, "dim": None},
    }

    # ORG_xxx
    m = re.search(r"(org[_-]?\d{1,4})", q_norm, re.IGNORECASE)
    if m:
        detected["org"]["value"] = m.group(1).upper()
        detected["org"]["dim"] = "DimOrg.orgId"

    # Token / n-grams
    tokens = re.findall(r"[a-zA-Z0-9_+]+", q_norm)
    ngrams = tokens + [" ".join(p) for p in zip(tokens, tokens[1:])]  # bigrams
    ngrams += [" ".join(p) for p in zip(tokens, tokens[1:], tokens[2:])]  # trigrams
    ngrams = [t.lower() for t in ngrams]

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
                for lv in low_vals:
                    if tok_l in lv:
                        match = lv
                        break
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
                        detected["service"]["value"], detected["service"]["dim"] = v, dim
                    elif dim.startswith("DimCountry."):
                        detected["country"]["value"], detected["country"]["dim"] = v, dim
                    elif dim.startswith("DimProject."):
                        detected["project"]["value"], detected["project"]["dim"] = v, dim

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
    """Heuristisch in Teilanfragen splitten."""
    subs = []
    q_lower = q.lower()

    if "driver" in q_lower or "cost driver" in q_lower:
        subs.append({"type": "drivers", "org": parsed.get("org"), "fy_old": parsed.get("fy_old"), "fy_new": parsed.get("fy_new")})

    if ("why" in q_lower or "reason" in q_lower) and parsed.get("service"):
        subs.append({"type": "why_service", "org": parsed.get("org"), "service": parsed.get("service"), "service_dim": parsed.get("service_dimension")})

    if parsed.get("country"):
        subs.append({"type": "totals", "country": parsed.get("country"), "country_dim": parsed.get("country_dimension")})

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
        f"scope change {fy_new} {org}",
    ]
    return base

def _json_safe(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

def _extract_fy_from_filters(q: dict) -> str | None:
    for f in q.get("filters", []) or []:
        dim = f.get("dimension") or f.get("member")
        if (dim or "").endswith("FctItCosts.fiscalYear") or (dim or "") == "FctItCosts.fiscalYear":
            vals = f.get("values") or []
            if isinstance(vals, list) and vals:
                return str(vals[0]).strip().upper()
    return None

def _classify_llm_query(q: dict) -> str:
    dims = set(q.get("dimensions", []) or [])
    # Country totals?
    if "DimCountry.countryName" in dims:
        return "totals"
    # App/Service detail (why_service)?
    if "DimApp.appName" in dims or "DimService.serviceName" in dims:
        # Wenn nur eine App/Service gefiltert wird → why_service, sonst drivers
        names = {"DimApp.appName", "DimService.serviceName"}
        filtered = [f for f in (q.get("filters") or []) if (f.get("dimension") or f.get("member")) in names]
        if filtered:
            # Eine konkrete App/Service im Filter → warum teuer?
            return "why_service"
        return "drivers"
    return "drivers"

# ---------------------------------------------------------------------------
#  LLM-FORMAT
# ---------------------------------------------------------------------------

def format_for_llm(result: dict) -> dict:
    """
    Vereinheitlicht numerische Ergebnisse (aus deterministischem oder LLM-Flow)
    in ein LLM-freundliches Format mit Top-5-Deltas, App-Kosten, Ländersummen
    und Kontexttexten. Robuster gegen fehlende Spalten oder inkonsistente Benennungen.
    """
    import pandas as pd

    parsed = result.get("parsed", {})
    subs = result.get("subquery_results", []) or result.get("cube_results", [])

    out = {
        "summary": {
            "org": parsed.get("org"),
            "fiscal_years": [parsed.get("fy_old"), parsed.get("fy_new")],
            "country": parsed.get("country"),
        },
        "drivers": [],
        "app_costs": [],
        "country_totals": {},
        "context_docs": [],
    }

    for sub in subs:
        t = sub.get("type")

        # ------------------------------------------------------------------
        # DRIVERS
        # ------------------------------------------------------------------
        if t == "drivers" and sub.get("data"):
            df = pd.DataFrame(sub["data"])

            # Erkenne Spaltennamen flexibel
            name_col = None
            for cand in ["DimService.serviceName", "DimApp.appName", "service", "app"]:
                if cand in df.columns:
                    name_col = cand
                    break
            if not name_col:
                dim_guess = [c for c in df.columns if c.startswith("Dim") and c.endswith((".serviceName", ".appName"))]
                name_col = dim_guess[0] if dim_guess else None
            if not name_col or "FctItCosts.actualCost" not in df.columns:
                continue

            # FY-Spalte ggf. rekonstruieren
            if "FctItCosts.fiscalYear" not in df.columns:
                df["FctItCosts.fiscalYear"] = "FY?"

            df = df.rename(columns={
                name_col: "service",
                "FctItCosts.fiscalYear": "fy",
                "FctItCosts.actualCost": "cost",
            })
            df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
            df_grouped = df.groupby(["service", "fy"], as_index=False)["cost"].sum()

            fy_cols = df_grouped["fy"].unique().tolist()
            top_24 = (
                df_grouped[df_grouped["fy"] == "FY24"]
                .sort_values("cost", ascending=False)
                .head(5)["service"].tolist()
                if "FY24" in fy_cols else []
            )
            top_25 = (
                df_grouped[df_grouped["fy"] == "FY25"]
                .sort_values("cost", ascending=False)
                .head(5)["service"].tolist()
                if "FY25" in fy_cols else []
            )
            top_union = sorted(set(top_24 + top_25)) or df_grouped.sort_values("cost", ascending=False).head(5)["service"].tolist()

            df_filtered = df_grouped[df_grouped["service"].isin(top_union)]
            df_pivot = df_filtered.pivot(index="service", columns="fy", values="cost").reset_index()

            if "FY24" in df_pivot.columns and "FY25" in df_pivot.columns:
                df_pivot["delta_abs"] = df_pivot["FY25"] - df_pivot["FY24"]
                df_pivot["delta_pct"] = df_pivot["delta_abs"] / df_pivot["FY24"].replace(0, pd.NA)

            out["drivers"] = []
            for _, row in df_pivot.iterrows():
                rec = {
                    "service": row["service"],
                    "FY24": float(row.get("FY24")) if "FY24" in df_pivot.columns else None,
                    "FY25": float(row.get("FY25")) if "FY25" in df_pivot.columns else None,
                    "delta_abs": float(row["delta_abs"]) if "delta_abs" in df_pivot.columns else None,
                    "delta_pct": f"{round(row['delta_pct'] * 100, 1)}%" if not pd.isna(row.get("delta_pct")) else None,
                }
                for k in ["FY24", "FY25", "delta_abs"]:
                    if rec.get(k) is not None:
                        rec[k] = round(rec[k], 2)
                out["drivers"].append(rec)

        # ------------------------------------------------------------------
        # WHY SERVICE
        # ------------------------------------------------------------------
        elif t == "why_service" and sub.get("numeric"):
            df = pd.DataFrame(sub["numeric"])
            if df.empty:
                continue

            df = df.rename(columns={
                "DimApp.appName": "app",
                "DimService.serviceName": "app",
                "FctItCosts.fiscalYear": "fy",
                "FctItCosts.actualCost": "cost",
                "FctItCosts.price": "price",
                "FctItCosts.quantity": "quantity",
            })

            for c in ["cost", "price", "quantity"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # --- sicherstellen, dass Kombinationen eindeutig sind ---
            df_grouped = (
                df.groupby(["app", "fy"], as_index=False)["cost"]
                .sum()
            )

            pivot = df_grouped.pivot(index="app", columns="fy", values="cost").reset_index()

            # Falls beide FYs existieren, Deltas berechnen
            if "FY24" in pivot.columns and "FY25" in pivot.columns:
                pivot["delta_abs"] = pivot["FY25"] - pivot["FY24"]
                pivot["delta_pct"] = pivot["delta_abs"] / pivot["FY24"].replace(0, pd.NA)

            out["app_costs"] = []
            for _, row in pivot.iterrows():
                rec = {
                    "app": row["app"],
                    "FY24_cost": float(row.get("FY24")) if "FY24" in pivot.columns else None,
                    "FY25_cost": float(row.get("FY25")) if "FY25" in pivot.columns else None,
                    "delta_abs": float(row.get("delta_abs")) if "delta_abs" in pivot.columns else None,
                    "delta_pct": f"{round(row['delta_pct'] * 100, 1)}%" if not pd.isna(row.get("delta_pct")) else None,
                }
                for k in ["FY24_cost", "FY25_cost", "delta_abs"]:
                    if rec.get(k) is not None:
                        rec[k] = round(rec[k], 2)
                out["app_costs"].append(rec)

            if sub.get("textual"):
                out["context_docs"].extend([d["text"] for d in sub["textual"]])

        # ------------------------------------------------------------------
        # COUNTRY TOTALS
        # ------------------------------------------------------------------
        elif t == "totals" and sub.get("data"):
            for row in sub["data"]:
                name = row.get("DimCountry.countryName") or row.get("country")
                if name:
                    val = float(row.get("FctItCosts.actualCost", 0))
                    out["country_totals"][name] = round(val, 2)

    return out



# ---------------------------------------------------------------------------
#  MAIN ORCHESTRATION LOGIC
# ---------------------------------------------------------------------------

def orchestrate(question: str):
    print(f"[Orchestrator] Starting orchestration for query: {question}")
    timings = {}
    t_start = time.time()

    # --- Startphase: Parsing & Setup zählen zum Overhead ---
    parsed = parse_question(question, VALID_VALUES)
    subqueries = split_into_subqueries(question, parsed)

    if useLLM and GeminiClient is not None:
        print(f"[Orchestrator] Running in simplified LLM mode with Chroma context ({llm_mode})")
        from copy import deepcopy

        schema = build_llm_schema()
        llm = GeminiClient(model="gemini-2.5-flash")
        cube = CubeClient()
        chroma = ChromaClient()
        LLM_DEBUG = os.getenv("LLM_DEBUG", "false").lower() == "true"

        # -------- 1) Query-Generierung durch LLM ----------
        t_llm_start = time.time()
        timings["t_before_llm_api"]= t_llm_start - t_start
        queries = []
        try:
            if llm_mode in ("generate_queries", "full"):
                queries, t_info = llm.generate_queries(question, schema)
                timings.update({
                    "t_llm_api_1": t_info.get("t_llm_api_1", 0),
                    "t_llm_parse_queries": t_info.get("t_llm_parse_queries", 0)
                })
                print(f"[Orchestrator] LLM generated {len(queries)} Cube.js queries.")
                if LLM_DEBUG:
                    os.makedirs(RESULTS_DIR, exist_ok=True)
                    debug_q_path = os.path.join(RESULTS_DIR, "debug_llm_queries.json")
                    with open(debug_q_path, "w", encoding="utf-8") as f:
                        json.dump(queries, f, indent=2)
                    print(f"[DEBUG] Saved raw LLM queries to {debug_q_path}")
        except Exception as e:
            print(f"[Orchestrator] LLM query generation failed: {e}")

        # -------- 2) Cube.js-Ausführung ----------
        t_cube_start = time.time()
        raw_results, query_times = [], []
        for i, q in enumerate(queries, 1):
            try:
                t_qs = time.time()
                print(f"[Orchestrator] Executing Cube.js query #{i}: {q}")
                for f in q.get("filters", []) or []:
                    if "member" in f and "dimension" not in f:
                        f["dimension"] = f.pop("member")
                df = cube.query(deepcopy(q))
                raw_results.append({
                    "query_index": i,
                    "query": q,
                    "rows": df.to_dict(orient="records"),
                    "columns": df.columns.tolist(),
                })
                query_times.append(time.time() - t_qs)
            except Exception as e:
                print(f"[Orchestrator] Failed query #{i}: {e}")
                query_times.append(None)
                raw_results.append({"query_index": i, "query": q, "error": str(e)})
        timings["t_cube_exec"] = time.time() - t_cube_start
        timings["cube_query_times"] = query_times

        # -------- 3) Kontext (Chroma) ----------
        t_chroma_start = time.time()
        chroma_docs = []
        try:
            parsed_llm = parse_question(question, VALID_VALUES)
            fy_old = parsed_llm.get("fy_old") or parsed.get("fy_old") or "FY24"
            fy_new = parsed_llm.get("fy_new") or parsed.get("fy_new") or "FY25"
            service = parsed_llm.get("service") or parsed.get("service")

            if not service:
                for q in queries:
                    for f in q.get("filters", []):
                        val = (f.get("values") or [None])[0]
                        if f.get("member") in ["DimApp.appName", "DimService.serviceName"] and val:
                            service = val
                            break
                    if service:
                        break

            if service:
                query_text = f"Reasons for {service} cost change {fy_old} vs {fy_new}"
                chroma_docs = chroma.query_expanded_for_service(
                    question=query_text,
                    app_name=service,
                    fiscal_years=[fy_old, fy_new],
                    section_filter=["pricing", "changes", "sla"],
                    top_k_init=5,
                    max_snippets=6,
                    max_chars_per_snippet=320,
                )
            else:
                chroma_docs = chroma.query(question=f"General IT cost overview for {fy_new}", top_k=5)
        except Exception as e:
            print(f"[Orchestrator] ChromaDB retrieval failed: {e}")
            chroma_docs = []
        t_chroma_end = time.time()

        # -------- 4) Antwort-Generierung (2. LLM) ----------
        t_ans_start = time.time()
        answer = None
        if llm_mode in ("interpret_results", "full") and raw_results:
            try:
                prompt = (
                    "You are an IT cost analyst.\n"
                    f"Question:\n{question}\n\n"
                    "Below are raw Cube.js query results and retrieved contextual documents.\n"
                    "Analyze them together to identify top cost drivers, year-over-year changes, "
                    "and root causes for cost differences.\n\n"
                    f"RAW RESULTS:\n{json.dumps(raw_results, indent=2)}\n\n"
                    f"CONTEXT DOCS:\n{json.dumps(chroma_docs, indent=2)}\n\n"
                    "Provide a concise, factual summary."
                )
                answer, t_info = llm.generate_answer(prompt=prompt)
                timings.update({
                    "t_llm_api_2": t_info.get("t_llm_api_2", 0),
                    "t_llm_postprocess_answer": t_info.get("t_llm_postprocess_answer", 0)
                })
                ts = int(time.time())
                answer_path = os.path.join(RESULTS_DIR, f"llm_answer_{ts}.txt")
                with open(answer_path, "w", encoding="utf-8") as f:
                    f.write(answer or "No content returned.")
                print(f"[Orchestrator] LLM answer written to {answer_path}")
            except Exception as e:
                print(f"[Orchestrator] LLM interpretation failed: {e}")

        # -------- 5) Overhead & Gesamtzeit ----------
        t_end = time.time()

        # Alle relevanten LLM- und Systemzeiten summieren
        llm_total = (
                timings.get("t_llm_api_1", 0)
                + timings.get("t_llm_parse_queries", 0)
                + timings.get("t_llm_api_2", 0)
                + timings.get("t_llm_postprocess_answer", 0)
        )

        # interner Overhead = alles, was nicht LLM oder Cube ist (z. B. Parsing, Chroma, Logging)
        sum_known = (
                timings.get("t_before_llm_api", 0)
                + timings.get("t_llm_api_1", 0)
                + timings.get("t_llm_parse_queries", 0)
                + timings.get("t_cube_exec", 0)
                + timings.get("t_llm_api_2", 0)
                + timings.get("t_llm_postprocess_answer", 0)
        )
        timings["t_internal_overhead"] = max(0, t_end - t_start - sum_known)

        # Gesamtzeit für orchestrate()
        timings["total"] = t_end - t_start

        # Bundle aus allen Ergebnissen + Timings
        raw_bundle = {
            "question": question,
            "llm_queries": queries,
            "cube_results_raw": raw_results,
            "context_docs": chroma_docs,
            "answer": answer,
            "timing": timings,
        }

        # Speichern der Ergebnisse
        ts = int(time.time())
        raw_path = os.path.join(RESULTS_DIR, f"raw_llm_cube_results_{ts}.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_bundle, f, indent=2)
        print(f"[Orchestrator] Saved raw Cube.js + Chroma results to {raw_path}")

        return raw_bundle, raw_path

    # ===== Deterministischer Modus (alter Flow) =====
    print("[Orchestrator] Running in deterministic mode (useLLM=False)")
    cube = CubeClient()
    chroma = ChromaClient()
    subquery_results: List[Dict[str, Any]] = []

    if subqueries:
        print(f"[Orchestrator] Detected {len(subqueries)} subqueries: {[s.get('type') for s in subqueries]}")

        for sub in subqueries:
            t = str(sub.get("type", "")).strip().lower()

            # --- 1) Drivers ---
            if t == "drivers":
                print(f"[Orchestrator] Executing driver query for {sub.get('org')} FY {sub.get('fy_old')} → {sub.get('fy_new')}")
                df = cube.top_cost_drivers(sub.get("org"), sub.get("fy_old"), sub.get("fy_new"), top_n=5)
                _round_dataframe(df, digits=2)
                totals = getattr(df, "attrs", {}).get("total_costs", {})
                totals = {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in totals.items()}
                subquery_results.append({"type": t, "data": df.to_dict(orient="records"), "totals": totals})
                continue

            # --- 2) Why Service Costs High ---
            if t == "why_service":
                print(f"[Orchestrator] Executing why_service query for {sub.get('service')} ({sub.get('service_dim')})")
                df = cube.total_cost_by_service_fy(sub.get("org"), sub.get("service"), sub.get("service_dim"))
                _round_dataframe(df, digits=2)
                numeric_data = df.to_dict(orient="records")
                for r in numeric_data:
                    if "delta_pct" in r and isinstance(r["delta_pct"], (float, int)):
                        r["delta_pct"] = f"{round(r['delta_pct'] * 100, 1)}%"

                fy_new = normalize_fiscal_year(sub.get("fy_new"))
                fy_old = normalize_fiscal_year(sub.get("fy_old"))
                if not fy_new and fy_old:
                    fy_new = fy_old; fy_old = None
                if fy_new and not fy_old:
                    try:
                        y = int(fy_new.replace("FY", ""))
                        fy_old = f"FY{y-1:02d}"
                    except Exception:
                        fy_old = fy_new

                query_text = f"Reasons for {sub.get('service')} cost change {fy_old} vs {fy_new}"
                context_strings = chroma.query_expanded_for_service(
                    question=query_text,
                    app_name=sub.get("service"),
                    fiscal_years=[fy_old, fy_new],
                    section_filter=["pricing", "changes", "sla"],
                    top_k_init=5,
                    max_snippets=6,
                    max_chars_per_snippet=320
                )
                context_docs = [{"text": s} for s in context_strings]

                subquery_results.append({"type": t, "numeric": numeric_data, "textual": context_docs})
                continue

            # --- 3) Totals ---
            if t == "totals":
                print(f"[Orchestrator] Executing totals query for {sub.get('country')} ({sub.get('country_dim')})")
                df = cube.total_cost_by_country(sub.get("country"), sub.get("country_dim"))
                _round_dataframe(df, digits=2)
                subquery_results.append({"type": t, "data": df.to_dict(orient="records")})
                continue

        # Combine & persist
        result = {
            "question": question,
            "parsed": parsed,
            "subquery_results": subquery_results,
            "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"result_{int(time.time())}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[Orchestrator] Multi-query result written to {out_path}")

        llm_input = format_for_llm(result)
        llm_path = out_path.replace(".json", "_llm.json")
        with open(llm_path, "w", encoding="utf-8") as f:
            json.dump(llm_input, f, indent=2, default=_json_safe)
        print(f"[Orchestrator] LLM-formatted data written to {llm_path}")

        # Optional: Antwort generieren (nur wenn explizit gewünscht)
        if useLLM and GeminiClient is not None:
            try:
                llm_client = GeminiClient(model="gemini-2.5-flash")
                prompt = f"Given the following analytical results and context, answer this question: {question}"
                answer = llm_client.generate_answer(prompt=prompt, context=llm_input)
                answer_path = llm_path.replace("_llm.json", "_llm_answer.txt")
                with open(answer_path, "w", encoding="utf-8") as f:
                    f.write(answer or "No content returned.")
                print(f"[Orchestrator] LLM answer written to {answer_path}")
            except Exception as e:
                print(f"[Orchestrator] LLM generation failed: {e}")

        return result, out_path

    # Single-query fallback
    print("[Orchestrator] No explicit subqueries detected, running standard pipeline.")
    cube = CubeClient()
    chroma = ChromaClient()
    df_service = cube.total_cost_by_service_fy(parsed["org"], parsed["service"], parsed["service_dimension"])
    df_drivers = cube.top_cost_drivers(parsed["org"], parsed["fy_old"], parsed["fy_new"], top_n=parsed["intent"]["top_n"])
    df_delta = cube.cost_delta_summary(parsed["org"], parsed["fy_old"], parsed["fy_new"])
    _round_dataframe(df_service, digits=2)
    _round_dataframe(df_drivers, digits=2)
    _round_dataframe(df_delta, digits=2)

    queries = build_retrieval_queries(parsed)
    docs = ChromaClient().query(queries, top_k=5)

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
        json.dump(llm_input, f, indent=2, default=_json_safe)
    print(f"[Orchestrator] LLM-formatted data written to {llm_path}")

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
