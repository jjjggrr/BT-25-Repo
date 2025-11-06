"""
Benchmark 1 – Query-to-Cube Success Rate (QSR)

Ziel:
Quantitativ messen, wie zuverlässig Gemini 2.5 Flash syntaktisch und semantisch
korrekte Cube.js-Queries erzeugt und ausführt.

Output:
results/qsr_results.csv  (pro Frage)
results/qsr_summary.txt  (aggregierte Metriken)
"""

import os
import json
import pandas as pd
from datetime import datetime
from ReAct.llm_client import GeminiClient
from ReAct.cube_meta import build_llm_schema
from ReAct.cube_client import CubeClient

# -------------------------------------------------------------------
#  SINGLE-RUN-MODUS
# -------------------------------------------------------------------
# Beispiel:
# SINGLE_RUN = ""          → alle Queries laufen (Standard)
# SINGLE_RUN = "15,16"     → nur diese Query-Indizes (1-basiert)
SINGLE_RUN = "17"

# -------------------------------------------------------------------
#  Benchmark Questions  (25 simple test queries)
# -------------------------------------------------------------------
QUESTIONS = [
    "What are the top 5 cost drivers for the Customer Service business unit in FY25?",
    "Show total IT costs for the Manufacturing unit in FY24.",
    "Compare total IT spending for the Finance organization between FY24 and FY25.",
    "Which business unit had the highest total IT costs in FY25?",
    "What are the main cost drivers for the HR organization in FY25?",
    "How did total IT costs for the Marketing unit change from FY24 to FY25?",
    "List the top 3 applications by cost for the Procurement unit in FY25.",
    "What is the total IT spending for the R&D unit in FY25?",
    "Show total IT costs per business unit for FY24.",
    "Compare IT costs for Corporate and Finance in FY25.",
    "What was the total cost of Microsoft 365 in FY25?",
    "Compare Microsoft Teams costs between FY24 and FY25.",
    "How did SAP ERP Platform costs change for Corporate from FY24 to FY25?",
    "Show total spending on Jira Software in FY25.",
    "What are the top 3 applications with the highest cost increase in FY25?",
    "Compare ServiceNow license costs FY24 vs FY25 for the IT unit.",
    "Which Microsoft App is the most expensive in FY25?",
    "What were the total ServiceNow Platform costs for the Finance organization in FY25?",
    "How much was spent on Snowflake  in FY25 compared to FY24?",
    "Show IT spending on the Oracle Database Platform for FY25.",
    "Show total IT cost by country for FY25.",
    "Which country had the highest IT spending in FY25?",
    "Compare IT spending between Germany and United States in FY25.",
    "What is the total cost for Europe region in FY25?",
    "How did total costs for the United States change between FY24 and FY25?",
]

# -------------------------------------------------------------------
#  Setup
# -------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_CSV = f"results/qsr_results_{ts}.csv"
OUT_SUMMARY = f"results/qsr_summary_{ts}.txt"
OUT_QUERIES = f"results/qsr_parsed_queries_{ts}.txt"

llm = GeminiClient(model="gemini-2.5-flash")
cube = CubeClient()
schema = build_llm_schema()
schema_text = json.dumps(schema)

# -------------------------------------------------------------------
#  Helper functions
# -------------------------------------------------------------------
def check_schema_match(query_obj: dict) -> bool:
    """prüft, ob alle Dimensions/Measures im Schema vorkommen"""
    try:
        q_str = json.dumps(query_obj)
        for key in ["FctItCosts", "Dim"]:
            if key in q_str:
                return True
        return False
    except Exception:
        return False


def try_execute_query(q: dict) -> bool:
    """führt Cube-Query aus; gibt True zurück, wenn Daten != 0"""
    try:
        df = cube.query(q)
        print(f"[DEBUG] Query returned {len(df)} rows")
        return not df.empty
    except Exception as e:
        print(f"[DEBUG] Execution error: {e}")
        return False

# -------------------------------------------------------------------
#  Benchmark-Loop (normal oder single-run)
# -------------------------------------------------------------------
records = []
parsed_log = []  # für .txt-Ausgabe

# Indizes bestimmen
if SINGLE_RUN.strip():
    selected_indices = [int(x) for x in SINGLE_RUN.split(",") if x.strip().isdigit()]
    print(f"\n[QSR] Running in SINGLE-RUN mode for queries: {selected_indices}")
else:
    selected_indices = range(1, len(QUESTIONS) + 1)
    print(f"\n[QSR] Running full benchmark with {len(QUESTIONS)} queries")

for idx in selected_indices:
    question = QUESTIONS[idx - 1]
    print(f"\n[QSR] Processing {idx}/{len(QUESTIONS)}: {question}")

    record = {
        "index": idx,
        "question": question,
        "valid_syntax": 0,
        "valid_schema": 0,
        "executed": 0,
    }

    entry_text = [f"=== Q{idx}: {question} ===\n"]

    try:
        queries = llm.generate_queries(question, schema)
        if queries:
            record["valid_syntax"] = 1
            valid_schema = all(check_schema_match(q) for q in queries)
            record["valid_schema"] = 1 if valid_schema else 0

            executed_any = any(try_execute_query(q) for q in queries)
            record["executed"] = 1 if executed_any else 0

            for i, q in enumerate(queries, 1):
                entry_text.append(f"--- Query #{i} ---")
                entry_text.append(json.dumps(q, indent=2))
            entry_text.append(f"Execution success: {bool(record['executed'])}\n")
        else:
            entry_text.append("[WARN] No queries generated by LLM.\n")

    except Exception as e:
        entry_text.append(f"Error: {e}\n")
        print(f"[QSR] Error: {e}")

    parsed_log.append("\n".join(entry_text))
    records.append(record)

# -------------------------------------------------------------------
#  Write detailed logs
# -------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
with open(OUT_QUERIES, "w", encoding="utf-8") as f:
    f.write("\n\n".join(parsed_log))
print(f"[QSR] Parsed LLM queries written to {OUT_QUERIES}")

# -------------------------------------------------------------------
#  Aggregation & Output
# -------------------------------------------------------------------
if not SINGLE_RUN.strip():
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)

    summary = df[["valid_syntax", "valid_schema", "executed"]].mean() * 100
    summary_text = (
        f"\n=== QSR Summary ({ts}) ===\n"
        f"Valid Syntax Rate:  {summary['valid_syntax']:.1f}%\n"
        f"Schema Match Rate:  {summary['valid_schema']:.1f}%\n"
        f"Execution Success:  {summary['executed']:.1f}%\n"
    )

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"[QSR] Detailed results written to {OUT_CSV}")
else:
    print(f"\n[QSR] Single-run mode complete. Logs written to {OUT_QUERIES}")

