"""
Benchmark 2 – Retrieval & Answer Faithfulness (RAGAS)

Ziel:
Bewertet die Qualität der Systemantworten (LLM + Cube.js + Chroma)
gegenüber bekannten Gold-Antworten.

Ausgabe:
results/ragas_dataset_<timestamp>.json
results/ragas_results_<timestamp>.txt
"""

import os, json, time
import pandas as pd
from datetime import datetime
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

from ReAct.llm_client import GeminiClient
from ReAct.cube_meta import build_llm_schema
from ReAct.cube_client import CubeClient
from ReAct.chroma_client import ChromaClient
from ReAct.orchestrator import orchestrate

# -------------------------------------------------------------------
#  Fragen & Goldantworten
# -------------------------------------------------------------------
QUESTIONS_AND_GOLDS = [
    ("What are the top 5 cost drivers for the Customer Service business unit in FY25?",
     "ALM / Dev Collaboration 3,085,106.281; Enterprise Data Platform 2,784,555.114; HCM Platform 2,374,951.258; Intranet / Portal 2,246,999.736; ERP Procurement Platform 1,901,550.486"),
    ("Show total IT costs for the Manufacturing unit in FY24.", "11,277,249.557"),
    ("Compare total IT spending for the Finance organization between FY24 and FY25.",
     "16,743,166.022 in FY24; 16,988,152.258 in FY25"),
    ("Which business unit had the highest total IT costs in FY25?", "Procurement 17,153,203.819"),
    ("What are the main cost drivers for the HR organization in FY25?",
     "Applications 11,267,217.572; Security 1,892,433.022; Workplace 1,736,848.138; Hosting / Compute 1,694,164.404; Network 246,047.018; Change 80,576.07"),
    ("How did total IT costs for the Marketing unit change from FY24 to FY25?",
     "13,569,878.654 in FY24; 13,735,394.861 in FY25"),
    ("List the top 3 applications by cost for the Procurement unit in FY25.",
     "Enterprise Data Lake 868,785.81; Snowflake Data Cloud 867,796.102; Microsoft SharePoint Online 865,466.959"),
    ("What is the total IT spending for the R&D unit in FY25?", "7,461,794.919"),
    ("Show total IT costs per business unit for FY24.",
     "Procurement 16,923,875.168; Corporate 16,820,535.362; IT & Digital 16,749,007.2; Finance & Controlling 16,743,166.022; Human Resources 16,735,670.955; Sales & Marketing 13,569,878.654; Customer Services 13,497,460.272; Manufacturing Operations 11,277,249.557; Supply Chain & Logistics 9,895,249.951; Research & Development 7,302,430.961"),
    ("Compare IT costs for Corporate and Finance in FY25.",
     "Corporate 16,820,535.362; Finance & Controlling 16,743,166.022"),
    ("What was the total cost of Microsoft 365 in FY25?", "4,707,522.35"),
    ("Compare Microsoft Teams costs between FY24 and FY25.",
     "4,086,481.562 in FY24; 4,722,327.246 in FY25"),
    ("How did SAP ERP Platform costs change for Corporate from FY24 to FY25?",
     "ERP Procurement Platform 10,054,281.996 in FY24; 10,058,850.12 in FY25; ERP Finance Platform 4,104,365.806 in FY24; 4,092,215.044 in FY25; ERP Sales Platform 4,781,650.656 in FY24; 4,782,354.336 in FY25"),
    ("Show total spending on Jira Software in FY25.", "5,343,965.935"),
    ("What are the top 3 applications with the highest cost increase in FY25?",
     "Microsoft Teams, Microsoft 365, Unassigned IT Application"),
    ("Compare ServiceNow license costs FY24 vs FY25 for the IT unit.",
     "483,532.296 in FY24; 482,144.203 in FY25"),
    ("Which Microsoft App is the most expensive in FY25?", "Microsoft SharePoint Online 7,181,517.111"),
    ("What were the total ServiceNow Platform costs for the Finance organization in FY25?",
     "485,055.757"),
    ("How much was spent on Snowflake in FY25 compared to FY24?",
     "7,212,778.831 in FY24; 7,210,060.066 in FY25"),
    ("Show IT spending on the Oracle Database Platform for FY25.", "6,689,306.159"),
    ("Show total IT cost by country for FY25.",
     "GERMANY 20,340,405.975; ITALY 15,251,033.548; UNITED STATES 14,955,479.474; MEXICO 14,273,084.623; BRAZIL 14,263,526.639; CANADA 12,286,952.05; FRANCE 11,773,428.135; NETHERLANDS 11,763,489.327; POLAND 11,116,512.507; SPAIN 10,963,736.417; CHINA 2,866,668.09; ARGENTINA 1,433,092.946"),
    ("Which country had the highest IT spending in FY25?", "GERMANY 20,340,405.975"),
    ("Compare IT spending between Germany and United States in FY25.",
     "GERMANY 20,340,405.975; UNITED STATES 14,955,479.474"),
    ("What is the total cost for Europe region in FY25?", "81,208,605.908"),
    ("How did total costs for the United States change between FY24 and FY25?",
     "14,749,533.314 in FY24; 14,955,479.474 in FY25"),
]

# -------------------------------------------------------------------
#  Setup
# -------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
DATASET_PATH = f"results/ragas_dataset_{ts}.json"
RESULT_PATH = f"results/ragas_results_{ts}.txt"

# -------------------------------------------------------------------
#  Collect system responses
# -------------------------------------------------------------------
dataset = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for i, (question, gold) in enumerate(QUESTIONS_AND_GOLDS, 1):
    print(f"\n[RAGAS] ({i}/{len(QUESTIONS_AND_GOLDS)}) Processing: {question}")
    try:
        result, _ = orchestrate(question)
        cube_results = result.get("cube_results_raw") or result.get("subquery_results") or []
        chroma_docs = result.get("context_docs") or []
        answer_path = None

        # suche LLM-Textantwort, falls generiert
        answer_files = [f for f in os.listdir("results") if f.startswith("llm_answer_")]
        answer_files.sort(key=lambda f: os.path.getmtime(os.path.join("results", f)))
        if answer_files:
            answer_path = os.path.join("results", answer_files[-1])
        system_answer = None
        if answer_path and os.path.exists(answer_path):
            with open(answer_path, "r", encoding="utf-8") as f:
                system_answer = f.read().strip()

        # Fallback, falls kein Answer-File vorhanden
        if not system_answer:
            system_answer = json.dumps(cube_results, indent=2)

        dataset["question"].append(question)
        dataset["answer"].append(system_answer)
        dataset["contexts"].append(chroma_docs)
        dataset["ground_truth"].append(gold)

    except Exception as e:
        print(f"[RAGAS] Error for {question}: {e}")

# -------------------------------------------------------------------
#  Save dataset for later inspection
# -------------------------------------------------------------------
with open(DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)
print(f"[RAGAS] Dataset saved to {DATASET_PATH}")

# -------------------------------------------------------------------
#  Evaluate with RAGAS
# -------------------------------------------------------------------
try:
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    df = pd.DataFrame([results])
    df.to_csv(f"results/ragas_metrics_{ts}.csv", index=False)

    summary = (
        f"\n=== RAGAS Results ({ts}) ===\n"
        f"Faithfulness:       {results['faithfulness']:.3f}\n"
        f"Answer Relevancy:   {results['answer_relevancy']:.3f}\n"
        f"Context Precision:  {results['context_precision']:.3f}\n"
        f"Context Recall:     {results['context_recall']:.3f}\n"
    )

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    print(f"[RAGAS] Evaluation complete. Metrics written to {RESULT_PATH}")

except Exception as e:
    print(f"[RAGAS] Evaluation failed: {e}")
