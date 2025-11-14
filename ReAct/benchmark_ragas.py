"""
Benchmark 2 – Retrieval & Answer Faithfulness (RAGAS, Local Evaluation)

Verwendet lokale Modelle (Mistral, HuggingFaceEmbeddings)
statt externer OpenAI-APIs.
"""

import os
import json
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import csv

# --- Lokale Modelle ---
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

# --- interne Module ---
from ReAct.orchestrator import orchestrate

# -------------------------------------------------------------------
#  Setup
# -------------------------------------------------------------------
load_dotenv()
os.environ["USE_LLM"] = "true"
os.environ["LLM_MODE"] = "full"
os.environ["LLM_DEBUG"] = "true"
BASE_DIR = os.path.join("results", "RAGAS")
os.makedirs(BASE_DIR, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
DATASET_PATH = os.path.join(BASE_DIR, f"ragas_dataset_{ts}.json")
METRICS_PATH = os.path.join(BASE_DIR, f"ragas_metrics_{ts}.csv")
RESULT_PATH = os.path.join(BASE_DIR, f"ragas_summary_{ts}.txt")
QUERIES_PATH = os.path.join(BASE_DIR, f"ragas_queries_{ts}.txt")
ANSWERS_PATH = os.path.join(BASE_DIR, f"ragas_answers_{ts}.txt")
TIME_PATH = os.path.join(BASE_DIR, f"ragas_time_measurements_{ts}.csv")

# -------------------------------------------------------------------
#  Lokale LLM- und Embedding-Modelle
# -------------------------------------------------------------------
llm = OllamaLLM(model="mistral")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("[RAGAS] Using local LLM (Mistral) and local embeddings (MiniLM-L6-v2)")
# -------------------------------------------------------------------
#  Fragen & Goldantworten
# -------------------------------------------------------------------
QUESTIONS_AND_GOLDS = [
    ("What are the top 5 cost drivers for the Customer Service business unit in FY25?",
     "The top 5 cost drivers for the Customer Service business unit in FY are: ALM / Dev Collaboration 1,546,505.202; Enterprise Data Platform 1,393,600.595; HCM Platform 1,186,216.825; Intranet / Portal 1,129,849.136; ERP Procurement Platform 946,825.638"),
    ("Show total IT costs for the Manufacturing unit in FY24.", "Total cost of the Manufacturing unit in FY24 was 11,279,186.118"),
    ("Compare total IT spending for the Finance organization between FY24 and FY25.",
     "The Finance Organisation spent 16,748,080.32 in FY24 and 17,000,194.281 in FY25"),
    ("Which business unit had the highest total IT costs in FY25?", "The Procurement Business Unit had the highest cost with 17,163,702.347"),
    ("What are the main cost drivers for the HR organization in FY25?",
     "The main cost drivers for the HR organisations in FY25 are: Applications 11,268,715.455; Security 1,890,875.282; Workplace 1,735,542.496; Hosting / Compute 1,691,662.181; Network 247,194.479; Change 80,576.07"),
    ("How did total IT costs for the Marketing unit change from FY24 to FY25?",
     "The IT costs from the Marketing Unit increased from 13,549,852.133 in FY24 to 13,725,013.274 in FY25"),
    ("List the top 3 applications by cost for the Procurement unit in FY25.",
     "Top 3 applications for the Procurement business unit in FY25: Snowflake Data Cloud 870,691.957; Enterprise Data Lake 869,486.28;  Microsoft SharePoint Online 865,391.799"),
    ("What is the total IT spending for the R&D unit in FY25?",
     "The total IT spending for the Research and Development unit in FY25 was 7,472,680.947."),
    ("Show total IT costs per business unit for FY24.",
     "In FY24, the total IT costs per business unit were as follows: Procurement 16,926,415.855; Corporate 16,846,861.726; IT & Digital 16,756,609.937; Finance & Controlling 16,748,080.32; Human Resources 16,725,690.895; Sales & Marketing 13,549,852.133; Customer Services 13,492,265.474; Manufacturing Operations 11,279,186.118; Supply Chain & Logistics 9,894,030.293; and Research & Development 7,304,306.852."),
    ("Compare IT costs for Corporate and Finance in FY25.",
     "In FY25, the Corporate unit had IT costs of 17,034,694.035, while the Finance & Controlling unit spent 17,000,194.281."),
    ("What was the total cost of Microsoft 365 in FY25?",
     "The total cost of Microsoft 365 in FY25 was 4,707,291.688."),
    ("Compare Microsoft Teams costs between FY24 and FY25.",
     "The cost of Microsoft Teams increased from 4,085,368.576 in FY24 to 4,716,886.717 in FY25."),
    ("How did SAP ERP Platform costs change for Corporate from FY24 to FY25?",
     "For the Corporate business unit, ERP Procurement Platform costs decreased slightly from 1,187,874.006 in FY24 to 1,184,980.896 in FY25. The ERP Finance Platform costs remained constant with 493,789.168 in FY24 to 493,682.103 in FY25, while the ERP Sales Platform decreased from 577,149.456 in FY24 to 574,583.856 in FY25."),
    ("Show total spending on Jira Software in FY25.",
     "The total spending on Jira Software in FY25 was 5,333,048.258."),
    ("What are the 3 most expensive services for the United States in 2025?",
     "The three most expensive services in 2025 for the United States are ALM / Dev Collaboration 1,543,182.002, Intranet / Portal 1,391,457.172, Enterprise Data Platform 1,389,348.47"),
    ("Compare ServiceNow license costs FY24 vs FY25 for the IT unit.",
     "For the IT unit, ServiceNow license costs were 486,371.189 in FY24 and slightly decreased to 482,639.167 in FY25."),
    ("Which Microsoft App is the most expensive in FY25?",
     "The most expensive Microsoft application in FY25 was Microsoft SharePoint Online, with total costs of 7,196,287.571."),
    ("What were the total ServiceNow Platform costs for the Finance organization in FY25?",
     "The total ServiceNow Platform costs for the Finance organization in FY25 amounted to 484,973.625."),
    ("How much was spent on Snowflake in FY25 compared to FY24?",
     "Spending on Snowflake was 7,217,537.538 in FY24 and slightly decreased to 7,214,861.2 in FY25."),
    ("Show IT spending on the Oracle Database Platform for FY25.",
     "The total IT spending on the Oracle Database Platform in FY25 was 6,677,959.079."),
    ("Show total IT cost by country for FY25.",
     "In FY25, total IT costs by country were as follows: Germany 20,341,512.585; Italy 15,260,385.228; United States 14,942,322.751; Mexico 14,281,530.195; Brazil 14,273,245.879; Canada 12,285,466.909; France 11,764,300.349; Netherlands 11,763,073.657; Poland 11,115,188.911; Spain 10,963,862.034; China 2,870,746.887; and Argentina 1,428,923.186."),
    ("Which country had the highest IT spending in FY25?",
     "Germany had the highest IT spending in FY25, totaling 20,341,512.585."),
    ("Compare IT spending between Germany and United States in FY25.",
     "In FY25, Germany spent 20,341,512.585 on IT, while the United States spent 14,942,322.751."),
    ("What is the total cost for Europe region in FY25?",
     "The total IT cost for the Europe region in FY25 was 81,208,322.764."),
    ("How did total costs for the United States change between FY24 and FY25?",
     "Total IT costs for the United States increased from 14,765,239.781 in FY24 to 14,942,322.751 in FY25."),
]

# -------------------------------------------------------------------
#  Datensammler
# -------------------------------------------------------------------
query_log = []   # für ragas_queries_<ts>.txt
answer_log = []  # für ragas_answers_<ts>.txt
dataset = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

time_records = []

for i, (question, gold) in enumerate(QUESTIONS_AND_GOLDS, 1):
    print(f"\n[RAGAS] ({i}/{len(QUESTIONS_AND_GOLDS)}) Processing: {question}")
    try:
        # (1) Benchmark-Startzeitpunkt
        t_bench_start = time.time()

        # (2) Orchestrator einmal aufrufen und Laufzeit messen
        t_orch_start = time.time()
        result, _ = orchestrate(question)  # LLM-Modus muss in orchestrator.py aktiv sein


        # (3) Artefakte aus dem Orchestrator-Ergebnis
        cube_results = result.get("cube_results_raw") or result.get("subquery_results") or []
        chroma_docs  = result.get("context_docs") or []
        llm_queries  = result.get("llm_queries") or []
        timings      = result.get("timing", {})
        t_llm_gen_queries = timings.get("t_llm_api_1", 0)
        t_llm_interpretation = timings.get("t_llm_api_2", 0)
        # Overhead-Blöcke dem Orchestrator zuschlagen
        t_orchestrator = (
                timings.get("t_before_llm_api", 0)  # das fügen wir unten hinzu
                + timings.get("t_llm_parse_queries", 0)
                + timings.get("t_llm_postprocess_answer", 0)
                + timings.get("t_internal_overhead", 0)
        )

        t_cube_exec = timings.get("t_cube_exec", 0)
        n_queries    = len(llm_queries)

        # (4) Zeitpunkte zusammensetzen
        record = {
            "question": question,
            "n_llm_queries": len(result.get("llm_queries") or []),
            "t_total_benchmark": time.time() - t_bench_start,
            "t_orchestrator": t_orchestrator,
            "t_llm_gen_queries": t_llm_gen_queries,
            "t_cube_exec": t_cube_exec,
            "t_llm_interpretation": t_llm_interpretation,
        }

        time_records.append(record)

        # (5) LLM-Antwort bestimmen (bevorzugt direkt aus result)
        system_answer = result.get("answer")
        if not system_answer:
            # Fallback: jüngste llm_answer_*.txt aus results/ (nur wenn vorhanden)
            try:
                candidates = [f for f in os.listdir("results") if f.startswith("llm_answer_")]
                candidates.sort(key=lambda f: os.path.getmtime(os.path.join("results", f)))
                if candidates:
                    with open(os.path.join("results", candidates[-1]), "r", encoding="utf-8") as f:
                        system_answer = f.read().strip()
            except Exception:
                pass
        if not system_answer:
            # letzter Fallback: Cube-Rohdaten als Text
            system_answer = json.dumps(cube_results, indent=2)

        # --------------------------------------------
        # LLM Answer ermitteln
        # --------------------------------------------
        answer_files = [f for f in os.listdir("results") if f.startswith("llm_answer_")]
        answer_files.sort(key=lambda f: os.path.getmtime(os.path.join("results", f)))
        system_answer = None
        if answer_files:
            answer_path = os.path.join("results", answer_files[-1])
            with open(answer_path, "r", encoding="utf-8") as f:
                system_answer = f.read().strip()
        if not system_answer:
            system_answer = json.dumps(cube_results, indent=2)

        # --------------------------------------------
        # Logs für kombinierte Datei aufbauen
        # --------------------------------------------
        section = []
        section.append(f"=== Q{i}: {question} ===\n")
        if llm_queries:
            section.append("--- Queries ---\n")
            section.append(json.dumps(llm_queries, indent=2))
        else:
            section.append("--- Queries ---\nNo queries generated.\n")

        section.append("\n--- LLM Answer ---\n")
        section.append(system_answer or "No LLM answer returned.")
        section.append("\n" + "=" * 80 + "\n")
        query_log.append("\n".join(section))

        # Separates Answer-Log (nur Textantworten + Kontexte)
        answer_log.append(f"=== Q{i}: {question} ===\n")
        answer_log.append(system_answer or "No answer")
        answer_log.append("\n--- Context Docs ---\n")
        answer_log.append(json.dumps(chroma_docs, indent=2))
        answer_log.append("\n" + "=" * 80 + "\n")

        # --------------------------------------------
        # Für RAGAS Dataset
        # --------------------------------------------
        dataset["question"].append(question)
        dataset["answer"].append(system_answer)
        dataset["contexts"].append(chroma_docs)
        dataset["ground_truth"].append(gold)

    except Exception as e:
        print(f"[RAGAS] Error for {question}: {e}")


# -------------------------------------------------------------------
#  Kombinierte Logs schreiben
# -------------------------------------------------------------------
with open(QUERIES_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(query_log))
with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(answer_log))
with open(DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)
with open(TIME_PATH, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "question", "n_llm_queries", "t_total_benchmark",
        "t_orchestrator", "t_llm_gen_queries",
        "t_cube_exec", "t_llm_interpretation"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(time_records)

print(f"[RAGAS] Timing measurements saved to {TIME_PATH}")

print(f"[RAGAS] Saved combined queries → {QUERIES_PATH}")
print(f"[RAGAS] Saved combined answers → {ANSWERS_PATH}")
print(f"[RAGAS] Saved dataset          → {DATASET_PATH}")

'''
# -------------------------------------------------------------------
#  Evaluate (lokales LLM + lokale Embeddings wie im Reevaluate)
# -------------------------------------------------------------------
try:
    # Datensatz in ein HF Dataset konvertieren
    ragas_ds = Dataset.from_dict(dataset)

    # LLM/Embeddings: identisch zu Ragas_reevaluate.py
    print("[RAGAS] Using local model (mistral:7b-instruct via Ollama) and local embeddings (MiniLM-L6-v2)")
    llm = OllamaLLM(model="mistral:7b-instruct")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("[RAGAS] Starting evaluation...")
    results = evaluate(
        dataset=ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embedding_model
    )

    # Ergebnisse speichern (DataFrame + Mean-Scores)
    df = results.to_pandas()
    df.to_csv(METRICS_PATH, index=False)

    mean_scores = {
        "faithfulness": df["faithfulness"].mean(),
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
        "context_recall": df["context_recall"].mean(),
    }

    summary = (
        f"\n=== RAGAS Results (Local, {ts}) ===\n"
        f"Faithfulness:       {mean_scores['faithfulness']:.3f}\n"
        f"Answer Relevancy:   {mean_scores['answer_relevancy']:.3f}\n"
        f"Context Precision:  {mean_scores['context_precision']:.3f}\n"
        f"Context Recall:     {mean_scores['context_recall']:.3f}\n"
    )

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    print(f"[RAGAS] Evaluation complete. Results saved to {RESULT_PATH}")

except Exception as e:
    print(f"[RAGAS] Evaluation failed: {e}")
'''
