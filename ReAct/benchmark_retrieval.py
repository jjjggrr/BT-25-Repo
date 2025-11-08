"""
Benchmark 3 – Hybrid Retrieval Ablation (BEIR-Style)
----------------------------------------------------

Vergleicht die Qualität von BM25, Dense und Hybrid (RRF) Retrieval
auf deinem bestehenden Chroma-Vektorkorpus (service_agreements + project_briefs).

Ergebnis:
Recall@5, MRR@10, AvgRank für jede Methode.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import time

from chromadb.utils import embedding_functions
from langchain_huggingface import HuggingFaceEmbeddings
from ReAct.chroma_client import ChromaClient


# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
BASE_DIR = os.path.join("results", "Retrieval")
os.makedirs(BASE_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_TIMING = os.path.join(BASE_DIR, f"retrieval_timings_{ts}.csv")


OUT_CSV = os.path.join(BASE_DIR, f"retrieval_results_{ts}.csv")
OUT_SUMMARY = os.path.join(BASE_DIR, f"retrieval_summary_{ts}.txt")

# -------------------------------------------------------------------
# Chroma-Client initialisieren
# -------------------------------------------------------------------
chroma = ChromaClient()

print("\n[Retrieval] Verifying available documents in ChromaDB...")
chroma.list_docs("service_agreements", limit=5)
chroma.list_docs("project_briefs", limit=5)
print("[Retrieval] Verification complete.\n")

def get_all_docs(collection):
    items = collection.get()
    docs = []
    for i in range(len(items["ids"])):
        docs.append({
            "title": items["metadatas"][i].get("file_name", f"doc_{i}"),
            "text": items["documents"][i],
        })
    return docs

docs = get_all_docs(chroma.sa_col) + get_all_docs(chroma.prj_col)
titles = [d["title"] for d in docs]
texts = [d["text"] for d in docs]
tokenized = [t.split() for t in texts]

print(f"[Retrieval] Loaded {len(docs)} documents from ChromaDB.")

# -------------------------------------------------------------------
# Embedding & BM25 Setup
# -------------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
ef = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
bm25 = BM25Okapi(tokenized)

# -------------------------------------------------------------------
# 50 Fragen + Gold-Dokument-Zuordnung
# -------------------------------------------------------------------
QUESTIONS_AND_GOLDS = [
    # --- Service Agreements: Kosten & Änderungen ---
    ("Why did Microsoft 365 costs rise in FY25?", "SLA_Microsoft_365_FY25.pdf"),
    ("What are the main pricing changes for ServiceNow Platform in FY25?", "SLA_ServiceNow_Platform_FY25.pdf"),
    ("How did Snowflake Data Cloud costs evolve from FY24 to FY25?", "SLA_Snowflake_Data_Cloud_FY25.pdf"),
    ("What explains the cost reduction in Oracle Database Platform for FY25?", "SLA_Oracle_Database_Platform_FY25.pdf"),
    ("What are the SLA updates for SAP S4HANA Finance in FY25?", "SLA_SAP_S4HANA_Finance_FY25.pdf"),
    ("Which cost driver affected Jira Software expenses in FY25?", "SLA_Jira_Software_FY25.pdf"),
    ("Were there any major vendor price changes for Google Workspace in FY25?", "SLA_Google_Workspace_FY25.pdf"),
    ("What are the FY25 service-level metrics for Enterprise Data Platform?", "SLA_Enterprise_Data_Platform_FY25.pdf"),
    ("How did HCM Platform costs change between FY24 and FY25?", "SLA_HCM_Platform_FY25.pdf"),
    ("Which applications saw the largest percentage increase in total cost in FY25?", "SLA_Applications_Summary_FY25.pdf"),

    # --- Projects: Initiatives & Benefits (alle 10 existierenden Projekte) ---
    ("What were the objectives of the Data Center Refresh project?", "PRJ_Data_Center_Refresh_FY25.pdf"),
    ("What benefits were realized from the Modernize Endpoint project?", "PRJ_Modernize_Endpoint_FY25.pdf"),
    ("Which departments were involved in the New Collaboration Suite project?", "PRJ_New_Collaboration_Suite_FY25.pdf"),
    ("What were the main deliverables of the ERP Upgrade project?", "PRJ_ERP_Upgrade_FY25.pdf"),
    ("What were the security improvements in the Security Enhancement Program?", "PRJ_Security_Enhancement_Program_FY25.pdf"),
    ("How did the Cloud Migration project improve infrastructure scalability?", "PRJ_Cloud_Migration_FY25.pdf"),
    ("What automation results were achieved by the AI Automation Program?", "PRJ_AI_Automation_Program_FY25.pdf"),
    ("What were the cost savings from the IT Budget Optimization project?", "PRJ_IT_Budget_Optimization_FY25.pdf"),
    ("Which network segments were upgraded in the Network Transformation project?", "PRJ_Network_Transformation_FY25.pdf"),
    ("Which business risks were mitigated through the Risk Reduction Initiative?", "PRJ_Risk_Reduction_Initiative_FY25.pdf"),

    # --- Comparative / Cross-Service ---
    ("Compare the total FY25 costs of Microsoft 365 and Google Workspace.", "SLA_Microsoft_365_FY25.pdf"),
    ("Which had higher FY25 license fees: SAP S4HANA Finance or Oracle Database Platform?", "SLA_SAP_S4HANA_Finance_FY25.pdf"),
    ("How do ServiceNow and Jira Software differ in their FY25 SLA targets?", "SLA_ServiceNow_Platform_FY25.pdf"),
    ("Compare FY24 vs FY25 uptime commitments for Microsoft 365.", "SLA_Microsoft_365_FY25.pdf"),
    ("What is the difference in FY25 per-user pricing between Snowflake and Enterprise Data Platform?", "SLA_Snowflake_Data_Cloud_FY25.pdf"),
    ("Which service achieved the best cost-to-performance ratio in FY25?", "SLA_Enterprise_Data_Platform_FY25.pdf"),
    ("How did hosting and compute costs change for ERP Finance Platform in FY25?", "SLA_ERP_Finance_Platform_FY25.pdf"),
    ("Compare FY25 support-hours commitments for SAP and Oracle services.", "SLA_SAP_S4HANA_Finance_FY25.pdf"),
    ("Which FY25 SLA includes 24/7 support coverage?", "SLA_ServiceNow_Platform_FY25.pdf"),
    ("Which vendor introduced sustainability clauses in their FY25 agreements?", "SLA_ServiceNow_Platform_FY25.pdf"),

    # --- SLA Details & Performance ---
    ("What is the uptime percentage target for Microsoft Teams in FY25?", "SLA_Microsoft_Teams_FY25.pdf"),
    ("What are the incident response time commitments in the ServiceNow SLA?", "SLA_ServiceNow_Platform_FY25.pdf"),
    ("Which penalties apply for SLA violations in FY25?", "SLA_Summary_Terms_FY25.pdf"),
    ("How are data-backup intervals defined for Snowflake in FY25?", "SLA_Snowflake_Data_Cloud_FY25.pdf"),
    ("What security compliance standards are included in Google Workspace FY25 SLA?", "SLA_Google_Workspace_FY25.pdf"),
    ("What are the FY25 service-availability KPIs for HCM Platform?", "SLA_HCM_Platform_FY25.pdf"),
    ("Which SLA section describes vendor-escalation procedures?", "SLA_Summary_Terms_FY25.pdf"),
    ("What monitoring tools are referenced in the Enterprise Data Platform agreement?", "SLA_Enterprise_Data_Platform_FY25.pdf"),
    ("How are change-management processes defined in SAP S4HANA SLA?", "SLA_SAP_S4HANA_FY25.pdf"),
    ("What are the disaster-recovery objectives for Oracle Database Platform in FY25?", "SLA_Oracle_Database_Platform_FY25.pdf"),
]



# -------------------------------------------------------------------
# Retrieval-Funktionen
# -------------------------------------------------------------------
def retrieve_bm25(query, top_k=10):
    scores = bm25.get_scores(query.split())
    return np.argsort(scores)[::-1][:top_k]

print("[Retrieval] Precomputing dense embeddings for all docs...")
doc_embeddings = [np.array(ef([text]))[0] for text in tqdm(texts, desc="Embedding docs")]

def retrieve_dense(query, top_k=10):
    qv = np.array(ef([query]))[0]
    sims = [np.dot(qv, dv) for dv in doc_embeddings]
    return np.argsort(sims)[::-1][:top_k]


def retrieve_hybrid(query, top_k=10):
    idx_dense = retrieve_dense(query, top_k * 2)
    idx_bm25 = retrieve_bm25(query, top_k * 2)
    ranks = {}
    for r, i in enumerate(idx_dense):
        ranks[i] = ranks.get(i, 0) + 1 / (r + 1)
    for r, i in enumerate(idx_bm25):
        ranks[i] = ranks.get(i, 0) + 1 / (r + 1)
    fused = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in fused[:top_k]]

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

records = []
timing_records = []

for q, gold in tqdm(QUESTIONS_AND_GOLDS, desc="[Retrieval Benchmark]"):
    for method, retriever in {
        "BM25": retrieve_bm25,
        "Dense": retrieve_dense,
        "Hybrid": retrieve_hybrid,
    }.items():
        t_start = time.time()
        idxs = retriever(q, top_k=10)
        t_retrieval = time.time() - t_start

        ranked_titles = [titles[i] for i in idxs]

        found = any(gold.lower() in t.lower() for t in ranked_titles)
        rank = (
            next((r + 1 for r, t in enumerate(ranked_titles) if gold.lower() in t.lower()), None)
            if found else None
        )

        records.append({
            "question": q,
            "method": method,
            "gold": gold,
            "found": found,
            "rank": rank,
            "top_titles": ranked_titles,
        })

        timing_records.append({
            "question": q,
            "method": method,
            "retrieval_time_sec": round(t_retrieval, 4),
            "found": found,
            "rank": rank
        })

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
timing_df = pd.DataFrame(timing_records)
timing_df.to_csv(OUT_TIMING, index=False)
print(f"[Retrieval] Timing measurements saved to {OUT_TIMING}")

# -------------------------------------------------------------------
# Kennzahlen berechnen
# -------------------------------------------------------------------
def recall_at_k(df, k):
    return (df["rank"].notna() & (df["rank"] <= k)).sum() / len(df["question"].unique())

def mean_reciprocal_rank(df):
    valid = df["rank"].dropna()
    return (1 / valid).mean() if not valid.empty else 0

summary_lines = ["\n=== Retrieval Benchmark Summary ===\n"]
for method in ["BM25", "Dense", "Hybrid"]:
    df_m = df[df["method"] == method]
    recall5 = recall_at_k(df_m, 5)
    mrr10 = mean_reciprocal_rank(df_m)
    avg_rank = df_m["rank"].mean()
    summary_lines.append(f"{method:8} | Recall@5: {recall5:.3f} | MRR@10: {mrr10:.3f} | AvgRank: {avg_rank:.2f}")

summary = "\n".join(summary_lines)
print(summary)

with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\n[Retrieval] Results written to {OUT_SUMMARY}")
