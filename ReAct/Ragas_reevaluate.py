"""
Reruns RAGAS evaluation on an existing dataset file
using a local Ollama model (DeepSeek-R1 14B Q4) instead of OpenAI.
"""

import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------------------------------------------
#  Load config
# -------------------------------------------------------------------
load_dotenv()
DATASET_FILE = "results/RAGAS/ragas_dataset_20251112_192244.json"
BASE_DIR = "results/RAGAS"
os.makedirs(BASE_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

METRICS_PATH = os.path.join(BASE_DIR, f"ragas_metrics_{ts}.csv")
SUMMARY_PATH = os.path.join(BASE_DIR, f"ragas_summary_{ts}.txt")

# -------------------------------------------------------------------
#  Load dataset
# -------------------------------------------------------------------
print(f"[RAGAS] Loading dataset from {DATASET_FILE}")
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset_dict = json.load(f)

ragas_ds = Dataset.from_dict(dataset_dict)

# -------------------------------------------------------------------
#  Configure LOCAL model + embeddings
# -------------------------------------------------------------------
print("[RAGAS] Using local model (mistral:7b-instruct 7B Q4 via Ollama)")
llm = OllamaLLM(model="mistral:7b-instruct",)

# Local embeddings (MiniLM-L6-v2 is light and accurate for RAGAS)
print("[RAGAS] Using local embedding model (MiniLM-L6-v2)")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------------------------------------------------------
#  Evaluate with RAGAS using local LLM + embeddings
# -------------------------------------------------------------------
print("[RAGAS] Starting evaluation...")
results = evaluate(
    dataset=ragas_ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embedding_model
)

# -------------------------------------------------------------------
#  Save results
# -------------------------------------------------------------------
df = results.to_pandas()
df.to_csv(METRICS_PATH, index=False)

mean_scores = {
    "faithfulness": df["faithfulness"].mean(),
    "answer_relevancy": df["answer_relevancy"].mean(),
    "context_precision": df["context_precision"].mean(),
    "context_recall": df["context_recall"].mean(),
}

summary = (
    f"\n=== RAGAS Reevaluation (Local, {ts}) ===\n"
    f"Faithfulness:       {mean_scores['faithfulness']:.3f}\n"
    f"Answer Relevancy:   {mean_scores['answer_relevancy']:.3f}\n"
    f"Context Precision:  {mean_scores['context_precision']:.3f}\n"
    f"Context Recall:     {mean_scores['context_recall']:.3f}\n"
)

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)
print(f"[RAGAS] Results written to {SUMMARY_PATH}")
