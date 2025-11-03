from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_COLLECTION, CHROMA_HOST


class ChromaClient:
    """
    Enhanced Chroma client for semantic retrieval.
    - Query expansion
    - Cosine-similarity filtering
    - Section + DocType prioritization
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        self.client = chromadb.PersistentClient(path="data/embeddings")

        self.collections = {
            "service_agreements": self.client.get_or_create_collection("service_agreements"),
            "project_briefs": self.client.get_or_create_collection("project_briefs"),
        }

    # ---------------- QUERY ----------------

    def _expand_query(self, q: str) -> List[str]:
        """Lightweight query expansion."""
        base = [q]
        q_lower = q.lower()
        if "why" in q_lower:
            base.append(q + " root cause explanation pricing change cost increase")
        if "cost" in q_lower:
            base.append(q + " service level agreement SLA pricing model financial terms FY25 FY24")
        if "microsoft" in q_lower:
            base.append(q + " M365 Office365 licensing SLA")
        if "total" in q_lower:
            base.append(q + " aggregate cost total IT spend summary")
        return list(set(base))

    def query(
            self,
            query_texts: list[str],
            k: int = 5,
            min_similarity: float = 0.25,
            section_filter: list[str] | None = None,
            doc_type: str | None = None,
    ) -> list[dict[str, any]]:
        """
        Perform semantic retrieval across the correct Chroma collection.
        - Auto-selects service_agreements vs project_briefs
        - Expands queries for better match
        - Filters weak matches
        - Prefers relevant sections (pricing, sla, changes)
        """

        if not query_texts:
            raise ValueError("[ChromaClient] No query texts provided")

        # --- Step 1: Determine collection based on doc_type or heuristics ---
        if not doc_type:
            q_lower = " ".join(query_texts).lower()
            if any(word in q_lower for word in ["project", "initiative", "investment", "capex"]):
                doc_type = "project_briefs"
            else:
                doc_type = "service_agreements"

        collection = self.collections.get(doc_type)
        if not collection:
            raise ValueError(f"[ChromaClient] Unknown doc_type: {doc_type}")

        print(f"[ChromaClient] Querying collection: {doc_type}")

        # --- Step 2: Expand queries for richer retrieval ---
        expanded_queries = []
        for q in query_texts:
            expanded_queries.extend(self._expand_query(q))

        # --- Step 3: Execute query ---
        try:
            res = collection.query(query_texts=expanded_queries, n_results=k * 2)
        except Exception as e:
            print(f"[ChromaClient] ERROR during Chroma query: {e}")
            return []

        if not res or not res.get("documents"):
            print(f"[ChromaClient] No results found in {doc_type}.")
            return []

        # --- Step 4: Process and filter results ---
        results: list[dict[str, any]] = []
        for i, docs in enumerate(res["documents"]):
            for j, text in enumerate(docs):
                meta = res["metadatas"][i][j]
                score = 1.0 - res["distances"][i][j] if res.get("distances") else 0.0

                # Drop low-similarity matches
                if score < min_similarity:
                    continue

                # Optional: filter by section (e.g. only pricing or SLA)
                if section_filter and meta.get("section") not in section_filter:
                    continue

                results.append({
                    "doc_id": res["ids"][i][j],
                    "text": text,
                    "score": score,
                    "section": meta.get("section"),
                    "doctype": meta.get("doc_type"),
                    "service_id": meta.get("service_id"),
                    "fiscal_year": meta.get("fiscal_year"),
                })

        if not results:
            print(f"[ChromaClient] Retrieved 0 relevant documents after filtering ({doc_type}).")
            return []

        # --- Step 5: Re-rank for section priority ---
        for r in results:
            if r["section"] and "pricing" in r["section"]:
                r["score"] += 0.05
            if r["section"] and "sla" in r["section"]:
                r["score"] += 0.03
            if r["section"] and "changes" in r["section"]:
                r["score"] += 0.02

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        topk = results[:k]

        print(f"[ChromaClient] Retrieved {len(topk)} relevant documents (from {len(results)} candidates) in {doc_type}")
        return topk

