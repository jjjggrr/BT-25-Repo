
from typing import List, Dict, Any
import os

from config import CHROMA_HOST, CHROMA_COLLECTION

class ChromaClient:
    """
    Tries to connect to Chroma. If unavailable, returns deterministic mock results.
    """
    def __init__(self):
        self.available = False
        self.client = None
        self.collection = None
        try:
            import chromadb
            if CHROMA_HOST:
                self.client = chromadb.HttpClient(host=CHROMA_HOST.replace("http://","").replace("https://",""))
            else:
                self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(CHROMA_COLLECTION)
            self.available = True
        except Exception:
            self.available = False

        # Optional keyword BM25 fallback (tiny scorer)
        try:
            from rank_bm25 import BM25Okapi  # noqa
            self.has_bm25 = True
        except Exception:
            self.has_bm25 = False

        # load docs for bm25 mock if chroma not available
        self._mock_docs = [
            {"doc_id":"SLA_APP_042", "section":"Pricing & Financial Model", "page":2,
             "text":"Unit price for compute increased by 7% in FY25 due to new vendor terms."},
            {"doc_id":"PROJ_CRM_FY25", "section":"Contract Overview", "page":1,
             "text":"CRM rollout phase 1 adds 1.2M compute minutes in FY25."},
            {"doc_id":"OPS_M365", "section":"Operational & Governance Notes", "page":3,
             "text":"Microsoft 365 seat growth in Org001 (+1%) impacted license tiers."}
        ]

    def query(self, query_texts: List[str], k:int=5) -> List[Dict[str,Any]]:
        if self.available:
            # Simplified: only first query, n_results=k
            res = self.collection.query(query_texts=query_texts, n_results=k)
            docs = []
            # Normalize to a stable list of dicts
            for i, _q in enumerate(query_texts):
                for j in range(len(res["documents"][i])):
                    docs.append({
                        "doc_id": res["ids"][i][j] if res.get("ids") else f"doc_{j}",
                        "section": res["metadatas"][i][j].get("section") if res.get("metadatas") else "unknown",
                        "page": res["metadatas"][i][j].get("page") if res.get("metadatas") else None,
                        "text": res["documents"][i][j],
                        "score": res["distances"][i][j] if res.get("distances") else None
                    })
            return docs[:k]

        # Fallback: crude keyword ranking over mock docs
        merged_query = " ".join(query_texts).lower()
        ranked = sorted(self._mock_docs, key=lambda d: -sum(int(w in d["text"].lower()) for w in merged_query.split()))
        return ranked[:k]
