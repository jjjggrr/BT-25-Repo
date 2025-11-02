from typing import List, Dict, Any
import os
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_HOST, CHROMA_COLLECTION


class ChromaClient:
    """
    High-performance Chroma client for retrieval.
    - Uses GPU if CUDA available, else CPU
    - Uses same model as embedder (all-mpnet-base-v2)
    - Raises error if Chroma cannot be reached (no mock fallback)
    """

    def __init__(self):
        # Detect GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ChromaClient] Initializing with all-mpnet-base-v2 on device: {self.device}")

        # Initialize embedding function (identical to embedder)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2",
            device=self.device
        )

        # Connect to Chroma (local or HTTP)
        try:
            if CHROMA_HOST:
                # HTTP client (remote Chroma server)
                print(f"[ChromaClient] Connecting to remote Chroma host: {CHROMA_HOST}")
                self.client = chromadb.HttpClient(
                    host=CHROMA_HOST.replace("http://", "").replace("https://", "")
                )
            else:
                # Local persistent Chroma instance
                print("[ChromaClient] Connecting to local Chroma instance")
                self.client = chromadb.Client()

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                embedding_function=self.embedding_function
            )

            # Test connection
            _ = self.collection.count()
            print(f"[ChromaClient] Connected successfully to collection '{CHROMA_COLLECTION}'")

        except Exception as e:
            raise ConnectionError(
                f"[ChromaClient] Failed to connect to Chroma collection '{CHROMA_COLLECTION}': {e}"
            ) from e

    # ------------------ Query Interface ------------------

    def query(self, query_texts: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic retrieval for a list of text queries.
        Returns a list of document chunks with metadata and scores.
        """
        if not query_texts:
            raise ValueError("[ChromaClient] No query texts provided")

        try:
            res = self.collection.query(query_texts=query_texts, n_results=k)
        except Exception as e:
            raise RuntimeError(f"[ChromaClient] Query failed: {e}") from e

        if not res or "documents" not in res or not res["documents"]:
            print("[ChromaClient] Warning: No matching documents found.")
            return []

        docs: List[Dict[str, Any]] = []
        for i, _ in enumerate(query_texts):
            docs_for_query = len(res["documents"][i])
            for j in range(docs_for_query):
                docs.append({
                    "doc_id": res["ids"][i][j] if res.get("ids") else f"doc_{j}",
                    "section": res["metadatas"][i][j].get("section") if res.get("metadatas") else None,
                    "page": res["metadatas"][i][j].get("page") if res.get("metadatas") else None,
                    "text": res["documents"][i][j],
                    "score": res["distances"][i][j] if res.get("distances") else None
                })
        print(f"[ChromaClient] Retrieved {len(docs)} documents (top {k} per query)")
        return docs
