from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
    _DEVICE = "cuda" if _HAS_CUDA else "cpu"
except Exception:
    _HAS_CUDA = False
    _DEVICE = "cpu"


class Embedder:
    """
    Handles embedding generation and ingestion into ChromaDB.
    Uses GPU (CUDA) if available, else falls back to CPU.
    """

    def __init__(self, persist_dir: Path):
        # Persistent Chroma client
        self.client: Client = chromadb.Client(Settings(persist_directory=str(persist_dir)))

        # Log device info
        print(f"[Embedder] Initializing all-mpnet-base-v2 on device: {_DEVICE}")
        self.model = SentenceTransformer("all-mpnet-base-v2", device=_DEVICE)

        # Create or load collections
        self.sa_col = self.client.get_or_create_collection("service_docs")
        self.prj_col = self.client.get_or_create_collection("project_docs")

    # ---------------- Embedding core ----------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text chunks."""
        if not texts:
            return []
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False).tolist()

    # ---------------- Metadata helpers ----------------

    def _pack_metadata(self, meta: Dict) -> Dict:
        """Clean metadata dict for Chroma compatibility (no None, only primitives or JSON strings)."""
        clean = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, dict)):
                try:
                    clean[k] = json.dumps(v, default=str)
                except Exception:
                    clean[k] = str(v)
            else:
                clean[k] = str(v)
        return clean

    # ---------------- Add documents ----------------

    def add_service_chunks(self, *, pdf_name: str, meta, chunks: List[str]):
        now = datetime.utcnow().isoformat() + "Z"
        meta_dict = meta.model_dump()
        meta_dict.update({
            "source_pdf_name": pdf_name,
            "generated_at_utc": now,
        })
        embeddings = self._embed(chunks)
        ids = [f"sa::{pdf_name}::sec::{i}" for i in range(len(chunks))]
        metadatas = [self._pack_metadata(meta_dict) for _ in chunks]
        self.sa_col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    def add_project_chunks(self, *, pdf_name: str, meta, chunks: List[str]):
        now = datetime.utcnow().isoformat() + "Z"
        alloc = getattr(meta, "allocation_vector", None)
        meta_dict = meta.model_dump()
        if alloc is not None:
            from .models import ProjectAllocation
            meta_dict["allocation_vector"] = [
                a.model_dump() if isinstance(a, ProjectAllocation) else a for a in alloc
            ]
        meta_dict.update({
            "source_pdf_name": pdf_name,
            "generated_at_utc": now,
        })
        embeddings = self._embed(chunks)
        ids = [f"prj::{pdf_name}::sec::{i}" for i in range(len(chunks))]
        metadatas = [self._pack_metadata(meta_dict) for _ in chunks]
        self.prj_col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
