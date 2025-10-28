from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .models import ServiceDocMeta, ProjectDocMeta, ProjectAllocation

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False


class Embedder:
    def __init__(self, persist_dir: Path):
        self.client: Client = chromadb.Client(Settings(persist_directory=str(persist_dir)))
        device = "cuda" if _HAS_CUDA else "cpu"
        self.model = SentenceTransformer("all-mpnet-base-v2", device=device)
        self.sa_col = self.client.get_or_create_collection("service_docs")
        self.prj_col = self.client.get_or_create_collection("project_docs")

    def _embed(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def _pack_metadata(self, meta: Dict) -> Dict:
        """Clean metadata dict for Chroma compatibility (no None, only primitives or JSON strings)."""
        clean = {}
        for k, v in meta.items():
            if v is None:
                continue
            # Primitives bleiben unverändert
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            # Listen oder Dicts -> JSON-String
            elif isinstance(v, (list, dict)):
                try:
                    clean[k] = json.dumps(v, default=str)
                except Exception:
                    clean[k] = str(v)
            # Sonst -> String-Repräsentation
            else:
                clean[k] = str(v)
        return clean

    def add_service_chunks(self, *, pdf_name: str, meta: ServiceDocMeta, chunks: List[str]):
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

    def add_project_chunks(self, *, pdf_name: str, meta: ProjectDocMeta, chunks: List[str]):
        now = datetime.utcnow().isoformat() + "Z"
        alloc = meta.allocation_vector
        meta_dict = meta.model_dump()
        if alloc is not None:
            meta_dict["allocation_vector"] = [a.model_dump() if isinstance(a, ProjectAllocation) else a for a in alloc]
        meta_dict.update({
            "source_pdf_name": pdf_name,
            "generated_at_utc": now,
        })
        embeddings = self._embed(chunks)
        ids = [f"prj::{pdf_name}::sec::{i}" for i in range(len(chunks))]
        metadatas = [self._pack_metadata(meta_dict) for _ in chunks]
        self.prj_col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)