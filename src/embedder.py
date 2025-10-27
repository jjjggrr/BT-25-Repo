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


class Embedder:
    def __init__(self, persist_dir: Path):
        self.client: Client = chromadb.Client(Settings(persist_directory=str(persist_dir)))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sa_col = self.client.get_or_create_collection("service_docs")
        self.prj_col = self.client.get_or_create_collection("project_docs")

    def _embed(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def _pack_metadata(self, meta: Dict) -> Dict:
        # Chroma metadata must be JSON-serializable and reasonably small
        return meta

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
        # Convert allocation objects to plain dicts for metadata
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