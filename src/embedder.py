from datetime import datetime
from chromadb.utils import embedding_functions
import torch
from pathlib import Path
from .config import EMB_DIR  # Pfad zu deinen Embeddings, z. B. data/embeddings


class Embedder:
    def __init__(self, base_dir=None):
        import chromadb

        # === 1) Modellkonfiguration ===
        self.model_name = "sentence-transformers/all-mpnet-base-v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedder] Initializing {self.model_name} on device: {self.device}")

        # === 2) Embedding-Funktion definieren ===
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name,
            device=self.device,
        )

        # === 3) PersistentClient erstellen ===
        emb_path = Path(EMB_DIR)
        emb_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(emb_path))

        # === 4) interne Helper-Funktion zur robusten Collection-Erstellung ===
        def _get_or_recreate(name: str):
            try:
                return self.client.get_or_create_collection(
                    name=name,
                    embedding_function=self.embed_fn,
                )
            except ValueError as e:
                msg = str(e).lower()
                if "embedding function already exists" in msg or "conflict" in msg:
                    print(f"[Embedder] Detected embedding function conflict in '{name}', recreating collection...")
                    self.client.delete_collection(name)
                    return self.client.get_or_create_collection(
                        name=name,
                        embedding_function=self.embed_fn,
                    )
                raise

        # === 5) Zwei persistente Collections ===
        self.sa_col = _get_or_recreate("service_agreements")
        self.prj_col = _get_or_recreate("project_briefs")

    # === 6) SLA-Dokumente (Apps) hinzufügen ===
    def add_service_chunks(self, pdf_name, meta, chunks, sections):
        for chunk, section in zip(chunks, sections):
            metadata = {
                "doc_type": meta.get("DocType", "SLA"),
                "fiscal_year": meta.get("FiscalYear"),
                "service_id": meta.get("ServiceID"),
                "section": section,
            }
            self.sa_col.add(
                documents=[chunk],
                metadatas=[metadata],
                ids=[f"{meta.get('ServiceID')}_{meta.get('FiscalYear')}_{section}_{hash(chunk)}"],
            )
        print(f"[Embedder] Added {len(chunks)} chunks from {pdf_name} (DocType={meta.get('DocType')})")

    # === 7) Projekt-Dokumente hinzufügen ===
    def add_project_chunks(self, pdf_name, meta, chunks, sections):
        for chunk, section in zip(chunks, sections):
            metadata = {
                "doc_type": meta.get("DocType") or "PRJ",
                "fiscal_year": meta.get("FiscalYear") or "UNKNOWN",
                "project_id": meta.get("ProjectID") or "UNKNOWN",
                "business_unit": meta.get("BusinessUnit") or "UNKNOWN",
                "section": section or "unspecified",
            }

            # Chroma akzeptiert keine None-Typen – sicherstellen, dass alles str/bool/float/int ist
            metadata = {k: (str(v) if v is not None else "UNKNOWN") for k, v in metadata.items()}

            self.prj_col.add(
                documents=[chunk],
                metadatas=[metadata],
                ids=[f"{metadata['project_id']}_{metadata['fiscal_year']}_{metadata['section']}_{abs(hash(chunk))}"],
            )

        print(f"[Embedder] Added {len(chunks)} chunks from {pdf_name} (DocType={metadata.get('doc_type')})")
