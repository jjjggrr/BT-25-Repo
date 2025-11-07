from typing import List, Dict, Any, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path


class ChromaClient:
    """
    Enhanced Chroma client for semantic retrieval.
    Supports contextual filtering by AppName and Fiscal Year (FY).
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        ROOT_DIR = Path(__file__).resolve().parents[1]  # geht 1 Ordner über 'ReAct' hinaus
        EMB_PATH = ROOT_DIR / "data" / "embeddings"
        self.client = chromadb.PersistentClient(path=str(EMB_PATH))

        # convenience handles
        self.sa_col = self.client.get_or_create_collection("service_agreements")
        self.prj_col = self.client.get_or_create_collection("project_briefs")

    # ---------------- BASIC QUERY ----------------
    def query(self, question: str, top_k: int = 5, fiscal_year: Optional[str] = None, app_name: Optional[str] = None):
        """
        Führt semantische Suche über SLA-Dokumente aus,
        filtert anschließend nach app_name und fiscal_year (analog zu test.py).
        Gibt formatierte Textblöcke für das LLM zurück.
        """

        query_text = question.strip()
        print(f"[ChromaClient] Querying for: '{query_text}' (FY={fiscal_year}, App={app_name})")

        # --- 1) Basissuche ---
        res = self.sa_col.query(
            query_texts=[query_text],
            n_results=top_k * 10,  # erstmal mehr holen
            where={"fiscal_year": fiscal_year} if fiscal_year else None,
        )

        if not res or not res.get("documents"):
            print("[ChromaClient] No results from base query.")
            return []

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        scores = res["distances"][0]

        # --- 2) Filter nach FY + AppName ---
        fy = (fiscal_year or "").strip().upper()
        app = (app_name or "").strip().lower()

        filtered = []
        for d, m, s in zip(docs, metas, scores):
            fy_meta = (m.get("fiscal_year") or "").upper()
            app_meta = (m.get("app_name") or "").lower()
            if fy and fy_meta != fy:
                continue
            if app and app_meta != app:
                continue
            filtered.append((d, m, s))

        if not filtered:
            print(f"[ChromaClient] No chunks found for app '{app_name}' ({fiscal_year})")
            return []

        print(f"[ChromaClient] Filtered to {len(filtered)} chunks for '{app_name}' ({fiscal_year})")

        # --- 3) Deduplicate nach section ---
        seen = set()
        final = []
        for d, m, s in filtered:
            sec = (m.get("section") or "").lower()
            if sec not in seen:
                seen.add(sec)
                final.append((d, m, s))
            if len(final) >= top_k:
                break

        # --- 4) Formatierte Ausgabe ---
        formatted = []
        for d, m, s in final:
            fy = m.get("fiscal_year", "?")
            app = m.get("app_name", "?")
            sec = m.get("section", "?")
            formatted.append(
                f"[{fy} | {app} | {sec}] (score={s:.3f})\n{d.strip()[:500]}"
            )

        print(f"[ChromaClient] Returning {len(formatted)} formatted chunks.")
        return formatted

    # ---------------- DOCUMENT EXPANSION (unchanged) ----------------
    def _expand_documents(self, collection, query_res, key="service_id"):
        """Hilfsfunktion: holt alle Chunks der getroffenen Dokumente nach (Document Expansion)."""
        expanded = []
        matched_ids = set()

        for m in query_res.get("metadatas", [[]])[0]:
            doc_id = m.get(key)
            if doc_id:
                matched_ids.add(doc_id)

        for doc_id in matched_ids:
            try:
                doc_chunks = collection.get(where={key: doc_id})
                if not doc_chunks or "documents" not in doc_chunks:
                    continue
                for doc, meta in zip(doc_chunks["documents"], doc_chunks["metadatas"]):
                    expanded.append({
                        "id": meta.get(key, "UNKNOWN"),
                        "text": doc,
                        "meta": meta,
                    })
                print(f"[ChromaClient] Expanded document {doc_id} with {len(doc_chunks['documents'])} chunks.")
            except Exception as e:
                print(f"[ChromaClient] Expansion failed for {doc_id}: {e}")

        return expanded

    # ---------------- ADVANCED QUERY: MULTI-YEAR / SECTION ----------------
    def query_expanded_for_service(
        self,
        question: str,
        service_id: Optional[str] = None,
        app_name: Optional[str] = None,
        fiscal_years: Optional[List[str]] = None,
        section_filter: Optional[List[str]] = None,
        top_k_init: int = 8,
        max_snippets: int = 6,
        max_chars_per_snippet: int = 320,
    ) -> List[str]:
        """
        Erweiterte Query-Variante:
        - Unterstützt mehrere Fiscal Years
        - Filtert nach ServiceID oder AppName
        - Optional: section_filter
        - Gibt kurze, formatierte Textausschnitte zurück
        """

        query_text = question.strip()
        print(f"[ChromaClient] Expanded query for '{query_text}' (Service={service_id or app_name})")

        res = self.sa_col.query(
            query_texts=[query_text],
            n_results=top_k_init * 3,
        )

        if not res or not res.get("documents"):
            print("[ChromaClient] No initial results.")
            return []

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        scores = res["distances"][0]

        # --- Filter nach Kriterien ---
        fiscal_years = [fy.upper() for fy in (fiscal_years or [])]
        filtered = []

        for d, m, s in zip(docs, metas, scores):
            fy = (m.get("fiscal_year") or "").upper()
            sid = (m.get("service_id") or "").upper()
            aname = (m.get("app_name") or "").lower()
            sec = (m.get("section") or "").lower()

            # robustere App-Übereinstimmung (substring statt exact)
            app_ok = True
            if app_name:
                app_q = app_name.lower().strip()
                app_ok = app_q in aname or aname in app_q

            fy_ok = not fiscal_years or fy in fiscal_years
            sid_ok = not service_id or sid == service_id.upper()
            sec_ok = not section_filter or sec in [x.lower() for x in section_filter]

            if fy_ok and app_ok and sid_ok and sec_ok:
                filtered.append((d, m, s))

        if not filtered:
            print(f"[ChromaClient] No matches for {service_id or app_name}.")
            return []

        print(f"[ChromaClient] Filtered to {len(filtered)} relevant chunks after constraints.")

        # --- Deduplicate by (FY, Section) ---
        seen = set()
        final = []
        for d, m, s in filtered:
            key = (m.get("fiscal_year"), (m.get("section") or "").lower())
            if key not in seen:
                seen.add(key)
                final.append((d, m, s))
            if len(final) >= max_snippets:
                break

        # --- Format output ---
        out = []
        for d, m, s in final:
            fy = m.get("fiscal_year", "?")
            app = m.get("app_name", "?")
            sec = m.get("section", "?")
            snippet = d.strip().replace("\n\n", "\n")[:max_chars_per_snippet]
            out.append(f"[{fy} | {app} | {sec}] (score={s:.3f})\n{snippet}")

        print(f"[ChromaClient] Returning {len(out)} contextual snippets.")
        return out

    def list_docs(self, collection_name="service_agreements", limit=10):
        col = self.sa_col if collection_name == "service_agreements" else self.prj_col
        items = col.get(limit=limit)
        print(f"[ChromaClient] Listing {len(items['ids'])} docs from {collection_name}:")
        for i, m in enumerate(items["metadatas"][:limit]):
            print(f"{i + 1:2d}. {m.get('file_name', '<no name>')} | {m.get('app_name')} | {m.get('fiscal_year')}")


