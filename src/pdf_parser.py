from pathlib import Path
import re
from typing import List, Dict, Any
from pdfminer.high_level import extract_text

SECTION_PATTERN = re.compile(r"###SECTION:\s*([a-zA-Z0-9_]+)")
META_PATTERN = re.compile(r"Meta:\s*(.*?)###SECTION:", re.DOTALL)

def parse_structured_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extrahiert strukturierte Metadaten und Abschnitte aus einem PDF,
    das Meta: und ###SECTION: Marker enthält.
    """
    text = extract_text(str(pdf_path))
    if not text or len(text.strip()) < 30:
        raise ValueError(f"PDF text extraction failed or too short: {pdf_path}")

    # --- Meta-Block extrahieren ---
    meta_block_match = re.search(r"Meta:(.*?)(###SECTION:|$)", text, flags=re.DOTALL)
    meta = {}
    if meta_block_match:
        meta_raw = meta_block_match.group(1)
        for line in meta_raw.splitlines():
            if ":" not in line:
                continue
            key, val = [x.strip() for x in line.split(":", 1)]
            if key and val:
                meta[key] = val
    else:
        meta = {"DocType": "Unknown"}

    # --- Sections extrahieren ---
    sections: List[Dict[str, Any]] = []
    matches = list(SECTION_PATTERN.finditer(text))
    for i, m in enumerate(matches):
        section_name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        sections.append({
            "section": section_name,
            "text": chunk_text
        })

    return {"meta": meta, "sections": sections}
