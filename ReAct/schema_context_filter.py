import json
import re
import difflib
from pathlib import Path

# === CONFIG ===
SCHEMA_PATH = Path("schema_cache_llm.json")  # oder dein aktueller Pfad
FUZZY_CUTOFF = 0.7  # Mindestscore für Fuzzy-Matches
CORE_DIMENSIONS = [
    "DimOrg.businessUnit",
    "DimApp.appName",
    "DimService.serviceName",
    "DimTower.towerName",
    "DimCountry.countryName",
]


# === Hilfsfunktionen ===
def normalize_text(text: str) -> str:
    """Normalize query text: lowercase, strip punctuation, normalize fiscal years."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize year patterns: 2025 -> FY25, fy2024 -> FY24
    text = re.sub(r"\b20(\d{2})\b", r"fy\1", text)
    text = re.sub(r"\bfy\s*(\d{2})\b", r"fy\1", text)
    return text

def detect_unspecific_query(normalized_q: str) -> bool:
    """Heuristisch erkennen, ob die Frage zu unspezifisch ist (z. B. 'overall', 'total')."""
    unspecific_terms = [
        "overall", "total", "all", "everything", "across", "combined",
        "complete", "global", "aggregate", "organization", "entire"
    ]
    for t in unspecific_terms:
        if t in normalized_q:
            return True
    return False

def fuzzy_match_one(query_text: str, candidates: list[str], cutoff: float = 0.6):
    """Hybrid substring + fuzzy matching for robust entity detection."""
    if not candidates:
        return None, 0.0

    query_lower = query_text.lower()
    best_match, best_score = None, 0.0

    for cand in candidates:
        cand_lower = cand.lower()

        # 1) Exact match
        if cand_lower == query_lower:
            return cand, 1.0

        # 2) Substring containment (Finance ⊂ Finance & Controlling)
        if query_lower in cand_lower or cand_lower in query_lower:
            score = 0.95  # High confidence for substring matches
        else:
            # 3) Fuzzy fallback
            score = difflib.SequenceMatcher(None, query_lower, cand_lower).ratio()

        if score > best_score:
            best_match, best_score = cand, score

    if best_score >= cutoff:
        return best_match, best_score
    return None, best_score



def extract_relevant_schema_context(question: str, schema_cache: dict, verbose: bool = False) -> dict:
    FUZZY_CUTOFF = 0.7
    cutoff = FUZZY_CUTOFF

    """Extracts the relevant subset of schema_cache based on fuzzy matching with the question text."""
    normalized_q = normalize_text(question)
    tokens = normalized_q.split()
    valid_values = schema_cache.get("valid_values", {})
    measures = schema_cache.get("measures", [])
    dimensions = schema_cache.get("dimensions", [])

    matched_values = {}
    matched_dims = set()

    # --- 1) Versuch: Fuzzy-Matching auf gesamten Text + Token-Fallback ---
    for dim, vals in valid_values.items():
        if not isinstance(vals, list):
            continue

        # 1a) Phrasenbasiertes Matching gegen gesamten Query-Text
        best_phrase, score_phrase = fuzzy_match_one(normalized_q, vals, cutoff)
        if best_phrase and score_phrase >= cutoff:
            matched_values.setdefault(dim, set()).add(best_phrase)
            matched_dims.add(dim)
            continue  # Wenn wir schon ein gutes Match auf Satzebene haben, Token-Fallback überspringen

        # 1b) Token-Fallback (z. B. "Teams" matcht "Microsoft Teams")
        for token in tokens:
            best, score = fuzzy_match_one(token, vals, cutoff)
            if best and score >= cutoff:
                matched_values.setdefault(dim, set()).add(best)
                matched_dims.add(dim)

    # --- 2) Fiscal Year-Handling (FY24, FY25, 2025 etc.) ---
    fy_matches = re.findall(r"fy\d{2}", normalized_q)
    if fy_matches:
        fy_matches = {fy.upper() for fy in fy_matches}  # dedupe + normalize
        for fy in fy_matches:
            matched_values.setdefault("DimDate.fiscalYear", set()).add(fy)
            matched_dims.add("DimDate.fiscalYear")

    # --- 3) Fallback: Wenn nichts erkannt, komplettes Schema zurückgeben ---
    if not matched_values:
        print("[SchemaFilter] No relevant entities detected — using full schema.")
        return schema_cache

    # --- 4) Measures (alle Fct*-Measures aus beliebiger Schema-Struktur) ---
    matched_measures = []

    # 1. flache Struktur (z. B. {"measures": ["FctItCosts.actualCost", ...]})
    flat_measures = schema_cache.get("measures", [])
    if flat_measures:
        matched_measures.extend([m for m in flat_measures if m.startswith("FctItCosts.") or m.startswith("Fct")])

    # 2. verschachtelte Struktur auf oberster Ebene ({"FctItCosts": {"measures": [...]}})
    for k, v in schema_cache.items():
        if isinstance(v, dict) and "measures" in v:
            if k.startswith("FctItCosts") or k.startswith("Fct"):
                matched_measures.extend(v["measures"])

    # 3. Struktur unter "cubes" ({"cubes": {"FctItCosts": {"measures": [...]}}})
    cubes = schema_cache.get("cubes", {})
    if isinstance(cubes, dict):
        for k, v in cubes.items():
            if isinstance(v, dict) and "measures" in v:
                if k.startswith("FctItCosts") or k.startswith("Fct"):
                    matched_measures.extend(v["measures"])

    matched_measures = sorted(list(set(matched_measures)))

    if not matched_measures:
        print("[SchemaFilter] Warning: no FctItCosts measures found in schema cache.")

    # --- 5) Schema-Reduktion ---
    reduced = {
        "measures": matched_measures,
        "dimensions": sorted(list(matched_dims)),
        "valid_values": {k: list(v) for k, v in matched_values.items()}
    }

    # --- 6) Detect generic / meaningless queries ---
    def is_generic_query(q: str, matched_dims: set) -> bool:
        generic_terms = [
            "overall", "total", "all", "everything", "nothing", "test", "compare costs",
            "aggregate", "complete", "general", "overview", "summary"
        ]
        # enthält ein generisches Wort?
        has_generic_word = any(t in q for t in generic_terms)
        # fehlen alle wesentlichen Entitäten?
        core_entities = [
            "DimOrg.businessUnit",
            "DimApp.appName",
            "DimService.serviceName",
            "DimTower.towerName",
            "DimCountry.countryName",
            "DimDate.fiscalYear",
            "FctItCosts.fiscalYear",
        ]
        has_core_entity = any(d in core_entities for d in matched_dims)
        has_no_entities = not has_core_entity

        return has_generic_word or has_no_entities

    if is_generic_query(normalized_q, matched_dims):
        print("[SchemaFilter] Generic or meaningless query detected → returning full schema.")
        return schema_cache

    # --- 7) Ensure Fiscal Year always included ---
    # --- Ensure Fiscal Year always included (nur wenn noch kein FY da ist) ---
    has_fy = any("fiscalyear" in d.lower() for d in matched_dims)
    if not has_fy:
        # nimm FYs aus valid_values (egal ob unter DimDate oder FctItCosts)
        fy_vals = (valid_values.get("FctItCosts.fiscalYear")
                   or valid_values.get("DimDate.fiscalYear")
                   or ["FY24", "FY25"])  # Fallback
        # >>> WICHTIG: direkt in matched_values eintragen (set!)
        matched_values.setdefault("FctItCosts.fiscalYear", set()).update(fy_vals)
        matched_dims.add("FctItCosts.fiscalYear")
        print(f"[SchemaFilter] Added fiscal year context (no FY specified in query) → {', '.join(fy_vals)}")

    # --- Build reduced schema (ohne irgendwas rauszufiltern) ---
    reduced = {
        "measures": sorted(set(m for m in matched_measures)),  # wie gehabt
        "dimensions": sorted(matched_dims),
        "valid_values": {k: sorted(list(v)) for k, v in matched_values.items()}
    }

    # --- Debug-Ausgabe ---
    print("\n=== [SchemaFilter] Relevant Context ===")
    print(f"Original Question: {question}")
    print(f"Normalized: {normalized_q}\n")
    for dim, vals in reduced["valid_values"].items():
        print(f"  {dim}: {', '.join(vals)}")
    print("\n  Measures: " + ", ".join(reduced["measures"]))
    print("  Dimensions: " + ", ".join(reduced["dimensions"]))
    print("========================================\n")

    return reduced


# === Main Test ===
if __name__ == "__main__":
    # Schema laden
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")
    schema_cache = json.loads(Path(SCHEMA_PATH).read_text(encoding="utf-8"))

    while True:
        q = input("\nEnter test query (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        extract_relevant_schema_context(q, schema_cache)
