
import os
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

CUBEJS_URL = os.environ.get("CUBEJS_API_URL", "http://localhost:4000/cubejs-api/v1")

CACHE_PATH = "schema_cache.json"
LLM_PATH = "schema_cache_llm.json"
MAX_THREADS = 8
LIMIT = 5000


# ---------------- Core Fetch Functions ----------------

def get_cube_meta():
    """Fetches all cubes, measures, and dimensions from Cube.js /v1/meta"""
    url = f"{CUBEJS_URL.rstrip('/')}/meta"
    r = requests.get(url)
    r.raise_for_status()
    meta = r.json().get("cubes", [])
    return meta


def pick_default_measure(cube):
    """
    Always return a usable measure name.
    Prefer the cube’s own first measure (including .count),
    otherwise fall back to FctItCosts.actualCost.
    """
    measures = [m["name"] for m in cube.get("measures", [])]
    if measures:
        return measures[0]
    return "FctItCosts.actualCost"



def get_valid_values_for_dimension(dimension, measure):
    """Fetch distinct values for a single dimension via Cube.js /v1/load"""
    payload = {
        "query": {
            "measures": [measure],
            "dimensions": [dimension],
            "limit": LIMIT
        }
    }
    url = f"{CUBEJS_URL.rstrip('/')}/load"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    rows = r.json().get("data", [])
    return sorted({row[dimension] for row in rows if row.get(dimension)})


def build_schema_cache():
    """Fetches metadata and valid dimension values, writes schema_cache.json"""
    print("[cube_meta] Fetching Cube.js metadata ...")
    meta = get_cube_meta()
    print(f"[cube_meta] Found {len(meta)} cubes: {[c['name'] for c in meta]}")

    valid_values = {}
    start = time.time()

    def fetch_dim(dim_name, measure_name):
        try:
            vals = get_valid_values_for_dimension(dim_name, measure_name)
            return dim_name, vals
        except Exception as e:
            return dim_name, f"ERROR: {e}"

    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for cube in meta:
            measure = pick_default_measure(cube) or "FctItCosts.actualCost"
            for dim in cube.get("dimensions", []):
                dim_name = dim["name"]
                tasks.append(executor.submit(fetch_dim, dim_name, measure))

        for i, f in enumerate(as_completed(tasks), 1):
            dim_name, result = f.result()
            if isinstance(result, str) and result.startswith("ERROR"):
                print(f"[{i:03}] {dim_name}: {result}")
            else:
                valid_values[dim_name] = result
                print(f"[{i:03}] {dim_name}: {len(result)} values")

    cache = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "valid_values": valid_values,
    }

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


    print(f"[cube_meta] Schema cache written → {CACHE_PATH} "
          f"({len(valid_values)} dimensions, {round(time.time()-start,1)}s)")

    with open(LLM_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    try:
        build_llm_schema(cache_path=LLM_PATH)
    except Exception as e:
        print(f"[cube_meta] Warning: failed to build LLM schema: {e}")
    return cache

def build_llm_schema(
    cache_path: str = "schema_cache.json",
    out_path: str = "schema_cache_llm.json",
    max_values_per_field: int = 50,   # Werte pro Feld begrenzen (LLM-freundlich)
) -> dict:
    import json, re

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # --- 1) Cubes kompakt (nur Namen der Measures/Dimensions) ---
    cubes = {}
    for c in cache.get("meta", []):
        cube_name = c.get("name")
        if not cube_name:
            continue
        measures = [m.get("name") for m in c.get("measures", []) if m.get("name")]
        dims     = [d.get("name") for d in c.get("dimensions", []) if d.get("name")]
        if measures or dims:
            cubes[cube_name] = {"measures": measures, "dimensions": dims}

    # --- 2) valid_values filtern ---
    # wir filtern Felder, deren *Feldname* (der Teil nach dem Punkt) wie ID/Key/UUID/… aussieht
    skip_re = re.compile(r"^(id|uuid|guid|hash|key|rowid|created(_at)?|updated(_at)?|timestamp)$", re.IGNORECASE)

    def is_skip_field(full_field_name: str) -> bool:
        # z.B. "FctItCosts.id" -> "id"
        tail = (full_field_name or "").split(".")[-1]
        if skip_re.match(tail):
            return True
        # explizit FctItCosts.id entfernen (falls du ganz sicher gehen willst)
        if full_field_name == "FctItCosts.id":
            return True
        return False

    raw_valid = cache.get("valid_values", {})
    filtered_valid = {}
    for field, values in raw_valid.items():
        if is_skip_field(field):
            continue
        # Werte listenkürzen & Duplikate entfernen, Reihenfolge beibehalten
        if isinstance(values, list):
            seen = set()
            dedup = []
            for v in values:
                if v not in seen:
                    seen.add(v)
                    dedup.append(v)
                if len(dedup) >= max_values_per_field:
                    break
            filtered_valid[field] = dedup
        else:
            # falls mal kein Listentyp – überspringen oder 1:1 kopieren
            filtered_valid[field] = values

    compact = {
        "cubes": cubes,
        "valid_values": filtered_valid,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    print(f"[cube_meta] Compact LLM schema written → {out_path} "
          f"(valid fields: {len(filtered_valid)}/{len(raw_valid)})")
    return compact





def load_schema_cache(force_refresh=False):
    """Load cache from disk or rebuild if missing/forced"""
    if force_refresh or not os.path.exists(CACHE_PATH):
        return build_schema_cache()
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    build_schema_cache()
