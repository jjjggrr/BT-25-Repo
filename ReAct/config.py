
import os

# --- Cube.js / DuckDB ---
CUBEJS_API_URL = os.environ.get("CUBEJS_API_URL", "").strip()  # e.g. http://localhost:4000/cubejs-api/v1/load
DUCKDB_PATH     = os.environ.get("DUCKDB_PATH", "synthetic.duckdb")

# --- Chroma ---
CHROMA_HOST = os.environ.get("CHROMA_HOST", "").strip()  # leave empty for local inproc client
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "sla_docs")

# --- General ---
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
