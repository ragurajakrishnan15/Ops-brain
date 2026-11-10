"""Centralized configuration — reads from .env or environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = DATA_DIR / "docs"


def _project_path(env_var: str, default: Path) -> Path:
    """Read a path from the environment; relative paths are relative to the project root."""
    path = Path(os.getenv(env_var, str(default)))
    return path if path.is_absolute() else PROJECT_ROOT / path


# Raw dataset (DataCo Smart Supply Chain, as downloaded from Kaggle)
RAW_CSV_PATH = _project_path("RAW_CSV_PATH", RAW_DIR / "DataCoSupplyChainDataset.csv")

# DuckDB
DUCKDB_PATH = str(_project_path("DUCKDB_PATH", PROCESSED_DIR / "warehouse.duckdb"))

# ChromaDB (its PostHog telemetry is incompatible with the installed posthog and only logs errors)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
CHROMA_PERSIST_DIR = str(_project_path("CHROMA_PERSIST_DIR", PROCESSED_DIR / "chroma_store"))

# Google Gemini (free key: https://aistudio.google.com/apikey)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_ROUTER_MODEL = os.getenv("GEMINI_ROUTER_MODEL", "gemini-3.5-flash-lite")

# MLflow — plain paths (relative to the project root) become file:// URIs,
# since MLflow reads a Windows path like D:\... as a URI with scheme "d"
_mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
if "://" not in _mlflow_uri:
    _mlflow_uri = (PROJECT_ROOT / _mlflow_uri).resolve().as_uri()
MLFLOW_TRACKING_URI = _mlflow_uri
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "opsbrain-forecast")

# Forecasting
# DataCo's catalog changes after 2017-10-02 (volume drops ~80%, most long-running
# categories end), so models are trained and evaluated on the stable period only.
MODEL_END_DATE = os.getenv("MODEL_END_DATE", "2017-10-02")
MIN_HISTORY_DAYS = int(os.getenv("MIN_HISTORY_DAYS", "365"))
FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "28"))

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
