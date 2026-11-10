"""
Prebuilt data bundle for deployments.

The raw Kaggle CSV isn't in the repository, so a hosted copy of the app
(Posit Connect Cloud, a fresh clone) downloads a zip with everything the
dashboard reads: the DuckDB warehouse, the Chroma document store and the
trained MLflow models. Build it with `python scripts/build_bundle.py` and
attach it to a GitHub release; DATA_BUNDLE_URL points at that asset.
"""

import hashlib
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from src.utils.config import (
    CHROMA_PERSIST_DIR,
    DUCKDB_PATH,
    MLFLOW_EXPERIMENT,
    PROJECT_ROOT,
)

DATA_BUNDLE_URL = os.getenv(
    "DATA_BUNDLE_URL",
    "https://github.com/ragurajakrishnan15/Ops-brain/releases/download/data-v1/opsbrain-data.zip",
)
DATA_BUNDLE_SHA256 = os.getenv("DATA_BUNDLE_SHA256", "")

# Only these top-level folders may be written from the bundle
BUNDLE_ROOTS = ("data/processed/", "mlruns/")


def data_is_present() -> bool:
    """True when the warehouse, document store and at least one trained model exist."""
    mlruns = PROJECT_ROOT / "mlruns"
    has_models = mlruns.exists() and any(
        (p / "meta.yaml").exists() and MLFLOW_EXPERIMENT in (p / "meta.yaml").read_text()
        for p in mlruns.iterdir()
        if p.is_dir()
    )
    return Path(DUCKDB_PATH).exists() and Path(CHROMA_PERSIST_DIR).exists() and has_models


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        target = (PROJECT_ROOT / name).resolve()
        if not name.startswith(BUNDLE_ROOTS) or PROJECT_ROOT.resolve() not in target.parents:
            raise ValueError(f"Unexpected path in data bundle: {info.filename}")
        members.append(info)
    return members


def download_bundle(url: str = DATA_BUNDLE_URL, sha256: str = DATA_BUNDLE_SHA256) -> int:
    """Download and extract the data bundle into the project. Returns the number of files."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.zip"
        with urllib.request.urlopen(url, timeout=120) as response, open(path, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)

        if sha256:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != sha256.lower():
                raise ValueError(f"Data bundle checksum mismatch: {digest}")

        with zipfile.ZipFile(path) as archive:
            members = _safe_members(archive)
            archive.extractall(PROJECT_ROOT, members=members)
    return len(members)


def ensure_data() -> str:
    """Make sure the dashboard has data; download the bundle if it doesn't."""
    if data_is_present():
        return "present"
    count = download_bundle()
    if not data_is_present():
        raise RuntimeError("Data bundle extracted but the warehouse or models are still missing")
    return f"downloaded {count} files"
