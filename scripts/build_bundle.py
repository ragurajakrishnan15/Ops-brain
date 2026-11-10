"""
Build the deployment data bundle: dist/opsbrain-data.zip.

Contains the DuckDB warehouse, the Chroma document store and the trained
MLflow models — everything the dashboard reads. Run after setup_data.py
and training, then attach the zip to the `data-v1` GitHub release:

    python scripts/build_bundle.py
    gh release upload data-v1 dist/opsbrain-data.zip --clobber
"""

import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import precompute  # noqa: E402
from src.utils.bundle import data_is_present  # noqa: E402
from src.utils.config import PROJECT_ROOT  # noqa: E402

INCLUDE = [
    "data/processed/warehouse.duckdb",
    "data/processed/chroma_store",
    "data/processed/precomputed",
    "mlruns",
]
SKIP_SUFFIXES = {".wal", ".tmp", ".lock"}


def main() -> None:
    if not data_is_present():
        sys.exit("Run scripts/setup_data.py and python -m src.forecasting.train first.")

    print("Precomputing dashboard tables:", ", ".join(precompute.write_all()))

    out = PROJECT_ROOT / "dist" / "opsbrain-data.zip"
    out.parent.mkdir(exist_ok=True)
    files = []
    for entry in INCLUDE:
        path = PROJECT_ROOT / entry
        files += [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    files = [f for f in files if f.suffix not in SKIP_SUFFIXES and "__pycache__" not in f.parts]

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for f in sorted(files):
            archive.write(f, f.relative_to(PROJECT_ROOT).as_posix())

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files)")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
