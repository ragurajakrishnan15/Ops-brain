"""
Prepare a fresh environment (used as the Docker entrypoint before the dashboard starts).

- Builds the warehouse and vector store if they don't exist (needs the raw CSV).
- Trains forecasting models if none are logged in MLflow.

Each step is skipped when its output already exists, so restarts are fast.

Usage:
    python scripts/bootstrap.py            # TRAIN_TRIALS env var sets Optuna trials (default 10)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import CHROMA_PERSIST_DIR, DUCKDB_PATH, RAW_CSV_PATH  # noqa: E402


def main() -> None:
    if not Path(DUCKDB_PATH).exists() or not Path(CHROMA_PERSIST_DIR).exists():
        if not RAW_CSV_PATH.exists():
            print(
                f"Raw dataset not found at {RAW_CSV_PATH}.\n"
                "Download DataCoSupplyChainDataset.csv from Kaggle into data/raw/ "
                "(mounted into the container at /app/data/raw). Starting the dashboard without data."
            )
            return
        print("Warehouse not found - running the data pipeline...")
        from scripts.setup_data import main as setup

        setup()

    from src.forecasting.predict import list_trained_series

    if list_trained_series().empty:
        trials = int(os.getenv("TRAIN_TRIALS", "10"))
        print(f"No trained models found - training all series ({trials} Optuna trials each)...")
        from src.forecasting.train import summarize, train_all

        print(summarize(train_all(n_optuna_trials=trials)).to_string(index=False))
    else:
        print("Warehouse and models found - starting the dashboard.")


if __name__ == "__main__":
    main()
