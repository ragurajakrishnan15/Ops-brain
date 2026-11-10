"""
Precomputed dashboard tables, shipped in the data bundle.

Forecast health (30 forecasts), the significance tests (30 series of
backtests) and the late-delivery models take ~20 s to compute, which a
cold-started cloud app would otherwise pay on its first page view. They are
computed once by scripts/build_bundle.py and stored as Parquet/JSON (never
pickle: the bundle is downloaded, and unpickling runs code).

Each file records a fingerprint of its inputs — the latest MLflow run ids or
the order count — and is ignored when it no longer matches.
"""

import json
from collections.abc import Callable

import pandas as pd

from src.utils.config import PROCESSED_DIR
from src.utils.db import run_query_df

PRECOMPUTED_DIR = PROCESSED_DIR / "precomputed"


def models_fingerprint() -> str:
    from src.forecasting.predict import list_trained_series

    return ",".join(sorted(list_trained_series()["run_id"]))


def data_fingerprint() -> str:
    row = run_query_df("SELECT COUNT(*) AS n, MAX(order_date) AS last FROM orders").iloc[0]
    return f"{row['n']}:{row['last']}"


def _read_fingerprint(name: str) -> str | None:
    path = PRECOMPUTED_DIR / f"{name}.fingerprint"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _write_fingerprint(name: str, fingerprint: str) -> None:
    (PRECOMPUTED_DIR / f"{name}.fingerprint").write_text(fingerprint, encoding="utf-8")


def table(
    name: str, compute: Callable[[], pd.DataFrame], fingerprint: Callable[[], str]
) -> pd.DataFrame:
    """Load a precomputed DataFrame if its fingerprint matches, else compute it."""
    path = PRECOMPUTED_DIR / f"{name}.parquet"
    if path.exists() and _read_fingerprint(name) == fingerprint():
        return pd.read_parquet(path)
    return compute()


def late_delivery() -> dict:
    """Late-delivery analysis: precomputed if current, otherwise fitted now."""
    from src.analytics.late_delivery import analyze

    folder = PRECOMPUTED_DIR / "late_delivery"
    if folder.exists() and _read_fingerprint("late_delivery") == data_fingerprint():
        result = {
            k: pd.read_parquet(folder / f"{k}.parquet")
            for k in ("chi_square", "odds_ratios", "by_shipping_mode")
        }
        result["model"] = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        return result
    return analyze()


def write_all() -> list[str]:
    """Compute and store every precomputed table. Returns the files written."""
    from src.analytics.late_delivery import analyze
    from src.forecasting.alerts import forecast_health
    from src.forecasting.significance import significance_table

    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    models = models_fingerprint()
    for name, compute in (
        ("forecast_health", forecast_health),
        ("significance", significance_table),
    ):
        compute().to_parquet(PRECOMPUTED_DIR / f"{name}.parquet", index=False)
        _write_fingerprint(name, models)
        written.append(f"{name}.parquet")

    folder = PRECOMPUTED_DIR / "late_delivery"
    folder.mkdir(exist_ok=True)
    result = analyze()
    for key in ("chi_square", "odds_ratios", "by_shipping_mode"):
        result[key].to_parquet(folder / f"{key}.parquet", index=False)
    (folder / "model.json").write_text(json.dumps(result["model"], indent=1), encoding="utf-8")
    _write_fingerprint("late_delivery", data_fingerprint())
    written.append("late_delivery/")
    return written
