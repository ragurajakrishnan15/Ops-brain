"""
Generate demand forecasts using trained models.

Loads the latest MLflow run for a series and produces, for the dashboard:
- recent history (actuals only),
- the out-of-sample backtest on the holdout window (actual vs. forecast),
- a future forecast for the next `horizon` days after MODEL_END_DATE,
all from the series' champion method, with 95% intervals calibrated on
rolling backtests (see train.py).
"""

from functools import lru_cache
from io import StringIO

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from src.data_pipeline.feature_store import load_daily_units
from src.forecasting.features import (
    moving_average_forecast,
    recursive_forecast,
    seasonal_naive_forecast,
)
from src.utils.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MODEL_END_DATE


def list_trained_series(level: str | None = None) -> pd.DataFrame:
    """
    Latest finished run per series, with its champion and holdout metrics.

    Filtering by name happens in pandas, so names with quotes are safe.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    columns = [
        "level",
        "name",
        "champion",
        "holdout_wape",
        "holdout_wape_moving_average",
        "holdout_coverage",
        "run_id",
    ]
    if runs.empty or "tags.series_level" not in runs:
        return pd.DataFrame(columns=columns)

    runs = runs.drop_duplicates(subset=["tags.series_level", "tags.series_name"], keep="first")
    if level is not None:
        runs = runs[runs["tags.series_level"] == level]

    return (
        pd.DataFrame(
            {
                "level": runs["tags.series_level"],
                "name": runs["tags.series_name"],
                "champion": runs["tags.champion"],
                "holdout_wape": runs.get("metrics.holdout_wape"),
                "holdout_wape_moving_average": runs.get("metrics.holdout_wape_moving_average"),
                "holdout_coverage": runs.get("metrics.holdout_coverage"),
                "run_id": runs["run_id"],
            }
        )
        .sort_values(["level", "name"])
        .reset_index(drop=True)
    )


@lru_cache(maxsize=64)
def load_series(level: str, name: str, run_id: str | None = None) -> tuple:
    """Return (model, meta, holdout DataFrame) for a series' latest run (or `run_id`)."""
    if run_id is None:
        runs = list_trained_series(level)
        match = runs[runs["name"] == name]
        if match.empty:
            raise ValueError(
                f"No trained model found for {level} {name!r}. Run: python -m src.forecasting.train"
            )
        run_id = match.iloc[0]["run_id"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    meta = mlflow.artifacts.load_dict(f"runs:/{run_id}/forecast_meta.json")
    holdout = pd.read_csv(
        StringIO(mlflow.artifacts.load_text(f"runs:/{run_id}/holdout.csv")),
        parse_dates=["order_date"],
    )
    return model, meta, holdout


def _with_bounds(df: pd.DataFrame, widths: list[float]) -> pd.DataFrame:
    w = np.asarray(widths)[df["h"].to_numpy() - 1]
    df["lower_bound"] = np.clip(df["forecast"] - w, 0, None)
    df["upper_bound"] = df["forecast"] + w
    return df


def forecast_series(
    level: str,
    name: str,
    history_days: int = 120,
) -> pd.DataFrame:
    """
    Forecast a series with its champion method.

    Returns one row per date with columns: order_date, segment
    ('history' | 'backtest' | 'future'), actual, forecast, lower_bound,
    upper_bound, residual, method.
    """
    model, meta, holdout = load_series(level, name)
    champion, horizon = meta["champion"], meta["horizon"]
    widths = meta["interval_widths"][champion]

    units = load_daily_units(level, name, end_date=meta.get("model_end_date", MODEL_END_DATE))
    holdout_start = pd.Timestamp(meta["holdout_start"])

    history = units[units.index < holdout_start].iloc[-history_days:]
    history_df = pd.DataFrame(
        {
            "order_date": history.index,
            "segment": "history",
            "actual": history.to_numpy(),
        }
    )

    backtest = holdout[holdout["method"] == champion].copy()
    backtest = _with_bounds(backtest, widths)
    backtest["segment"] = "backtest"

    if champion == "xgboost":
        future = recursive_forecast(model, units, horizon)
    elif champion == "moving_average":
        future = moving_average_forecast(units, horizon)
    else:
        future = seasonal_naive_forecast(units, horizon)
    future_df = pd.DataFrame(
        {
            "order_date": future.index,
            "segment": "future",
            "forecast": future.to_numpy(),
            "h": np.arange(1, horizon + 1),
        }
    )
    future_df = _with_bounds(future_df, widths)

    out = pd.concat([history_df, backtest, future_df], ignore_index=True)
    out["residual"] = out["actual"] - out["forecast"]
    out["method"] = champion
    out["series"] = name
    return out[
        [
            "order_date",
            "segment",
            "actual",
            "forecast",
            "lower_bound",
            "upper_bound",
            "residual",
            "method",
            "series",
        ]
    ]


def forecast_category(category: str, **kwargs) -> pd.DataFrame:
    """Forecast a product category (see forecast_series)."""
    return forecast_series("category", category, **kwargs)


def get_forecast_accuracy(forecast_df: pd.DataFrame) -> dict:
    """Accuracy metrics on rows that have both an actual and a forecast (the backtest)."""
    scored = forecast_df.dropna(subset=["actual", "forecast"])
    actual = scored["actual"].to_numpy(float)
    predicted = scored["forecast"].to_numpy(float)
    if len(scored) == 0:
        return {
            "mae": np.nan,
            "mape": np.nan,
            "wape": np.nan,
            "rmse": np.nan,
            "coverage_95ci": np.nan,
        }

    nonzero = actual > 0
    in_bounds = (actual >= scored["lower_bound"]) & (actual <= scored["upper_bound"])
    return {
        "mae": round(float(np.mean(np.abs(actual - predicted))), 2),
        # MAPE is undefined on zero-demand days, so it only uses days with demand
        "mape": round(
            float(np.mean(np.abs(actual[nonzero] - predicted[nonzero]) / actual[nonzero])), 4
        )
        if nonzero.any()
        else np.nan,
        "wape": round(float(np.abs(actual - predicted).sum() / max(actual.sum(), 1e-9)), 4),
        "rmse": round(float(np.sqrt(np.mean((actual - predicted) ** 2))), 2),
        "coverage_95ci": round(float(in_bounds.mean()), 3),
    }
