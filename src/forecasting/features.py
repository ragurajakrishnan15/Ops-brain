"""
Python mirror of the SQL feature definitions, plus recursive forecasting.

Training uses the SQL feature tables (one-step-ahead rows). Forecasting
several days ahead needs features for days whose demand is not known yet,
so they are rebuilt here from history extended with earlier predictions.
Definitions must match _feature_table_sql() in data_pipeline/transforms.py
(tests/test_pipeline_sql.py checks parity).

The model predicts the deviation from the trailing 28-day average
(BASELINE_COL), so a forecast is that average plus the model's correction.
"""

import numpy as np
import pandas as pd

# Feature columns the model uses (target is 'total_units').
# All are known before the day being predicted and depend only on past units,
# so they can be rebuilt exactly during recursive forecasting.
FEATURE_COLS = [
    "day_of_week",
    "month",
    "quarter",
    "is_weekend",
    "lag_1d_units",
    "lag_7d_units",
    "lag_14d_units",
    "lag_28d_units",
    "rolling_7d_avg_units",
    "rolling_14d_avg_units",
    "rolling_28d_avg_units",
    "rolling_14d_std_units",
]

TARGET_COL = "total_units"

# The model learns total_units - BASELINE_COL
BASELINE_COL = "rolling_28d_avg_units"

MIN_HISTORY = 28  # days of history needed to compute every feature


def features_for_day(history: np.ndarray, day: pd.Timestamp) -> dict:
    """
    Compute the model features for `day` from the units on the days before it.

    `history` holds daily units in date order, ending the day before `day`.
    """
    if len(history) < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} days of history, got {len(history)}")

    dow = (day.dayofweek + 1) % 7  # DuckDB DOW: Sunday = 0 ... Saturday = 6
    return {
        "day_of_week": dow,
        "month": day.month,
        "quarter": day.quarter,
        "is_weekend": int(dow in (0, 6)),
        "lag_1d_units": history[-1],
        "lag_7d_units": history[-7],
        "lag_14d_units": history[-14],
        "lag_28d_units": history[-28],
        "rolling_7d_avg_units": history[-7:].mean(),
        "rolling_14d_avg_units": history[-14:].mean(),
        "rolling_28d_avg_units": history[-28:].mean(),
        "rolling_14d_std_units": history[-14:].std(ddof=1),  # DuckDB STDDEV = sample std
    }


def predict_units(model, X: pd.DataFrame) -> np.ndarray:
    """Turn the model's predicted deviation into units (never negative)."""
    return np.clip(X[BASELINE_COL].to_numpy() + model.predict(X[FEATURE_COLS]), 0, None)


def recursive_forecast(model, history: pd.Series, horizon: int) -> pd.Series:
    """
    Forecast `horizon` days after the end of `history` (daily units indexed by date).

    Each day's prediction is appended to the history and used to build the
    next day's features.
    """
    values = history.to_numpy(dtype=float)
    start = history.index[-1] + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=horizon, freq="D")

    preds = []
    for day in dates:
        X = pd.DataFrame([features_for_day(values, day)])
        pred = float(predict_units(model, X)[0])
        preds.append(pred)
        values = np.append(values, pred)
    return pd.Series(preds, index=dates, name="forecast")


def moving_average_forecast(history: pd.Series, horizon: int, window: int = 28) -> pd.Series:
    """Baseline: repeat the trailing `window`-day average for every future day."""
    start = history.index[-1] + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=horizon, freq="D")
    return pd.Series(float(history.iloc[-window:].mean()), index=dates, name="forecast")


def seasonal_naive_forecast(history: pd.Series, horizon: int, season: int = 7) -> pd.Series:
    """Baseline: repeat the last `season` days (same weekday last week)."""
    start = history.index[-1] + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=horizon, freq="D")
    last = history.to_numpy(dtype=float)[-season:]
    return pd.Series(np.resize(last, horizon), index=dates, name="forecast")
