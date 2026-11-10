"""
Forecast health alerts — flags series that need a planner's attention.

Thresholds follow data/docs/demand_planning_process.md:
- holdout WAPE above target (25% for series averaging 10+ units/day, else 60%)
- 95% interval coverage on the holdout below 85%
- for high-volume series, future intervals wider than 1.5x the forecast (for low-volume
  count data, intervals wider than the forecast are expected, so they aren't flagged)
"""

import numpy as np
import pandas as pd

from src.data_pipeline.feature_store import load_daily_units
from src.forecasting.predict import forecast_series, list_trained_series, load_series

HIGH_VOLUME_UNITS_PER_DAY = 10
WAPE_TARGET_HIGH_VOLUME = 0.25
WAPE_TARGET_LOW_VOLUME = 0.60
MIN_COVERAGE = 0.85
MAX_RELATIVE_WIDTH = 1.5


def series_health(level: str, name: str) -> dict:
    """Accuracy, interval and volume indicators for one trained series, plus alerts."""
    _, meta, _ = load_series(level, name)
    champion = meta["champion"]
    holdout = meta["holdout"][champion]

    units = load_daily_units(level, name, end_date=meta["model_end_date"])
    avg_units = float(units.iloc[-90:].mean())
    high_volume = avg_units >= HIGH_VOLUME_UNITS_PER_DAY
    wape_target = WAPE_TARGET_HIGH_VOLUME if high_volume else WAPE_TARGET_LOW_VOLUME

    future = forecast_series(level, name).query("segment == 'future'")
    width = (future["upper_bound"] - future["lower_bound"]).mean()
    relative_width = float(width / max(future["forecast"].mean(), 1e-9))

    alerts = []
    if holdout["wape"] > wape_target:
        alerts.append(f"WAPE {holdout['wape']:.0%} above {wape_target:.0%} target")
    if holdout["coverage"] < MIN_COVERAGE:
        alerts.append(f"interval coverage {holdout['coverage']:.0%} below {MIN_COVERAGE:.0%}")
    if high_volume and relative_width > MAX_RELATIVE_WIDTH:
        alerts.append(f"interval width {relative_width:.1f}x the forecast")

    return {
        "level": level,
        "series": name,
        "champion": champion,
        "avg_daily_units": round(avg_units, 1),
        "holdout_wape": round(holdout["wape"], 3),
        "wape_target": wape_target,
        "wape_vs_moving_average": round(
            holdout["wape"] - meta["holdout"]["moving_average"]["wape"], 3
        ),
        "coverage": round(holdout["coverage"], 2),
        "relative_interval_width": round(relative_width, 2),
        "alerts": "; ".join(alerts),
        "status": "alert" if alerts else "ok",
    }


def forecast_health() -> pd.DataFrame:
    """Health table for every trained series, alerts first."""
    trained = list_trained_series()
    if trained.empty:
        return pd.DataFrame()
    df = pd.DataFrame([series_health(r.level, r.name) for r in trained.itertuples()])
    df["_order"] = np.where(df["status"] == "alert", 0, 1)
    return (
        df.sort_values(["_order", "level", "holdout_wape"], ascending=[True, True, False])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
