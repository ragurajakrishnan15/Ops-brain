"""
Demand forecasting: XGBoost vs. baselines, with walk-forward validation,
Optuna tuning, conformal-style intervals and MLflow tracking.

For each series (a category or a department) with enough history:

1. Hold out the last FORECAST_HORIZON_DAYS days. Nothing below sees them
   until step 4.
2. Tune XGBoost with Optuna on one-step-ahead walk-forward splits.
3. Rolling-origin backtests on the pre-holdout data: from several origins,
   forecast the next horizon days recursively with XGBoost and with two
   baselines (28-day moving average, same weekday last week). The method with
   the lowest backtest WAPE becomes the series' champion, and the absolute
   errors at each horizon set the 95% interval widths.
4. Score every method once on the holdout (point accuracy and interval coverage).
5. Refit XGBoost on all data up to MODEL_END_DATE and log everything to MLflow.

The XGBoost model predicts the deviation from the trailing 28-day average
(see features.py), so where demand has no structure beyond its level it
falls back toward the moving average instead of overfitting.

Usage:
    python -m src.forecasting.train                     # all eligible series
    python -m src.forecasting.train --level department --trials 10
    python -m src.forecasting.train --series "Cleats"
"""

import argparse

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from src.data_pipeline.feature_store import (
    SERIES_LEVELS,
    get_series_names,
    load_daily_units,
    load_features,
)
from src.forecasting.features import (
    BASELINE_COL,
    FEATURE_COLS,
    TARGET_COL,
    moving_average_forecast,
    predict_units,
    recursive_forecast,
    seasonal_naive_forecast,
)
from src.utils.config import (
    FORECAST_HORIZON_DAYS,
    MIN_HISTORY_DAYS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_END_DATE,
)

__all__ = ["FEATURE_COLS", "TARGET_COL", "train_series", "train_all", "walk_forward_split"]

METHODS = ("xgboost", "moving_average", "seasonal_naive")
INTERVAL_LEVEL = 0.95
N_BACKTEST_FOLDS = 6
MIN_TRAIN_ROWS = 180  # minimum training rows for a backtest origin

optuna.logging.set_verbosity(optuna.logging.WARNING)


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_days: int = 14,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits for time-series cross-validation.

    Each split trains on all data up to a cutoff and tests on the next
    `test_days` dates. Cutoffs are evenly spaced over the second half of
    the data, so every split has at least half the history to train on.
    """
    dates = sorted(df["order_date"].unique())
    total_days = len(dates)

    first_cutoff_idx = int(total_days * 0.5)
    step = (total_days - first_cutoff_idx - test_days) // n_splits

    splits = []
    for i in range(n_splits):
        cutoff_idx = first_cutoff_idx + i * step
        cutoff_date = dates[cutoff_idx]
        test_end_idx = min(cutoff_idx + test_days, total_days - 1)
        test_end_date = dates[test_end_idx]

        train = df[df["order_date"] <= cutoff_date]
        test = df[(df["order_date"] > cutoff_date) & (df["order_date"] <= test_end_date)]

        if len(test) > 0:
            splits.append((train, test))

    return splits


def wape(actual, predicted) -> float:
    """Weighted absolute percentage error: sum|error| / sum(actual)."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    total = actual.sum()
    abs_error = np.abs(actual - predicted).sum()
    return float(abs_error / total) if total > 0 else float(abs_error / len(actual))


def fit_model(train_df: pd.DataFrame, params: dict) -> XGBRegressor:
    """Fit XGBoost on the deviation of demand from its trailing 28-day average."""
    model = XGBRegressor(
        objective="reg:absoluteerror", random_state=42, verbosity=0, n_jobs=4, **params
    )
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL] - train_df[BASELINE_COL])
    return model


def objective(trial: optuna.Trial, df: pd.DataFrame) -> float:
    """Optuna objective: minimize one-step-ahead WAPE across walk-forward splits."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }

    scores = []
    for train_df, test_df in walk_forward_split(df, n_splits=4, test_days=FORECAST_HORIZON_DAYS):
        model = fit_model(train_df, params)
        scores.append(wape(test_df[TARGET_COL], predict_units(model, test_df)))
    return float(np.mean(scores))


def forecast_all_methods(model, history: pd.Series, horizon: int) -> dict[str, pd.Series]:
    """Forecast `horizon` days past the end of `history` with every method."""
    return {
        "xgboost": recursive_forecast(model, history, horizon),
        "moving_average": moving_average_forecast(history, horizon),
        "seasonal_naive": seasonal_naive_forecast(history, horizon),
    }


def rolling_backtest(
    features: pd.DataFrame,
    units: pd.Series,
    params: dict,
    origins: list[pd.Timestamp],
    horizon: int,
) -> pd.DataFrame:
    """
    From each origin, train on earlier data and forecast the next `horizon` days.

    Returns one row per (origin, method, horizon step) with actual and forecast.
    """
    rows = []
    for origin in origins:
        model = fit_model(features[features["order_date"] < origin], params)
        history = units[units.index < origin]
        actual = units[origin : origin + pd.Timedelta(days=horizon - 1)]
        for method, forecast in forecast_all_methods(model, history, horizon).items():
            rows.append(
                pd.DataFrame(
                    {
                        "origin": origin,
                        "method": method,
                        "order_date": actual.index,
                        "h": np.arange(1, len(actual) + 1),
                        "actual": actual.to_numpy(),
                        "forecast": forecast.to_numpy()[: len(actual)],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def interval_widths(cv: pd.DataFrame, horizon: int, window: int = 3) -> dict[str, list[float]]:
    """
    Per-method, per-horizon interval half-widths from backtest errors.

    The width at horizon h is the INTERVAL_LEVEL quantile of absolute errors at
    horizons h-window..h+window across all folds, made non-decreasing in h.
    """
    widths = {}
    for method, group in cv.groupby("method"):
        abs_err = np.abs(group["actual"] - group["forecast"])
        by_h = []
        for h in range(1, horizon + 1):
            near = abs_err[(group["h"] - h).abs() <= window]
            by_h.append(float(np.quantile(near, INTERVAL_LEVEL)))
        widths[method] = np.maximum.accumulate(by_h).round(3).tolist()
    return widths


def score(frame: pd.DataFrame, widths: list[float]) -> dict:
    """Point and interval metrics for one method's holdout forecast."""
    w = np.asarray(widths)[frame["h"].to_numpy() - 1]
    lower = np.clip(frame["forecast"] - w, 0, None)
    upper = frame["forecast"] + w
    actual, forecast = frame["actual"], frame["forecast"]
    return {
        "mae": float(np.mean(np.abs(actual - forecast))),
        "wape": wape(actual, forecast),
        "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
    }


def train_series(
    level: str,
    name: str,
    n_optuna_trials: int = 20,
    horizon: int = FORECAST_HORIZON_DAYS,
    log_mlflow: bool = True,
) -> dict:
    """
    Train, evaluate and log the forecaster for one series. See module docstring.

    Returns a dict with the final model, champion method and metrics,
    or {} if the series has too little history.
    """
    features = load_features(level, name, end_date=MODEL_END_DATE).dropna(subset=FEATURE_COLS)
    features["order_date"] = pd.to_datetime(features["order_date"])
    if len(features) < MIN_HISTORY_DAYS:
        print(f"Skipping {level} {name!r}: only {len(features)} feature rows")
        return {}

    units = load_daily_units(level, name, end_date=MODEL_END_DATE)
    holdout_start = units.index[-horizon]
    dev = features[features["order_date"] < holdout_start]

    # 1-2. Tune on pre-holdout data only
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, dev), n_trials=n_optuna_trials)
    params = study.best_params

    # 3. Rolling-origin backtests -> champion + interval widths
    origins = [
        holdout_start - pd.Timedelta(days=horizon * k) for k in range(N_BACKTEST_FOLDS, 0, -1)
    ]
    origins = [o for o in origins if (dev["order_date"] < o).sum() >= MIN_TRAIN_ROWS]
    cv = rolling_backtest(dev, units[units.index < holdout_start], params, origins, horizon)
    cv_wape = {m: wape(g["actual"], g["forecast"]) for m, g in cv.groupby("method")}
    champion = min(METHODS, key=cv_wape.get)
    widths = interval_widths(cv, horizon)

    # 4. Score every method once on the holdout
    holdout = rolling_backtest(features, units, params, [holdout_start], horizon)
    holdout_metrics = {m: score(g, widths[m]) for m, g in holdout.groupby("method")}

    # 5. Final model on all data up to MODEL_END_DATE
    final_model = fit_model(features, params)

    h = holdout_metrics
    print(
        f"{level:>10} {name:<24} champion={champion:<15} "
        f"holdout WAPE xgb={h['xgboost']['wape']:.3f} ma28={h['moving_average']['wape']:.3f} "
        f"naive7={h['seasonal_naive']['wape']:.3f} | coverage={h[champion]['coverage']:.2f}"
    )

    meta = {
        "series_level": level,
        "series_name": name,
        "champion": champion,
        "horizon": horizon,
        "interval_level": INTERVAL_LEVEL,
        "model_end_date": MODEL_END_DATE,
        "holdout_start": str(holdout_start.date()),
        "backtest_origins": [str(o.date()) for o in origins],
        "cv_wape": cv_wape,
        "holdout": holdout_metrics,
        "interval_widths": widths,
    }

    if log_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"{level}:{name}"):
            mlflow.set_tags({"series_level": level, "series_name": name, "champion": champion})
            mlflow.log_params(params)
            mlflow.log_params(
                {
                    "horizon": horizon,
                    "model_end_date": MODEL_END_DATE,
                    "optuna_trials": n_optuna_trials,
                    "backtest_folds": len(origins),
                }
            )
            mlflow.log_metric("train_rows", len(features))
            mlflow.log_metric("optuna_best_wape_1step", study.best_value)
            for method in METHODS:
                mlflow.log_metric(f"cv_wape_{method}", cv_wape[method])
                for key, value in holdout_metrics[method].items():
                    mlflow.log_metric(f"holdout_{key}_{method}", value)
            for key, value in holdout_metrics[champion].items():
                mlflow.log_metric(f"holdout_{key}", value)
            mlflow.log_dict(meta, "forecast_meta.json")
            mlflow.log_text(holdout.to_csv(index=False), "holdout.csv")
            mlflow.xgboost.log_model(final_model, artifact_path="model")

    return {"model": final_model, **meta, "best_params": params}


def train_all(
    levels: tuple[str, ...] = ("category", "department"),
    n_optuna_trials: int = 20,
) -> list[dict]:
    """Train models for every eligible series at the given levels."""
    results = []
    for level in levels:
        names = get_series_names(level)
        print(f"\n{'=' * 60}\nTraining {len(names)} {level} series\n{'=' * 60}")
        for name in names:
            result = train_series(level, name, n_optuna_trials=n_optuna_trials)
            if result:
                results.append(result)
    return results


def summarize(results: list[dict]) -> pd.DataFrame:
    """One row per series: champion and holdout WAPE for each method."""
    return pd.DataFrame(
        [
            {
                "level": r["series_level"],
                "series": r["series_name"],
                "champion": r["champion"],
                **{f"wape_{m}": round(r["holdout"][m]["wape"], 3) for m in METHODS},
                "coverage": round(r["holdout"][r["champion"]]["coverage"], 2),
            }
            for r in results
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--level",
        choices=list(SERIES_LEVELS),
        action="append",
        help="series level(s) to train (default: all)",
    )
    parser.add_argument("--series", help="train a single series by name")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials per series")
    args = parser.parse_args()

    if args.series:
        level = (args.level or ["category"])[0]
        train_series(level, args.series, n_optuna_trials=args.trials)
    else:
        all_results = train_all(tuple(args.level or SERIES_LEVELS), n_optuna_trials=args.trials)
        print("\n" + summarize(all_results).to_string(index=False))
