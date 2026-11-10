"""
End-to-end forecasting test on synthetic data.

Builds a 450-day DataCo-format CSV (weekly seasonality + Poisson noise),
runs the real pipeline into a temporary warehouse, then trains, logs to a
temporary MLflow store, forecasts, explains and health-checks the series.
Runs without the real dataset.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import feature_store, ingest, transforms
from src.forecasting import alerts, predict, train
from src.forecasting.features import FEATURE_COLS
from tests.test_pipeline_sql import _row

CATEGORY = "Kids' Golf Clubs"  # apostrophe on purpose
N_DAYS = 450
END = date(2017, 10, 2)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("forecasting")
    rng = np.random.default_rng(0)
    start = END - timedelta(days=N_DAYS - 1)

    rows, order_id = [], 1
    for i in range(N_DAYS):
        d = start + timedelta(days=i)
        units = int(rng.poisson(8 + 4 * (d.weekday() >= 5)))
        if units:
            day = (d - date(2017, 1, 1)).days  # _row() counts days from 2017-01-01
            rows.append(_row(order_id, order_id, day, CATEGORY, "Golf", units))
            order_id += 1
    csv_path = tmp / "dataco.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="latin-1")

    db_path = str(tmp / "warehouse.duckdb")
    mlflow_uri = (tmp / "mlruns").as_uri()
    mp = pytest.MonkeyPatch()
    for module in (ingest, transforms):
        mp.setattr(module, "DUCKDB_PATH", db_path)
    mp.setattr("src.utils.db.DUCKDB_PATH", db_path)
    mp.setattr(train, "MLFLOW_TRACKING_URI", mlflow_uri)
    mp.setattr(predict, "MLFLOW_TRACKING_URI", mlflow_uri)
    mp.setattr(train, "MODEL_END_DATE", END.isoformat())

    ingest.ingest_supply_chain(csv_path)
    ingest.create_clean_orders_table()
    transforms.build_daily_demand_table()
    transforms.build_feature_table()
    transforms.build_department_tables()
    predict.load_series.cache_clear()

    result = train.train_series("category", CATEGORY, n_optuna_trials=2)
    yield result
    predict.load_series.cache_clear()
    mp.undo()


def test_series_is_eligible(trained):
    assert CATEGORY in feature_store.get_series_names("category", end_date=END.isoformat())


def test_training_result(trained):
    assert trained["champion"] in train.METHODS
    assert set(trained["holdout"]) == set(train.METHODS)
    assert trained["holdout_start"] == (END - timedelta(days=27)).isoformat()
    assert len(trained["backtest_origins"]) == train.N_BACKTEST_FOLDS
    # Origins are 28 days apart and all before the holdout
    origins = pd.to_datetime(trained["backtest_origins"])
    assert (origins.diff().dropna() == pd.Timedelta(days=28)).all()
    assert origins.max() < pd.Timestamp(trained["holdout_start"])
    for widths in trained["interval_widths"].values():
        assert len(widths) == 28 and widths == sorted(widths)  # non-decreasing in horizon
    # Seasonal Poisson demand around 9 units/day: every method should be well under 100% WAPE
    assert all(m["wape"] < 0.6 for m in trained["holdout"].values())


def test_logged_run_and_forecast(trained):
    runs = predict.list_trained_series()
    assert runs["name"].tolist() == [CATEGORY]
    assert runs["champion"].iloc[0] == trained["champion"]

    df = predict.forecast_series("category", CATEGORY)
    assert set(df["segment"]) == {"history", "backtest", "future"}
    future = df[df["segment"] == "future"]
    assert len(future) == 28
    assert future["order_date"].min() == pd.Timestamp(END + timedelta(days=1))
    assert (future["lower_bound"] <= future["forecast"]).all()
    assert (future["forecast"] <= future["upper_bound"]).all()
    assert (future["lower_bound"] >= 0).all()

    metrics = predict.get_forecast_accuracy(df)
    assert metrics["wape"] == pytest.approx(
        trained["holdout"][trained["champion"]]["wape"], abs=1e-3
    )
    assert 0 <= metrics["coverage_95ci"] <= 1


def test_explanations_add_up(trained):
    from src.forecasting.explain import explain_prediction, get_global_importance

    model, _, _ = predict.load_series("category", CATEGORY)
    features = feature_store.load_features("category", CATEGORY).dropna(subset=FEATURE_COLS)
    importance = get_global_importance(model, features)
    assert set(importance["feature"]) == set(FEATURE_COLS)

    e = explain_prediction(model, features.tail(1), top_n=len(FEATURE_COLS))
    raw = e["baseline_units"] + e["base_value"] + sum(d["contribution"] for d in e["top_drivers"])
    assert e["predicted_value"] == pytest.approx(max(raw, 0), abs=1e-3)


def test_health_check(trained):
    health = alerts.series_health("category", CATEGORY)
    assert health["status"] in {"ok", "alert"}
    assert health["avg_daily_units"] > alerts.HIGH_VOLUME_UNITS_PER_DAY - 3
    assert health["wape_target"] in {alerts.WAPE_TARGET_HIGH_VOLUME, alerts.WAPE_TARGET_LOW_VOLUME}


def test_recursive_forecast_matches_one_step_on_first_day(trained):
    """Day 1 of a recursive forecast uses only real history, so it equals the one-step prediction."""
    from src.forecasting.features import predict_units, recursive_forecast

    model, _, _ = predict.load_series("category", CATEGORY)
    units = feature_store.load_daily_units("category", CATEGORY)
    features = feature_store.load_features("category", CATEGORY)
    features["order_date"] = pd.to_datetime(features["order_date"])
    origin = units.index[-10]

    recursive = recursive_forecast(model, units[units.index < origin], horizon=3)
    one_step = predict_units(model, features[features["order_date"] == origin])
    assert recursive.iloc[0] == pytest.approx(one_step[0])
