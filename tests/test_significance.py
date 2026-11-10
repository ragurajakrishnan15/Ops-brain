"""Tests for the Diebold–Mariano test and multiple-testing adjustment."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.significance import (
    adjust_pvalues,
    compare_methods,
    diebold_mariano,
    newey_west_variance,
)


def test_newey_west_equals_sample_variance_without_lags():
    x = np.random.default_rng(0).normal(size=200)
    assert newey_west_variance(x, lag=0) == pytest.approx(np.var(x))


def test_newey_west_larger_for_positively_autocorrelated_series():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=2000)
    ar = np.zeros_like(noise)
    for t in range(1, len(ar)):
        ar[t] = 0.8 * ar[t - 1] + noise[t]
    assert newey_west_variance(ar, lag=20) > 2 * newey_west_variance(ar, lag=0)


def test_identical_forecasts_are_not_different():
    e = np.random.default_rng(2).normal(size=100)
    result = diebold_mariano(e, e, lag=5)
    assert result["p_value"] == 1.0 and result["mean_diff"] == 0


def test_clearly_better_forecast_is_significant_and_signed():
    rng = np.random.default_rng(3)
    good = rng.normal(0, 1, size=300)
    bad = rng.normal(0, 3, size=300)
    result = diebold_mariano(good, bad, lag=3)
    assert result["stat"] < 0 and result["mean_diff"] < 0
    assert result["p_value"] < 0.001
    assert result["ci_low"] < result["mean_diff"] < result["ci_high"] < 0


def test_equally_good_forecasts_rarely_rejected():
    rng = np.random.default_rng(4)
    p_values = [
        diebold_mariano(rng.normal(size=200), rng.normal(size=200), lag=3)["p_value"]
        for _ in range(400)
    ]
    # Under the null, about 5% of tests should reject at the 5% level
    assert 0.02 < np.mean(np.array(p_values) < 0.05) < 0.09


def test_compare_methods_pairs_by_date():
    dates = pd.date_range("2017-01-01", periods=60)
    actual = np.arange(60, dtype=float)
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "order_date": dates,
                    "method": "xgboost",
                    "actual": actual,
                    "forecast": actual + 0.5,
                }
            ),
            pd.DataFrame(
                {
                    "order_date": dates,
                    "method": "moving_average",
                    "actual": actual,
                    "forecast": actual + 2.0,
                }
            ),
            # shuffled rows must not matter
        ]
    ).sample(frac=1, random_state=0)
    result = compare_methods(long, lag=5)
    assert result["n"] == 60 and result["mean_diff"] == pytest.approx(-1.5)


def test_benjamini_hochberg():
    # BH: sorted p_i * m / i, made monotone -> [0.004, 0.04, 0.0533, 0.5]
    reject, adjusted = adjust_pvalues([0.001, 0.02, 0.04, 0.5])
    assert adjusted == pytest.approx([0.004, 0.04, 0.04 * 4 / 3, 0.5])
    assert reject.tolist() == [True, True, False, False]
