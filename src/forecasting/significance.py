"""
Is one forecast really more accurate than another? Diebold–Mariano test.

For each day t we have the absolute error of two methods. The loss
differential d_t = |e_A,t| - |e_B,t| has mean 0 if the methods are equally
accurate. Multi-day forecasts made from the same origin share errors, so d_t
is autocorrelated; the variance of its mean uses a Newey–West (HAC) estimate
with lag = horizon - 1, and the statistic is compared with a t distribution
(Harvey, Leybourne & Newbold small-sample practice).

Negative statistic / mean difference => method A has the smaller errors.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def newey_west_variance(x: np.ndarray, lag: int) -> float:
    """Long-run variance of a series with Bartlett-weighted autocovariances up to `lag`."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    variance = np.dot(x, x) / n
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1 - k / (lag + 1)
        variance += 2 * weight * np.dot(x[k:], x[:-k]) / n
    return float(variance)


def diebold_mariano(errors_a, errors_b, lag: int, alpha: float = 0.05) -> dict:
    """
    Two-sided Diebold–Mariano test on absolute-error loss.

    Returns the statistic, p-value, mean loss differential (A - B, in units)
    and its (1 - alpha) confidence interval.
    """
    d = np.abs(np.asarray(errors_a, float)) - np.abs(np.asarray(errors_b, float))
    n = len(d)
    mean = float(d.mean())
    variance = newey_west_variance(d, lag)
    if n < 3 or variance <= 0:
        return {
            "n": n,
            "mean_diff": mean,
            "stat": 0.0,
            "p_value": 1.0,
            "ci_low": mean,
            "ci_high": mean,
        }

    se = np.sqrt(variance / n)
    stat = mean / se
    p_value = 2 * stats.t.sf(abs(stat), df=n - 1)
    half_width = stats.t.ppf(1 - alpha / 2, df=n - 1) * se
    return {
        "n": n,
        "mean_diff": mean,
        "stat": float(stat),
        "p_value": float(p_value),
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def compare_methods(
    forecasts: pd.DataFrame,
    method_a: str = "xgboost",
    method_b: str = "moving_average",
    lag: int = 27,
) -> dict:
    """
    Diebold–Mariano test between two methods from long-format forecasts.

    `forecasts` has columns order_date, method, actual, forecast (e.g. the
    rolling backtests plus the holdout). Days are paired by date.
    """
    wide = forecasts.pivot_table(index="order_date", columns="method", values="forecast")
    actual = forecasts.groupby("order_date")["actual"].first()
    wide = wide.join(actual).dropna(subset=[method_a, method_b, "actual"]).sort_index()
    return diebold_mariano(
        wide["actual"] - wide[method_a], wide["actual"] - wide[method_b], lag=lag
    )


def adjust_pvalues(p_values, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg false-discovery-rate adjustment across many series."""
    reject, adjusted, _, _ = multipletests(
        np.asarray(p_values, float), alpha=alpha, method="fdr_bh"
    )
    return reject, adjusted


def significance_table(alpha: float = 0.05) -> pd.DataFrame:
    """
    XGBoost vs. moving average for every trained series, with BH-adjusted p-values.

    mean_diff_pct expresses the mean absolute-error difference as a share of the
    series' average daily demand over the tested days, so series of different
    size are comparable. Negative = XGBoost has the smaller errors.
    """
    from src.forecasting.predict import _run_artifact_dir, list_trained_series, load_series

    rows = []
    for run in list_trained_series().itertuples():
        _, meta, holdout = load_series(run.level, run.name)
        result = meta.get("significance", {}).get("xgboost_vs_moving_average")
        if result is None:
            continue
        backtest = pd.read_csv(_run_artifact_dir(run.run_id) / "backtest.csv")
        tested = pd.concat([backtest, holdout])
        mean_actual = tested.groupby("order_date")["actual"].first().mean()
        scale = 100 / mean_actual if mean_actual > 0 else np.nan
        rows.append(
            {
                "level": run.level,
                "series": run.name,
                "champion": meta["champion"],
                "days_tested": result["n"],
                "avg_daily_units": mean_actual,
                "mean_diff_units": result["mean_diff"],
                "mean_diff_pct": result["mean_diff"] * scale,
                "ci_low_pct": result["ci_low"] * scale,
                "ci_high_pct": result["ci_high"] * scale,
                "dm_stat": result["stat"],
                "p_value": result["p_value"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    reject, adjusted = adjust_pvalues(df["p_value"], alpha=alpha)
    df["p_adjusted"] = adjusted
    df["verdict"] = np.select(
        [reject & (df["mean_diff_units"] < 0), reject & (df["mean_diff_units"] > 0)],
        ["XGBoost better", "Moving average better"],
        default="No significant difference",
    )
    return df.sort_values("mean_diff_pct").reset_index(drop=True)
