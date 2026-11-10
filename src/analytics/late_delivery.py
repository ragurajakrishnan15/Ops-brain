"""
What drives late deliveries? Statistical tests on the order lines.

1. Chi-square tests of independence between the late-delivery flag and each
   candidate driver, with Cramér's V as the effect size (0 = no association,
   1 = perfect). With 180k rows almost anything is "significant", so the
   effect size is what matters.
2. A logistic regression of late delivery on shipping mode, market, customer
   segment, department, quantity and discount, reported as odds ratios with
   95% confidence intervals.
3. A likelihood-ratio test of whether everything except shipping mode adds
   explanatory power, plus McFadden pseudo-R² for both models.

Every First Class order line is late (scheduled 1 day, shipped in 2), which
is perfect separation: its odds ratio is infinite and the regression can't
converge with it. First Class is reported as a deterministic rule and the
regression is fitted on the other shipping modes.

Canceled and suspected-fraud lines are excluded (their delivery status isn't
a real delivery outcome).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.utils.db import run_query_df

DRIVERS = {
    "shipping_mode": "Shipping mode",
    "market": "Market",
    "customer_segment": "Customer segment",
    "department_name": "Department",
    "category_name": "Category",
    "order_region": "Order region",
}

FULL_FORMULA = (
    "late ~ C(shipping_mode, Treatment('Standard Class')) + C(market) + C(customer_segment)"
    " + C(department_name) + quantity + discount_rate"
)
SHIPPING_ONLY_FORMULA = "late ~ C(shipping_mode, Treatment('Standard Class'))"


def load_order_lines() -> pd.DataFrame:
    return run_query_df(f"""
        SELECT late_delivery_risk AS late, {", ".join(DRIVERS)}, quantity, discount_rate,
               planned_ship_days, actual_ship_days
        FROM orders
        WHERE is_valid_demand
    """)


def cramers_v(table: pd.DataFrame, chi2: float) -> float:
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    """One row per driver: chi-square statistic, p-value, Cramér's V and late-rate range."""
    rows = []
    for column, label in DRIVERS.items():
        table = pd.crosstab(df[column], df["late"])
        chi2, p_value, dof, _ = stats.chi2_contingency(table)
        late_rate = df.groupby(column)["late"].mean() * 100
        rows.append(
            {
                "driver": label,
                "levels": table.shape[0],
                "chi2": round(float(chi2), 1),
                "dof": int(dof),
                "p_value": float(p_value),
                "cramers_v": round(cramers_v(table, chi2), 3),
                "late_pct_min": round(float(late_rate.min()), 1),
                "late_pct_max": round(float(late_rate.max()), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def _term_label(term: str) -> str:
    """'C(market)[T.LATAM]' -> 'Market: LATAM'."""
    if term.startswith("C("):
        variable = term[2:].split(",")[0].split(")")[0]
        level = term.split("[T.")[-1].rstrip("]")
        return f"{DRIVERS.get(variable, variable)}: {level}"
    return {"quantity": "Quantity (per unit)", "discount_rate": "Discount rate (per 1.0)"}.get(
        term, term
    )


def logistic_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fit the full and shipping-mode-only logistic models.

    Returns (odds-ratio table, model summary with pseudo-R² and the
    likelihood-ratio test of the extra drivers).
    """
    separated = df["shipping_mode"] == "First Class"
    first_class_late_pct = float(100 * df.loc[separated, "late"].mean())
    df = df[~separated]

    full = smf.logit(FULL_FORMULA, df).fit(disp=0, maxiter=200)
    shipping_only = smf.logit(SHIPPING_ONLY_FORMULA, df).fit(disp=0, maxiter=200)
    if not (full.mle_retvals["converged"] and shipping_only.mle_retvals["converged"]):
        raise RuntimeError("Late-delivery logistic regression did not converge")

    ci = full.conf_int()
    odds = pd.DataFrame(
        {
            "term": [_term_label(t) for t in full.params.index],
            "odds_ratio": np.exp(full.params.to_numpy()),
            "ci_low": np.exp(ci[0].to_numpy()),
            "ci_high": np.exp(ci[1].to_numpy()),
            "p_value": full.pvalues.to_numpy(),
        }
    )
    odds = odds[odds["term"] != "Intercept"].reset_index(drop=True)

    lr_stat = 2 * (full.llf - shipping_only.llf)
    lr_dof = int(full.df_model - shipping_only.df_model)
    summary = {
        "n": int(full.nobs),
        "excluded_first_class_lines": int(separated.sum()),
        "first_class_late_pct": first_class_late_pct,
        "pseudo_r2_full": float(full.prsquared),
        "pseudo_r2_shipping_only": float(shipping_only.prsquared),
        "lr_stat_extra_drivers": float(lr_stat),
        "lr_dof_extra_drivers": lr_dof,
        "lr_p_value_extra_drivers": float(stats.chi2.sf(lr_stat, lr_dof)),
    }
    return odds, summary


def schedule_vs_actual(df: pd.DataFrame) -> pd.DataFrame:
    """Planned vs. actual shipping days and late rate by shipping mode."""
    return (
        df.groupby("shipping_mode")
        .agg(
            order_lines=("late", "size"),
            planned_days=("planned_ship_days", "mean"),
            actual_days=("actual_ship_days", "mean"),
            late_pct=("late", lambda s: 100 * s.mean()),
        )
        .round(2)
        .sort_values("late_pct", ascending=False)
        .reset_index()
    )


def analyze() -> dict:
    """Run every analysis on the warehouse."""
    df = load_order_lines()
    odds, summary = logistic_regression(df)
    return {
        "chi_square": chi_square_tests(df),
        "odds_ratios": odds,
        "model": summary,
        "by_shipping_mode": schedule_vs_actual(df),
    }
