"""Statistical Tests — is the ML model really better, and what drives late deliveries?"""

import streamlit as st

from src.dashboard import data
from src.dashboard.components import COLORS, interval_plot

st.header("Statistical Tests")
st.caption(
    "Two questions answered with hypothesis tests instead of eyeballing charts: does XGBoost "
    "really beat a 28-day moving average, and what actually drives late deliveries?"
)

# --- 1. Forecast accuracy ----------------------------------------------------------------
st.subheader("1. Is XGBoost more accurate than a 28-day moving average?")
st.markdown(
    "For each series, the **Diebold–Mariano test** compares the two methods' daily absolute "
    "errors over 196 consecutive days (six rolling 28-day backtests plus the 28-day holdout). "
    "Errors from the same forecast origin are correlated, so the test uses a Newey–West (HAC) "
    "variance with lag 27. With 30 series tested, p-values are adjusted with "
    "**Benjamini–Hochberg** to control the false-discovery rate at 5%."
)

try:
    with st.spinner("Loading test results for every series..."):
        sig = data.significance_table()
except Exception as e:
    st.warning(data.TRAIN_HINT)
    st.error(f"Error: {e}")
    sig = None

if sig is not None and not sig.empty:
    counts = sig["verdict"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Series tested", len(sig))
    c2.metric("XGBoost significantly better", int(counts.get("XGBoost better", 0)))
    c3.metric("Moving average significantly better", int(counts.get("Moving average better", 0)))
    c4.metric("No significant difference", int(counts.get("No significant difference", 0)))

    verdict_colors = {
        "XGBoost better": COLORS["success"],
        "Moving average better": COLORS["danger"],
        "No significant difference": COLORS["gray"],
    }
    st.plotly_chart(
        interval_plot(
            [
                f"{s} ({'dept' if lvl == 'department' else 'cat'})"
                for s, lvl in zip(sig["series"], sig["level"])
            ],
            sig["mean_diff_pct"],
            sig["ci_low_pct"],
            sig["ci_high_pct"],
            title="Mean absolute-error difference, XGBoost − moving average (95% CI)",
            x_title="% of average daily demand (negative = XGBoost better)",
            colors=[verdict_colors[v] for v in sig["verdict"]],
        ),
        use_container_width=True,
    )
    st.caption(
        "Green = XGBoost significantly better, red = moving average significantly better, grey = "
        "no significant difference after Benjamini–Hochberg adjustment. Intervals are unadjusted 95% CIs."
    )
    with st.expander("Per-series test results"):
        st.dataframe(
            sig[
                [
                    "level",
                    "series",
                    "champion",
                    "avg_daily_units",
                    "mean_diff_units",
                    "mean_diff_pct",
                    "dm_stat",
                    "p_value",
                    "p_adjusted",
                    "verdict",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "level": "Level",
                "series": "Series",
                "champion": "Champion",
                "avg_daily_units": st.column_config.NumberColumn("Avg units/day", format="%.1f"),
                "mean_diff_units": st.column_config.NumberColumn(
                    "Error diff (units/day)", format="%.3f"
                ),
                "mean_diff_pct": st.column_config.NumberColumn(
                    "Error diff (% of demand)", format="%.2f"
                ),
                "dm_stat": st.column_config.NumberColumn("DM statistic", format="%.2f"),
                "p_value": st.column_config.NumberColumn("p-value", format="%.3f"),
                "p_adjusted": st.column_config.NumberColumn("BH-adjusted p", format="%.3f"),
                "verdict": "Verdict",
            },
        )

# --- 2. Late deliveries -------------------------------------------------------------------
st.subheader("2. What drives late deliveries?")
try:
    with st.spinner("Fitting the late-delivery models..."):
        late = data.late_delivery_analysis()
except Exception as e:
    st.warning(data.SETUP_HINT)
    st.error(f"Error: {e}")
    st.stop()

model = late["model"]
m1, m2, m3 = st.columns(3)
m1.metric("Order lines analysed", f"{model['n'] + model['excluded_first_class_lines']:,}")
m2.metric(
    "Drivers beyond shipping mode",
    f"p = {model['lr_p_value_extra_drivers']:.2f}",
    help=f"Likelihood-ratio test of the {model['lr_dof_extra_drivers']} extra terms (market, "
    "segment, department, quantity, discount) on top of shipping mode. A high p-value means "
    "they add no explanatory power.",
)
m3.metric(
    "Pseudo-R²: shipping mode → full model",
    f"{model['pseudo_r2_shipping_only']:.4f} → {model['pseudo_r2_full']:.4f}",
    help="McFadden pseudo-R², fitted without First Class (see below)",
)

st.markdown("**Chi-square tests of independence** (late flag vs. each driver)")
st.dataframe(
    late["chi_square"],
    use_container_width=True,
    hide_index=True,
    column_config={
        "driver": "Driver",
        "levels": "Levels",
        "chi2": st.column_config.NumberColumn("χ²", format="%.1f"),
        "dof": "df",
        "p_value": st.column_config.NumberColumn("p-value", format="%.2e"),
        "cramers_v": st.column_config.ProgressColumn(
            "Cramér's V (effect size)", min_value=0.0, max_value=1.0, format="%.3f"
        ),
        "late_pct_min": st.column_config.NumberColumn("Lowest late %", format="%.1f"),
        "late_pct_max": st.column_config.NumberColumn("Highest late %", format="%.1f"),
    },
)
st.caption(
    "With this many order lines, even tiny differences reach p < 0.05 (order region does), so the "
    "effect size is what matters: only shipping mode has a meaningful association."
)

left, right = st.columns([3, 2])
with left:
    odds = late["odds_ratios"].sort_values("odds_ratio", ascending=False)
    st.plotly_chart(
        interval_plot(
            odds["term"].tolist(),
            odds["odds_ratio"],
            odds["ci_low"],
            odds["ci_high"],
            title="Logistic regression: odds ratio of a late delivery (95% CI)",
            x_title="Odds ratio vs. reference level (log scale; 1 = no effect)",
            reference=1.0,
            log_x=True,
            colors=[COLORS["danger"] if p < 0.05 else COLORS["gray"] for p in odds["p_value"]],
        ),
        use_container_width=True,
    )
    st.caption(
        "Reference levels: Standard Class, Africa, Consumer, Apparel. Red = p < 0.05. First Class "
        f"is left out of the fit: all {model['excluded_first_class_lines']:,} of its lines are late "
        f"({model['first_class_late_pct']:.0f}%), which makes its odds ratio infinite."
    )
with right:
    st.markdown("**Scheduled vs. actual shipping days**")
    st.dataframe(
        late["by_shipping_mode"],
        use_container_width=True,
        hide_index=True,
        column_config={
            "shipping_mode": "Shipping mode",
            "order_lines": st.column_config.NumberColumn("Order lines"),
            "planned_days": st.column_config.NumberColumn("Scheduled days", format="%.1f"),
            "actual_days": st.column_config.NumberColumn("Actual days", format="%.2f"),
            "late_pct": st.column_config.NumberColumn("Late %", format="%.1f"),
        },
    )
    st.info(
        "**Takeaway:** late deliveries are a scheduling problem, not a product or customer "
        "problem. First Class is promised in 1 day and always takes 2; Second Class is promised "
        "in 2 and takes about 4. Promising realistic lead times would remove most late flags."
    )
