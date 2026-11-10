"""Demand Forecasts — champion forecasts, method comparison and SHAP drivers."""

import pandas as pd
import streamlit as st

from src.dashboard import data
from src.dashboard.components import forecast_chart, importance_bar, shap_waterfall
from src.forecasting.predict import get_forecast_accuracy

METHOD_LABELS = {
    "xgboost": "XGBoost",
    "moving_average": "28-day moving average",
    "seasonal_naive": "Same weekday last week",
}

st.header("Demand Forecasts")

try:
    trained = data.trained_series()
except Exception as e:
    st.warning(data.TRAIN_HINT)
    st.error(f"Error: {e}")
    st.stop()
if trained.empty:
    st.warning(data.TRAIN_HINT)
    st.stop()

col_level, col_series = st.columns([1, 3])
level = col_level.radio("Level", sorted(trained["level"].unique()), horizontal=True)
options = trained[trained["level"] == level]["name"].tolist()
name = col_series.selectbox(f"Select {level}", options)

with st.spinner(f"Forecasting {name}..."):
    forecast_df = data.forecast(level, name)
    meta = data.forecast_meta(level, name)
metrics = get_forecast_accuracy(forecast_df)
champion = meta["champion"]
baseline_wape = meta["holdout"]["moving_average"]["wape"]

st.caption(
    f"Champion method: **{METHOD_LABELS[champion]}** — lowest error across "
    f"{len(meta['backtest_origins'])} rolling 28-day backtests. Metrics are on the held-out last "
    f"28 days (from {meta['holdout_start']}), which no model saw during training or selection. "
    f"Models use data up to {meta['model_end_date']}."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "WAPE",
    f"{metrics['wape']:.1%}",
    delta=f"{metrics['wape'] - baseline_wape:+.1%} vs 28-day avg",
    delta_color="inverse",
    help="Weighted absolute percentage error: total |error| / total demand",
)
col2.metric("MAE", f"{metrics['mae']:.1f} units")
col3.metric("RMSE", f"{metrics['rmse']:.1f} units")
col4.metric(
    "95% interval coverage",
    f"{metrics['coverage_95ci']:.0%}",
    help="Share of holdout days where the actual fell inside the interval",
)

st.plotly_chart(forecast_chart(forecast_df, name), use_container_width=True)

tab_future, tab_methods, tab_drivers = st.tabs(
    ["Next 28 days", "Method comparison", "Forecast drivers"]
)

with tab_future:
    future = forecast_df[forecast_df["segment"] == "future"]
    w1, w2, w3 = st.columns(3)
    w1.metric("Forecast, next 7 days", f"{future['forecast'].head(7).sum():,.0f} units")
    w2.metric("Forecast, next 28 days", f"{future['forecast'].sum():,.0f} units")
    w3.metric("Avg per day", f"{future['forecast'].mean():.1f} units")
    st.dataframe(
        future[["order_date", "forecast", "lower_bound", "upper_bound"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "order_date": st.column_config.DateColumn("Date"),
            "forecast": st.column_config.NumberColumn("Forecast", format="%.1f"),
            "lower_bound": st.column_config.NumberColumn("95% low", format="%.1f"),
            "upper_bound": st.column_config.NumberColumn("95% high", format="%.1f"),
        },
    )

with tab_methods:
    rows = [
        {
            "Method": METHOD_LABELS[m] + (" ★" if m == champion else ""),
            "Backtest WAPE": meta["cv_wape"][m],
            "Holdout WAPE": meta["holdout"][m]["wape"],
            "Holdout MAE": meta["holdout"][m]["mae"],
            "Interval coverage": meta["holdout"][m]["coverage"],
        }
        for m in METHOD_LABELS
    ]
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Backtest WAPE": st.column_config.NumberColumn(format="%.3f"),
            "Holdout WAPE": st.column_config.NumberColumn(format="%.3f"),
            "Holdout MAE": st.column_config.NumberColumn(format="%.2f"),
            "Interval coverage": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "★ = champion, chosen on backtest WAPE only. Holdout columns are reported, not used for selection."
    )

with tab_drivers:
    from src.forecasting.explain import explain_prediction, get_global_importance
    from src.forecasting.features import FEATURE_COLS

    if champion != "xgboost":
        st.info(
            f"The published forecast for {name} uses the {METHOD_LABELS[champion].lower()}. "
            "The XGBoost model below was trained too; its drivers show what structure, if any, "
            "it found beyond the 28-day average."
        )

    model = data.forecast_model(level, name)
    features = data.model_features(level, name).dropna(subset=FEATURE_COLS)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            importance_bar(get_global_importance(model, features.tail(365))),
            use_container_width=True,
        )
    with right:
        dates = features["order_date"].dt.date.tolist()
        day = st.select_slider(
            "Explain the one-day-ahead prediction for", options=dates[-120:], value=dates[-1]
        )
        row = features[features["order_date"].dt.date == day]
        explanation = explain_prediction(model, row, top_n=len(FEATURE_COLS))
        st.plotly_chart(shap_waterfall(explanation), use_container_width=True)
        st.caption(f"Actual demand that day: {row['total_units'].iloc[0]:.0f} units.")
