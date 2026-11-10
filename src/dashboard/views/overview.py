"""Overview — headline KPIs, demand trend and forecast alerts."""

import plotly.express as px
import streamlit as st

from src.dashboard import data
from src.utils.config import MODEL_END_DATE

st.title("📊 OpsBrain")
st.caption("AI-Powered Supply Chain Analytics & Operations Intelligence")

try:
    kpis = data.overview_kpis()
    stats = data.summary_stats()
except Exception as e:
    st.warning(data.SETUP_HINT)
    st.error(f"Error: {e}")
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Orders", f"{kpis['orders']:,}", help=f"{kpis['order_lines']:,} order lines")
col2.metric(
    "Revenue",
    f"${kpis['revenue'] / 1e6:,.1f}M",
    help="Excludes canceled and suspected-fraud orders",
)
col3.metric("Categories", f"{kpis['categories']}")
col4.metric(
    "Late deliveries", f"{kpis['late_pct']:.1f}%", help="Share of order lines delivered late"
)
col5.metric(
    "Data range",
    f"{kpis['first_date']:%Y}–{kpis['last_date']:%y}",
    help=f"Orders from {kpis['first_date']:%d %b %Y} to {kpis['last_date']:%d %b %Y}",
)

st.subheader("Monthly demand by department")
monthly = data.monthly_units()
fig = px.area(
    monthly,
    x="month",
    y="units",
    color="department_name",
    labels={"month": "", "units": "Units", "department_name": "Department"},
    template="plotly_white",
    height=380,
    # 11 departments: the default 10-color palette would repeat colors
    color_discrete_sequence=px.colors.qualitative.Dark24,
)
fig.add_vline(x=MODEL_END_DATE, line=dict(color="#6b7280", dash="dot", width=1))
fig.add_annotation(
    x=MODEL_END_DATE,
    y=1,
    yref="paper",
    text="Catalog change — models use data up to here",
    showarrow=False,
    xanchor="right",
    yanchor="bottom",
    font=dict(size=11, color="#6b7280"),
)
fig.update_layout(margin=dict(l=40, r=20, t=30, b=30), legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast alerts")
try:
    with st.spinner("Checking forecast health for all series (first load takes ~15 s)..."):
        health = data.forecast_health()
except Exception:
    health = None
if health is None or health.empty:
    st.info(data.TRAIN_HINT)
else:
    alerts = health[health["status"] == "alert"]
    a1, a2, a3 = st.columns(3)
    a1.metric("Series forecast", len(health))
    a2.metric("Need attention", len(alerts))
    a3.metric("Champion: XGBoost", f"{(health['champion'] == 'xgboost').sum()} of {len(health)}")
    if len(alerts):
        st.dataframe(
            alerts[["level", "series", "avg_daily_units", "holdout_wape", "alerts"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "level": st.column_config.TextColumn("Level"),
                "series": st.column_config.TextColumn("Series"),
                "avg_daily_units": st.column_config.NumberColumn("Avg units/day", format="%.1f"),
                "holdout_wape": st.column_config.NumberColumn("Holdout WAPE", format="%.2f"),
                "alerts": st.column_config.TextColumn("Why", width="large"),
            },
        )

st.subheader("Category summary")
st.dataframe(
    stats,
    use_container_width=True,
    hide_index=True,
    column_config={
        "category_name": st.column_config.TextColumn("Category"),
        "department_name": st.column_config.TextColumn("Department"),
        "active_days": st.column_config.NumberColumn("Days with orders"),
        "avg_daily_units": st.column_config.NumberColumn("Avg units/day", format="%.1f"),
        "std_daily_units": st.column_config.NumberColumn("Std units/day", format="%.1f"),
        "total_revenue": st.column_config.NumberColumn("Revenue ($)", format="dollar"),
        "first_date": st.column_config.DateColumn("First order"),
        "last_date": st.column_config.DateColumn("Last order"),
    },
)
