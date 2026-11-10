"""Inventory Risk — late deliveries vs. demand volatility by category."""

import streamlit as st

from src.dashboard import data
from src.dashboard.components import risk_scatter

st.header("Inventory Risk Assessment")
st.caption(
    "Risk score = 0.6 × late delivery rate + 0.4 × demand volatility (coefficient of variation, "
    "capped at 100). Low-volume categories often hit the volatility cap — read their scores "
    "together with average daily units."
)

try:
    risk_df = data.risk_table()
except Exception as e:
    st.warning(data.SETUP_HINT)
    st.error(f"Error: {e}")
    st.stop()

min_units = st.slider(
    "Minimum average daily units",
    0.0,
    float(risk_df["avg_daily_units"].max()),
    5.0,
    step=0.5,
    help="Hides low-volume categories by default: their volatility hits the cap from noise alone, "
    "which pushes them to the top of the ranking. Drag to 0 to see all categories.",
)
hidden = int((risk_df["avg_daily_units"] < min_units).sum())
if hidden:
    st.caption(f"{hidden} categories under {min_units:g} units/day hidden.")
view = risk_df[risk_df["avg_daily_units"] >= min_units]

st.subheader("Highest risk categories")
for _, row in view.head(5).iterrows():
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    col1.write(f"**{row['category_name']}** ({row['department_name']})")
    col2.metric("Risk score", f"{row['risk_score']:.1f}")
    col3.metric("Late %", f"{row['late_delivery_pct']:.1f}%")
    col4.metric("Avg units/day", f"{row['avg_daily_units']:.1f}")

st.plotly_chart(risk_scatter(view), use_container_width=True)

with st.expander("View full risk table"):
    st.dataframe(view, use_container_width=True, hide_index=True)
