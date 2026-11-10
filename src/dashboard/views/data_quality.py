"""Data Quality — checks from data/docs/data_quality_standards.md."""

import streamlit as st

from src.dashboard import data

STATUS_ICONS = {"pass": "✅ pass", "warn": "❌ fail", "info": "🔎 review"}

st.header("Data Quality")
st.caption(
    "Completeness, validation and outlier rules from the Data Quality Standards. "
    "Outlier checks flag rows for review; they are not errors."
)

try:
    checks = data.quality_checks()
    profile = data.table_profile()
    nulls = data.null_rates()
except Exception as e:
    st.warning(data.SETUP_HINT)
    st.error(f"Error: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Checks passing", f"{(checks['status'] == 'pass').sum()} of {len(checks)}")
col2.metric("Failing rules", int((checks["status"] == "warn").sum()))
col3.metric("Flagged for review", int((checks["status"] == "info").sum()))

for section in checks["section"].unique():
    st.subheader(section)
    rows = checks[checks["section"] == section].assign(
        status=lambda d: d["status"].map(STATUS_ICONS)
    )
    st.dataframe(
        rows[["status", "rule", "failing_rows", "total_rows", "failing_pct"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "status": st.column_config.TextColumn("Status", width="small"),
            "rule": st.column_config.TextColumn("Rule", width="large"),
            "failing_rows": st.column_config.NumberColumn("Failing rows"),
            "total_rows": st.column_config.NumberColumn("Rows checked"),
            "failing_pct": st.column_config.NumberColumn("Failing %", format="%.2f%%"),
        },
    )

st.subheader("Freshness and coverage")
last = profile.loc[profile["table"] == "orders", "last_date"].iloc[0]
st.caption(
    f"Latest order: **{last}**. The standard is a daily refresh by 06:00 UTC; this is a historical "
    "dataset (DataCo, 2015–2018), so freshness is reported rather than enforced."
)
st.dataframe(profile, use_container_width=True, hide_index=True)

with st.expander("Null rates by column (orders)"):
    st.dataframe(nulls, use_container_width=True, hide_index=True)
