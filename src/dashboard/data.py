"""
Cached data access for the dashboard.

Every warehouse query and forecast goes through here so Streamlit reruns
(each widget interaction) don't recompute them. Data is cached for an hour;
models are cached for the life of the process.
"""

import pandas as pd
import streamlit as st

from src.utils.db import run_query_df

TTL = 3600


@st.cache_data(ttl=TTL, show_spinner=False)
def overview_kpis() -> dict:
    row = run_query_df("""
        SELECT
            COUNT(*) AS order_lines,
            COUNT(DISTINCT order_id) AS orders,
            SUM(revenue) FILTER (WHERE is_valid_demand) AS revenue,
            COUNT(DISTINCT category_name) AS categories,
            ROUND(100.0 * AVG(late_delivery_risk), 1) AS late_pct,
            MIN(order_date) AS first_date,
            MAX(order_date) AS last_date
        FROM orders
    """).iloc[0]
    return row.to_dict()


@st.cache_data(ttl=TTL, show_spinner=False)
def monthly_units() -> pd.DataFrame:
    return run_query_df("""
        SELECT DATE_TRUNC('month', order_date) AS month, department_name, SUM(total_units) AS units
        FROM daily_demand_department
        GROUP BY 1, 2
        ORDER BY 1
    """)


@st.cache_data(ttl=TTL, show_spinner=False)
def summary_stats() -> pd.DataFrame:
    from src.data_pipeline.feature_store import get_summary_stats

    return get_summary_stats()


@st.cache_data(ttl=TTL, show_spinner=False)
def risk_table() -> pd.DataFrame:
    return run_query_df("SELECT * FROM inventory_risk ORDER BY risk_score DESC")


@st.cache_data(ttl=TTL, show_spinner=False)
def trained_series() -> pd.DataFrame:
    from src.forecasting.predict import list_trained_series

    return list_trained_series()


@st.cache_data(ttl=TTL, show_spinner=False)
def forecast(level: str, name: str) -> pd.DataFrame:
    from src.forecasting.predict import forecast_series

    return forecast_series(level, name)


@st.cache_data(ttl=TTL, show_spinner=False)
def forecast_meta(level: str, name: str) -> dict:
    from src.forecasting.predict import load_series

    return load_series(level, name)[1]


@st.cache_resource(show_spinner=False)
def forecast_model(level: str, name: str):
    from src.forecasting.predict import load_series

    return load_series(level, name)[0]


@st.cache_data(ttl=TTL, show_spinner=False)
def model_features(level: str, name: str) -> pd.DataFrame:
    from src.data_pipeline.feature_store import load_features
    from src.utils.config import MODEL_END_DATE

    df = load_features(level, name, end_date=MODEL_END_DATE)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


@st.cache_data(ttl=TTL, show_spinner=False)
def forecast_health() -> pd.DataFrame:
    from src.forecasting.alerts import forecast_health as compute

    return compute()


@st.cache_data(ttl=TTL, show_spinner=False)
def quality_checks() -> pd.DataFrame:
    from src.data_pipeline.quality import run_quality_checks

    return run_quality_checks()


@st.cache_data(ttl=TTL, show_spinner=False)
def table_profile() -> pd.DataFrame:
    from src.data_pipeline.quality import table_profile as compute

    return compute()


@st.cache_data(ttl=TTL, show_spinner=False)
def null_rates() -> pd.DataFrame:
    from src.data_pipeline.quality import null_rates as compute

    return compute()


SETUP_HINT = "Run the data pipeline first:\n\n```bash\npython scripts/setup_data.py\n```"
TRAIN_HINT = "Train the forecasting models first:\n\n```bash\npython -m src.forecasting.train\n```"
