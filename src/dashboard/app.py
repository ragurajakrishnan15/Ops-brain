"""
OpsBrain Dashboard — Streamlit data product.

Run with:
    streamlit run src/dashboard/app.py

Pages (src/dashboard/views/):
1. Overview — headline KPIs, demand trend, forecast alerts
2. Demand Forecasts — champion forecasts with intervals, method comparison, SHAP drivers
3. Inventory Risk — late deliveries vs. demand volatility by category
4. Data Quality — checks from the data quality standards
5. Operations Assistant — chat over the data (SQL) and documents (RAG)
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit only puts this file's folder on sys.path; add the project root so `src.` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="OpsBrain — Supply Chain Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    [
        st.Page("views/overview.py", title="Overview", icon="📊", default=True),
        st.Page("views/forecasts.py", title="Demand Forecasts", icon="📈"),
        st.Page("views/inventory_risk.py", title="Inventory Risk", icon="⚠️"),
        st.Page("views/data_quality.py", title="Data Quality", icon="✅"),
        st.Page("views/assistant.py", title="Operations Assistant", icon="💬"),
    ]
)

st.sidebar.caption("AI-Powered Supply Chain Analytics & Operations Intelligence")
navigation.run()
