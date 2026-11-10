"""
OpsBrain Dashboard — Streamlit data product.

Run with:
    streamlit run streamlit_app.py

Pages (src/dashboard/views/):
1. Overview — what the project is, key findings, KPIs, demand trend, forecast alerts
2. Demand Forecasts — champion forecasts with intervals, method comparison, SHAP drivers
3. Statistical Tests — Diebold–Mariano tests of the models; what drives late deliveries
4. Inventory Risk — late deliveries vs. demand volatility by category
5. Data Quality — checks from the data quality standards
6. Operations Assistant — chat over the data (SQL) and documents (RAG)

On a fresh deployment without data, the prebuilt data bundle is downloaded
once at startup (see src/utils/bundle.py).
"""

import sys
from pathlib import Path

import streamlit as st

# Make `src.` importable however the app is launched
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.bundle import ensure_data  # noqa: E402

st.set_page_config(
    page_title="OpsBrain — Supply Chain Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def _prepare_data() -> str:
    return ensure_data()


try:
    with st.spinner("First start: downloading the prebuilt data bundle (~1 minute)..."):
        _prepare_data()
except Exception as e:
    st.error(
        f"Couldn't load the data bundle: {e}\n\n"
        "Locally, run `python scripts/setup_data.py` and `python -m src.forecasting.train`."
    )

VIEWS = "src/dashboard/views"
navigation = st.navigation(
    [
        st.Page(f"{VIEWS}/overview.py", title="Overview", icon="📊", default=True),
        st.Page(f"{VIEWS}/forecasts.py", title="Demand Forecasts", icon="📈"),
        st.Page(f"{VIEWS}/statistics.py", title="Statistical Tests", icon="🧪"),
        st.Page(f"{VIEWS}/inventory_risk.py", title="Inventory Risk", icon="⚠️"),
        st.Page(f"{VIEWS}/data_quality.py", title="Data Quality", icon="✅"),
        st.Page(f"{VIEWS}/assistant.py", title="Operations Assistant", icon="💬"),
    ]
)

st.sidebar.caption("AI-Powered Supply Chain Analytics & Operations Intelligence")
st.sidebar.markdown(
    "[GitHub repository](https://github.com/ragurajakrishnan15/Ops-brain) · "
    "[EDA report](https://ragurajakrishnan15.github.io/Ops-brain/eda.html)"
)
navigation.run()
