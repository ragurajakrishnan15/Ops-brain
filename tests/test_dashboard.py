"""
Smoke tests for the Streamlit dashboard.

Renders every page headlessly with Streamlit's AppTest against the real
warehouse and trained models, and fails on any exception. AppTest (Streamlit
1.42) doesn't execute st.navigation pages, so each view script runs directly. Skipped when
the pipeline or training hasn't been run.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.utils.config import DUCKDB_PATH, PROJECT_ROOT

VIEWS = PROJECT_ROOT / "src" / "dashboard" / "views"
# page -> text its title/header must contain
PAGES = {
    "overview": "OpsBrain",
    "forecasts": "Demand Forecasts",
    "inventory_risk": "Inventory Risk Assessment",
    "data_quality": "Data Quality",
    "statistics": "Statistical Tests",
    "assistant": "Operations Assistant",
}

pytestmark = pytest.mark.skipif(
    not Path(DUCKDB_PATH).exists() or not (PROJECT_ROOT / "mlruns").exists(),
    reason="run scripts/setup_data.py and python -m src.forecasting.train first",
)


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(VIEWS / f"{page}.py"), default_timeout=180)
    at.run()
    return at


def test_app_registers_every_view():
    app = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    for page in PAGES:
        assert f"{{VIEWS}}/{page}.py" in app
        assert (VIEWS / f"{page}.py").exists()


@pytest.mark.parametrize("page, heading", PAGES.items())
def test_page_renders_without_errors(page, heading):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    headings = [h.value for h in list(at.title) + list(at.header)]
    assert any(heading in h for h in headings), headings


def test_forecast_page_shows_metrics_and_switches_series():
    at = _run("forecasts")
    labels = [m.label for m in at.metric]
    assert {"WAPE", "MAE", "95% interval coverage"} <= set(labels)

    at.radio[0].set_value("department").run()
    assert not at.exception
    first = at.selectbox[0].options[0]
    at.selectbox[0].set_value(first).run()
    assert not at.exception and not at.error


def test_data_quality_page_reports_checks():
    at = _run("data_quality")
    passing = next(m for m in at.metric if m.label == "Checks passing")
    assert " of " in passing.value
