"""
End-to-end evaluation of the assistant against Gemini.

Skipped unless GOOGLE_API_KEY is set and the warehouse + vector store exist
(run scripts/setup_data.py first). Run with:

    pytest -m llm -v

Document questions check routing, the cited source and a key fact. Data
questions compare the answer with ground truth computed directly in DuckDB.
"""

import re
from pathlib import Path

import duckdb
import pytest

from src.utils.config import CHROMA_PERSIST_DIR, DUCKDB_PATH, GOOGLE_API_KEY

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not GOOGLE_API_KEY, reason="GOOGLE_API_KEY not set"),
    pytest.mark.skipif(
        not Path(DUCKDB_PATH).exists() or not Path(CHROMA_PERSIST_DIR).exists(),
        reason="run scripts/setup_data.py first",
    ),
]

DOC_CASES = [
    (
        "What multiplier is used for safety stock and what service level does it target?",
        "inventory_management_sop.md",
        ["1.65", "95"],
    ),
    ("Who has to approve Same Day shipping for orders above $5,000?", "shipping_policy.md", ["VP"]),
    (
        "Which order statuses are excluded from demand?",
        "order_status_and_cancellation_policy.md",
        ["CANCELED", "SUSPECTED_FRAUD"],
    ),
    ("What score makes a supplier 'at risk'?", "supplier_management_sop.md", ["70"]),
    ("How is the inventory risk score weighted?", "kpi_glossary.md", ["0.6", "0.4"]),
]

DATA_CASES = [
    ("How many order lines are in the orders table?", "SELECT COUNT(*) FROM orders"),
    (
        "How many distinct product categories are there in the orders data?",
        "SELECT COUNT(DISTINCT category_name) FROM orders",
    ),
    (
        "Which category has the highest risk score?",
        "SELECT category_name FROM inventory_risk ORDER BY risk_score DESC LIMIT 1",
    ),
    (
        "Which shipping mode has the most order lines?",
        "SELECT shipping_mode FROM orders GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1",
    ),
    (
        "How many order lines were canceled or flagged as suspected fraud?",
        "SELECT COUNT(*) FROM orders WHERE NOT is_valid_demand",
    ),
]


def _truth(sql: str):
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        return con.execute(sql).fetchone()[0]
    finally:
        con.close()


def _mentions(answer: str, value) -> bool:
    if isinstance(value, (int, float)):
        numbers = {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", answer)}
        return int(value) in numbers
    return str(value).strip().lower() in answer.lower()


@pytest.mark.parametrize("question, source, facts", DOC_CASES)
def test_document_questions(question, source, facts):
    from src.rag_assistant.chat import chat

    result = chat(question)
    assert result["mode"] == "docs"
    assert source in [s["source"] for s in result["sources"]]
    for fact in facts:
        assert fact.lower() in result["answer"].lower(), result["answer"]


@pytest.mark.parametrize("question, truth_sql", DATA_CASES)
def test_data_questions(question, truth_sql):
    from src.rag_assistant.chat import chat

    result = chat(question)
    assert result["mode"] == "data"
    assert result["error"] is None, result["sql"]
    assert _mentions(result["answer"], _truth(truth_sql)), (result["answer"], result["sql"])
