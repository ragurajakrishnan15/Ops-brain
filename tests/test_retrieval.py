"""
Retrieval quality on the real operations documents.

Uses the local embedding model (no API key). Each question must find its
expected document among the top-3 chunks.
"""

import pytest

from src.rag_assistant import ingest_docs, retriever

RETRIEVAL_CASES = [
    ("What happens when a supplier scores below 70 points?", "supplier_management_sop.md"),
    ("Do canceled orders count as demand?", "order_status_and_cancellation_policy.md"),
    ("How is the inventory risk score calculated?", "kpi_glossary.md"),
    ("Who must approve Same Day shipping over $5,000?", "shipping_policy.md"),
    ("How is safety stock calculated?", "inventory_management_sop.md"),
    ("What is the difference between WAPE and MAPE?", "kpi_glossary.md"),
    (
        "When are planners allowed to override the statistical forecast?",
        "demand_planning_process.md",
    ),
    ("How are outliers in daily demand detected?", "data_quality_standards.md"),
    (
        "What happens to orders flagged as suspected fraud?",
        "order_status_and_cancellation_policy.md",
    ),
    ("How is the champion forecasting method chosen?", "demand_planning_process.md"),
]


@pytest.fixture(scope="module")
def store_dir(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("chroma"))
    ingest_docs.ingest_all(persist_dir=path)
    yield path
    ingest_docs.get_vector_store.cache_clear()


def test_reingest_does_not_duplicate(store_dir):
    before = ingest_docs.get_vector_store(store_dir)._collection.count()
    ingest_docs.ingest_all(persist_dir=store_dir)
    assert ingest_docs.get_vector_store(store_dir)._collection.count() == before > 0


def test_chunks_carry_file_and_section(store_dir):
    hits = retriever.retrieve_with_scores("late delivery escalation", persist_dir=store_dir)
    assert all(h["source"].endswith(".md") and "/" not in h["source"] for h in hits)
    assert all(h["section"] for h in hits)


@pytest.mark.parametrize("question, expected", RETRIEVAL_CASES)
def test_expected_document_in_top_3(store_dir, question, expected):
    hits = retriever.retrieve_with_scores(question, top_k=3, persist_dir=store_dir)
    assert expected in [h["source"] for h in hits], [(h["source"], h["section"]) for h in hits]
