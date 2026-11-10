"""
Offline tests for the assistant's LLM plumbing.

A fake chat model stands in for Gemini, so these run without an API key
or network access.
"""

from types import SimpleNamespace

import pytest

from src.rag_assistant import chat, sql_agent
from src.utils import llm


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply)


def test_response_text_handles_string_and_parts():
    assert llm.response_text(SimpleNamespace(content="hello")) == "hello"
    parts = [
        {"type": "thinking", "thinking": "hidden reasoning"},
        {"type": "text", "text": "SELECT 1"},
        " LIMIT 1",
    ]
    assert llm.response_text(SimpleNamespace(content=parts)) == "SELECT 1 LIMIT 1"


def test_missing_key_gives_clear_error(monkeypatch):
    llm.get_llm.cache_clear()
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", "")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        llm.get_llm()
    llm.get_llm.cache_clear()


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("data", "data"),
        ("Docs\n", "docs"),
        ([{"type": "text", "text": "DATA"}], "data"),
    ],
)
def test_router_classification(monkeypatch, reply, expected):
    monkeypatch.setattr(chat, "get_llm", lambda *a, **k: FakeLLM(reply))
    assert chat.route_query("How many orders shipped late?") == expected


def test_text_to_sql_extracts_code_block(monkeypatch):
    fake = FakeLLM("Here you go:\n```sql\nSELECT COUNT(*) FROM orders LIMIT 50\n```")
    monkeypatch.setattr(sql_agent, "get_llm", lambda *a, **k: fake)
    assert sql_agent.text_to_sql("count orders") == "SELECT COUNT(*) FROM orders LIMIT 50"
    assert "count orders" in fake.prompts[0]


# --- SQL agent sandbox -------------------------------------------------------


@pytest.fixture
def tiny_db(tmp_path):
    import duckdb

    path = str(tmp_path / "tiny.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE orders AS SELECT range AS order_id, range % 3 AS qty FROM range(500)")
    con.close()
    (tmp_path / "secret.csv").write_text("a,b\n1,2\n")
    return path, tmp_path


def test_safe_query_runs_select_and_caps_rows(tiny_db):
    path, _ = tiny_db
    df, truncated = sql_agent.run_safe_query("SELECT * FROM orders;", db_path=path)
    assert len(df) == sql_agent.MAX_RESULT_ROWS and truncated
    df, truncated = sql_agent.run_safe_query(
        "WITH t AS (SELECT qty, COUNT(*) n FROM orders GROUP BY 1) SELECT * FROM t", db_path=path
    )
    assert len(df) == 3 and not truncated


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
        "COPY orders TO 'out.csv'",
        "ATTACH 'other.duckdb'",
        "SET enable_external_access = true",
        "INSTALL httpfs",
    ],
)
def test_unsafe_statements_rejected(tiny_db, sql):
    path, _ = tiny_db
    with pytest.raises(sql_agent.UnsafeQueryError):
        sql_agent.run_safe_query(sql, db_path=path)


def test_file_access_blocked_inside_select(tiny_db):
    import duckdb

    path, tmp = tiny_db
    secret = (tmp / "secret.csv").as_posix()
    with pytest.raises(duckdb.Error):
        sql_agent.run_safe_query(f"SELECT * FROM read_csv('{secret}')", db_path=path)


def test_ask_data_retries_with_error_feedback(monkeypatch, tiny_db):
    path, _ = tiny_db
    replies = iter(
        ["```sql\nSELECT nope FROM orders\n```", "```sql\nSELECT COUNT(*) AS n FROM orders\n```"]
    )
    calls = []

    def fake_text_to_sql(question, previous_sql=None, error=None):
        calls.append((previous_sql, error))
        return sql_agent.extract_sql(next(replies))

    monkeypatch.setattr(sql_agent, "text_to_sql", fake_text_to_sql)
    monkeypatch.setattr(sql_agent, "DUCKDB_PATH", path)
    result = sql_agent.ask_data("how many orders?", summarize=False)

    assert result["error"] is None and result["results"]["n"].iloc[0] == 500
    assert calls[0] == (None, None)
    assert calls[1][0] == "SELECT nope FROM orders" and "nope" in calls[1][1]


def test_chat_data_mode_returns_summary_and_table(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(chat, "route_query", lambda q: "data")
    monkeypatch.setattr(
        chat,
        "ask_data",
        lambda q: {
            "sql": "SELECT 1",
            "results": pd.DataFrame({"n": [1]}),
            "truncated": False,
            "summary": "There is 1 order.",
            "error": None,
            "row_count": 1,
        },
    )
    result = chat.chat("how many?")
    assert result["mode"] == "data" and result["answer"] == "There is 1 order."
    assert result["results"]["n"].tolist() == [1]
