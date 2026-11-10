"""DuckDB connection helper — thin wrapper for consistent access."""

import duckdb

from src.utils.config import DUCKDB_PATH


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection to the warehouse file."""
    return duckdb.connect(DUCKDB_PATH, read_only=read_only)


def run_query(sql: str, params: list | None = None) -> list[dict]:
    """Execute a SQL query and return results as a list of dicts."""
    con = get_connection(read_only=True)
    try:
        result = con.execute(sql, params or [])
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    finally:
        con.close()


def run_query_df(sql: str, params: list | None = None):
    """Execute a SQL query and return a pandas DataFrame."""
    con = get_connection(read_only=True)
    try:
        return con.execute(sql, params or []).fetchdf()
    finally:
        con.close()
