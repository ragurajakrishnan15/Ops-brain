"""
Feature store — loads feature tables into pandas for ML training.

Keeps the SQL layer (DuckDB) cleanly separated from the ML layer (sklearn/xgb).
Series exist at two levels: 'category' (demand_features) and
'department' (department_features).
"""

import pandas as pd

from src.utils.config import MIN_HISTORY_DAYS, MODEL_END_DATE
from src.utils.db import get_connection

# level -> (feature table, daily table, key column)
SERIES_LEVELS = {
    "category": ("demand_features", "daily_demand", "category_name"),
    "department": ("department_features", "daily_demand_department", "department_name"),
}


def _level(level: str) -> tuple[str, str, str]:
    if level not in SERIES_LEVELS:
        raise ValueError(f"Unknown series level {level!r}; expected one of {list(SERIES_LEVELS)}")
    return SERIES_LEVELS[level]


def load_features(
    level: str = "category",
    name: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load a feature table as a pandas DataFrame.

    Args:
        level: 'category' or 'department'.
        name: If provided, filter to a single series (e.g. "Men's Footwear").
        end_date: If provided, keep rows on or before this ISO date.
    """
    table, _, key = _level(level)
    query = f"SELECT * FROM {table} WHERE 1 = 1"
    params: list = []
    if name is not None:
        query += f" AND {key} = ?"
        params.append(name)
    if end_date is not None:
        query += " AND order_date <= CAST(? AS DATE)"
        params.append(end_date)
    query += " ORDER BY order_date"

    con = get_connection(read_only=True)
    try:
        return con.execute(query, params).fetchdf()
    finally:
        con.close()


def load_daily_units(level: str, name: str, end_date: str | None = None) -> pd.Series:
    """Return a series' dense daily units as a pandas Series indexed by date."""
    _, daily, key = _level(level)
    query = f"SELECT order_date, total_units FROM {daily} WHERE {key} = ?"
    params: list = [name]
    if end_date is not None:
        query += " AND order_date <= CAST(? AS DATE)"
        params.append(end_date)
    query += " ORDER BY order_date"

    con = get_connection(read_only=True)
    try:
        df = con.execute(query, params).fetchdf()
    finally:
        con.close()
    return df.set_index(pd.to_datetime(df["order_date"]))["total_units"].astype(float)


def get_series_names(
    level: str = "category",
    min_rows: int = MIN_HISTORY_DAYS,
    end_date: str | None = MODEL_END_DATE,
) -> list[str]:
    """Return series at a level with at least `min_rows` feature rows up to end_date."""
    table, _, key = _level(level)
    con = get_connection(read_only=True)
    try:
        result = con.execute(
            f"""
            SELECT {key}
            FROM {table}
            WHERE order_date <= CAST(? AS DATE)
            GROUP BY {key}
            HAVING COUNT(*) >= ?
            ORDER BY {key}
        """,
            [end_date or "9999-12-31", min_rows],
        ).fetchall()
        return [row[0] for row in result]
    finally:
        con.close()


def get_categories() -> list[str]:
    """Return all unique product categories in the dataset."""
    con = get_connection(read_only=True)
    try:
        result = con.execute(
            "SELECT DISTINCT category_name FROM demand_features ORDER BY category_name"
        ).fetchall()
        return [row[0] for row in result]
    finally:
        con.close()


def get_date_range() -> tuple[str, str]:
    """Return the min and max dates in the feature table."""
    con = get_connection(read_only=True)
    try:
        result = con.execute(
            "SELECT MIN(order_date), MAX(order_date) FROM demand_features"
        ).fetchone()
        return str(result[0]), str(result[1])
    finally:
        con.close()


def get_summary_stats() -> pd.DataFrame:
    """Return summary statistics per category for the dashboard overview."""
    con = get_connection(read_only=True)
    try:
        return con.execute("""
            SELECT
                category_name,
                COUNT(*) AS days_of_data,
                ROUND(AVG(total_units), 1) AS avg_daily_units,
                ROUND(STDDEV(total_units), 1) AS std_daily_units,
                ROUND(SUM(total_revenue), 0) AS total_revenue,
                MIN(order_date) AS first_date,
                MAX(order_date) AS last_date
            FROM demand_features
            GROUP BY category_name
            ORDER BY total_revenue DESC
        """).fetchdf()
    finally:
        con.close()
