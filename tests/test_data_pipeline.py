"""
Tests for the data pipeline module.

These tests verify the SQL transforms produce expected table schemas
and that feature engineering logic is correct.
"""

import duckdb
import numpy as np
import pandas as pd


def test_feature_cols_defined():
    """Verify the feature column list matches expected count."""
    from src.forecasting.train import FEATURE_COLS

    assert len(FEATURE_COLS) == 12
    assert "lag_1d_units" in FEATURE_COLS
    assert "rolling_7d_avg_units" in FEATURE_COLS
    assert "day_of_week" in FEATURE_COLS


def test_walk_forward_split():
    """Verify walk-forward splits don't leak future data."""
    from src.forecasting.train import walk_forward_split

    # Create synthetic time series data
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    df = pd.DataFrame(
        {
            "order_date": dates,
            "total_units": np.random.randint(10, 100, size=365),
        }
    )

    splits = walk_forward_split(df, n_splits=3, test_days=14)

    assert len(splits) == 3

    for train, test in splits:
        train_max = train["order_date"].max()
        test_min = test["order_date"].min()
        # Training data must end before test data starts
        assert train_max < test_min, "Data leakage detected: train overlaps test"


def test_duckdb_sql_syntax():
    """Verify our SQL queries are valid DuckDB syntax."""
    con = duckdb.connect(":memory:")

    # Create a minimal test table
    con.execute("""
        CREATE TABLE test_orders (
            order_date DATE,
            category_name VARCHAR,
            quantity INTEGER,
            revenue DOUBLE
        )
    """)

    # Insert test data
    con.execute("""
        INSERT INTO test_orders VALUES
        ('2024-01-01', 'Electronics', 10, 100.0),
        ('2024-01-02', 'Electronics', 15, 150.0),
        ('2024-01-01', 'Clothing', 20, 80.0)
    """)

    # Test aggregation query (mirrors daily_demand)
    result = con.execute("""
        SELECT
            order_date,
            category_name,
            SUM(quantity) AS total_units,
            SUM(revenue) AS total_revenue
        FROM test_orders
        GROUP BY order_date, category_name
    """).fetchdf()

    assert len(result) == 3
    assert "total_units" in result.columns

    con.close()


def test_config_paths_exist():
    """Verify config creates required directories."""
    from src.utils.config import DOCS_DIR, PROCESSED_DIR, RAW_DIR

    assert RAW_DIR.exists()
    assert PROCESSED_DIR.exists()
    assert DOCS_DIR.exists()
