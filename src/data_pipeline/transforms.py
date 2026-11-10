"""
SQL-based feature engineering — mimics Snowflake analytical workflows.

Builds materialized feature tables in DuckDB for the forecasting engine.
"""

import duckdb

from src.utils.config import DUCKDB_PATH


def build_daily_demand_table() -> int:
    """
    Aggregate orders into daily demand by product category.

    The series is densified: every category gets one row per calendar day
    between its first and last order, with zero demand on days without
    orders (is_filled = TRUE). This makes row-based window functions
    downstream equivalent to day-based ones.

    Canceled and suspected-fraud orders are excluded. A category's
    department is its most common department (Electronics appears in two).

    Creates the 'daily_demand' table used by the forecasting engine.
    """
    con = duckdb.connect(DUCKDB_PATH)
    try:
        con.execute("DROP TABLE IF EXISTS daily_demand")
        con.execute("""
            CREATE TABLE daily_demand AS
            WITH valid_orders AS (
                SELECT *
                FROM orders
                WHERE order_date IS NOT NULL AND is_valid_demand
            ),
            daily AS (
                SELECT
                    order_date,
                    category_name,
                    COUNT(DISTINCT order_id) AS order_count,
                    COUNT(*) AS order_lines,
                    SUM(quantity) AS total_units,
                    SUM(revenue) AS total_revenue,
                    AVG(unit_price) AS avg_unit_price,
                    AVG(discount_rate) AS avg_discount,
                    SUM(CASE WHEN late_delivery_risk = 1 THEN 1 ELSE 0 END) AS late_delivery_count,
                    COUNT(DISTINCT customer_id) AS unique_customers
                FROM valid_orders
                GROUP BY order_date, category_name
            ),
            category_department AS (
                SELECT category_name, MODE(department_name) AS department_name
                FROM valid_orders
                GROUP BY category_name
            ),
            calendar AS (
                SELECT
                    category_name,
                    CAST(UNNEST(generate_series(first_date, last_date, INTERVAL 1 DAY)) AS DATE)
                        AS order_date
                FROM (
                    SELECT category_name, MIN(order_date) AS first_date, MAX(order_date) AS last_date
                    FROM daily
                    GROUP BY category_name
                )
            )
            SELECT
                c.order_date,
                c.category_name,
                cd.department_name,
                COALESCE(d.order_count, 0) AS order_count,
                COALESCE(d.order_lines, 0) AS order_lines,
                COALESCE(d.total_units, 0) AS total_units,
                COALESCE(d.total_revenue, 0) AS total_revenue,
                d.avg_unit_price,
                d.avg_discount,
                COALESCE(d.late_delivery_count, 0) AS late_delivery_count,
                COALESCE(d.unique_customers, 0) AS unique_customers,
                d.order_date IS NULL AS is_filled
            FROM calendar c
            JOIN category_department cd ON cd.category_name = c.category_name
            LEFT JOIN daily d ON d.category_name = c.category_name AND d.order_date = c.order_date
            ORDER BY c.order_date, c.category_name
        """)

        row_count, filled = con.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE is_filled) FROM daily_demand"
        ).fetchone()
        print(f"Built daily_demand table: {row_count:,} rows ({filled:,} zero-demand days filled)")
        return row_count
    finally:
        con.close()


def _feature_table_sql(target: str, source: str, key: str) -> str:
    """
    SQL that builds a feature table from a dense daily demand table.

    Every model input for day t uses only data up to day t-1: lags start at 1
    and rolling windows end at 1 PRECEDING. total_units (the target) and the
    other same-day columns are kept for analysis but are not model inputs.

    src/forecasting/features.py mirrors these definitions in Python for
    recursive forecasting — keep the two in sync.
    """

    def rolling(fn: str, days: int) -> str:
        return (
            f"{fn}(total_units) OVER (PARTITION BY {key} ORDER BY order_date "
            f"ROWS BETWEEN {days} PRECEDING AND 1 PRECEDING)"
        )

    return f"""
        CREATE TABLE {target} AS
        WITH base AS (
            SELECT
                order_date,
                {key},
                is_filled,

                -- Target and same-day context (NOT model inputs)
                total_units,
                total_revenue,
                order_count,

                -- Date features (known in advance; DOW: Sunday = 0)
                EXTRACT(DOW FROM order_date) AS day_of_week,
                EXTRACT(MONTH FROM order_date) AS month,
                EXTRACT(QUARTER FROM order_date) AS quarter,
                CASE WHEN EXTRACT(DOW FROM order_date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,

                -- Lag features (calendar is dense, so N rows back = N days back)
                LAG(total_units, 1) OVER w AS lag_1d_units,
                LAG(total_units, 7) OVER w AS lag_7d_units,
                LAG(total_units, 14) OVER w AS lag_14d_units,
                LAG(total_units, 28) OVER w AS lag_28d_units,

                -- Rolling statistics over the previous N days (excluding today)
                {rolling("AVG", 7)} AS rolling_7d_avg_units,
                {rolling("AVG", 14)} AS rolling_14d_avg_units,
                {rolling("AVG", 28)} AS rolling_28d_avg_units,
                {rolling("STDDEV", 14)} AS rolling_14d_std_units

            FROM {source}
            WINDOW w AS (PARTITION BY {key} ORDER BY order_date)
        )
        SELECT *
        FROM base
        WHERE lag_28d_units IS NOT NULL  -- need 28 days of history for all lag features
        ORDER BY order_date, {key}
    """


def build_feature_table() -> int:
    """
    Create the category-level ML feature table (demand_features) with lag
    features, rolling statistics and calendar indicators.

    This is the primary input to the forecasting model.
    """
    con = duckdb.connect(DUCKDB_PATH)
    try:
        con.execute("DROP TABLE IF EXISTS demand_features")
        con.execute(_feature_table_sql("demand_features", "daily_demand", "category_name"))

        row_count = con.execute("SELECT COUNT(*) FROM demand_features").fetchone()[0]
        print(f"Built demand_features table: {row_count:,} rows")
        return row_count
    finally:
        con.close()


def build_department_tables() -> int:
    """
    Build department-level series: daily_demand_department (each department's
    categories summed per day, densified over the department's active range)
    and department_features (same features as demand_features).
    """
    con = duckdb.connect(DUCKDB_PATH)
    try:
        con.execute("DROP TABLE IF EXISTS daily_demand_department")
        con.execute("""
            CREATE TABLE daily_demand_department AS
            WITH daily AS (
                SELECT
                    order_date,
                    department_name,
                    SUM(order_count) AS order_count,
                    SUM(order_lines) AS order_lines,
                    SUM(total_units) AS total_units,
                    SUM(total_revenue) AS total_revenue,
                    SUM(late_delivery_count) AS late_delivery_count
                FROM daily_demand
                WHERE NOT is_filled
                GROUP BY order_date, department_name
            ),
            calendar AS (
                SELECT
                    department_name,
                    CAST(UNNEST(generate_series(first_date, last_date, INTERVAL 1 DAY)) AS DATE)
                        AS order_date
                FROM (
                    SELECT department_name, MIN(order_date) AS first_date, MAX(order_date) AS last_date
                    FROM daily
                    GROUP BY department_name
                )
            )
            SELECT
                c.order_date,
                c.department_name,
                COALESCE(d.order_count, 0) AS order_count,
                COALESCE(d.order_lines, 0) AS order_lines,
                COALESCE(d.total_units, 0) AS total_units,
                COALESCE(d.total_revenue, 0) AS total_revenue,
                COALESCE(d.late_delivery_count, 0) AS late_delivery_count,
                d.order_date IS NULL AS is_filled
            FROM calendar c
            LEFT JOIN daily d ON d.department_name = c.department_name AND d.order_date = c.order_date
            ORDER BY c.order_date, c.department_name
        """)

        con.execute("DROP TABLE IF EXISTS department_features")
        con.execute(
            _feature_table_sql("department_features", "daily_demand_department", "department_name")
        )

        row_count = con.execute("SELECT COUNT(*) FROM department_features").fetchone()[0]
        print(f"Built department_features table: {row_count:,} rows")
        return row_count
    finally:
        con.close()


def build_inventory_risk_view() -> int:
    """
    Create a view summarizing inventory risk signals by category.

    late_delivery_pct is late order lines / all order lines. The risk
    score blends it with demand volatility (coefficient of variation,
    capped at 100%) so both components are on a 0-100 scale.

    Used by the dashboard's inventory risk heatmap.
    """
    con = duckdb.connect(DUCKDB_PATH)
    try:
        con.execute("DROP VIEW IF EXISTS inventory_risk")
        con.execute("""
            CREATE VIEW inventory_risk AS
            WITH stats AS (
                SELECT
                    category_name,
                    department_name,
                    SUM(order_count) AS total_orders,
                    SUM(order_lines) AS total_order_lines,
                    SUM(late_delivery_count) AS total_late_deliveries,
                    SUM(late_delivery_count) * 100.0 / NULLIF(SUM(order_lines), 0) AS late_pct,
                    AVG(total_units) AS avg_units,
                    STDDEV(total_units) AS std_units,
                    AVG(avg_discount) AS avg_discount
                FROM daily_demand
                GROUP BY category_name, department_name
            )
            SELECT
                category_name,
                department_name,
                total_orders,
                total_order_lines,
                total_late_deliveries,
                ROUND(late_pct, 1) AS late_delivery_pct,
                ROUND(avg_units, 1) AS avg_daily_units,
                ROUND(std_units, 1) AS demand_volatility,
                ROUND(avg_discount, 3) AS avg_discount_rate,
                -- Risk score: 60% late-delivery rate + 40% demand volatility (CV, capped at 100)
                ROUND(
                    COALESCE(late_pct, 0) * 0.6
                    + LEAST(COALESCE(std_units / NULLIF(avg_units, 0), 0) * 100, 100) * 0.4,
                    1
                ) AS risk_score
            FROM stats
            ORDER BY risk_score DESC
        """)

        row_count = con.execute("SELECT COUNT(*) FROM inventory_risk").fetchone()[0]
        print(f"Built inventory_risk view: {row_count} categories")
        return row_count
    finally:
        con.close()


if __name__ == "__main__":
    build_daily_demand_table()
    build_feature_table()
    build_department_tables()
    build_inventory_risk_view()
