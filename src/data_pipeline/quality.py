"""
Data quality checks — implements data/docs/data_quality_standards.md.

Each check returns one row: what was checked, how many rows fail, and a
status ('pass', 'warn' or 'info'). Used by the dashboard's Data Quality page.
"""

import pandas as pd

from src.utils.db import get_connection

# Shipping modes as they appear in the DataCo order data
VALID_SHIPPING_MODES = ("Standard Class", "Second Class", "First Class", "Same Day")

# (section, rule, SQL counting failing rows, SQL counting all rows, status when failing)
CHECKS = [
    # 2. Completeness
    (
        "Completeness",
        "Order ID is present",
        "SELECT COUNT(*) FROM orders WHERE order_id IS NULL",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Completeness",
        "Order item ID is unique per line",
        "SELECT COUNT(*) - COUNT(DISTINCT order_item_id) FROM orders",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Completeness",
        "Order date is a valid date",
        "SELECT COUNT(*) FROM orders WHERE order_date IS NULL",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Completeness",
        "Customer ID is present",
        "SELECT COUNT(*) FROM orders WHERE customer_id IS NULL",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Completeness",
        "Product name and category are present",
        "SELECT COUNT(*) FROM orders WHERE product_name IS NULL OR category_name IS NULL",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Completeness",
        "Shipping mode is a known mode",
        f"SELECT COUNT(*) FROM orders WHERE shipping_mode NOT IN {VALID_SHIPPING_MODES} "
        "OR shipping_mode IS NULL",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    # 3. Validation rules
    (
        "Validity",
        "Quantity is a positive integer",
        "SELECT COUNT(*) FROM orders WHERE quantity IS NULL OR quantity <= 0 OR quantity <> FLOOR(quantity)",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Validity",
        "Revenue is non-negative",
        "SELECT COUNT(*) FROM orders WHERE revenue < 0",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Validity",
        "Discount rate is between 0 and 1",
        "SELECT COUNT(*) FROM orders WHERE discount_rate < 0 OR discount_rate > 1",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Validity",
        "Unit price is positive",
        "SELECT COUNT(*) FROM orders WHERE unit_price <= 0",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Validity",
        "Ship date is on or after order date",
        "SELECT COUNT(*) FROM orders WHERE ship_date < order_date",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Validity",
        "Order date is not in the future",
        "SELECT COUNT(*) FROM orders WHERE order_date > CURRENT_DATE",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    (
        "Validity",
        "Late delivery risk flag is 0 or 1",
        "SELECT COUNT(*) FROM orders WHERE late_delivery_risk NOT IN (0, 1)",
        "SELECT COUNT(*) FROM orders",
        "warn",
    ),
    # 5. Outliers (flagged for review, not errors)
    (
        "Outliers",
        "Daily category demand within 3 std of its rolling 28-day mean",
        """SELECT COUNT(*) FROM (
            SELECT total_units,
                   AVG(total_units) OVER w AS mean_28d,
                   STDDEV(total_units) OVER w AS std_28d
            FROM daily_demand
            WINDOW w AS (PARTITION BY category_name ORDER BY order_date
                         ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING)
        ) WHERE std_28d > 0 AND ABS(total_units - mean_28d) > 3 * std_28d""",
        "SELECT COUNT(*) FROM daily_demand",
        "info",
    ),
    (
        "Outliers",
        "Single order revenue at or below $10,000",
        "SELECT COUNT(*) FROM (SELECT order_id FROM orders GROUP BY 1 HAVING SUM(revenue) > 10000)",
        "SELECT COUNT(DISTINCT order_id) FROM orders",
        "info",
    ),
    (
        "Outliers",
        "Order lines with negative profit (reviewed, not excluded)",
        "SELECT COUNT(*) FROM orders WHERE profit < 0",
        "SELECT COUNT(*) FROM orders",
        "info",
    ),
]


def run_quality_checks() -> pd.DataFrame:
    """Run every check against the warehouse and return one row per check."""
    con = get_connection(read_only=True)
    try:
        rows = []
        for section, rule, failing_sql, total_sql, fail_status in CHECKS:
            failing = int(con.execute(failing_sql).fetchone()[0] or 0)
            total = int(con.execute(total_sql).fetchone()[0] or 0)
            rows.append(
                {
                    "section": section,
                    "rule": rule,
                    "failing_rows": failing,
                    "total_rows": total,
                    "failing_pct": round(100 * failing / total, 2) if total else 0.0,
                    "status": "pass" if failing == 0 else fail_status,
                }
            )
        return pd.DataFrame(rows)
    finally:
        con.close()


def table_profile() -> pd.DataFrame:
    """Row counts and date coverage for the warehouse tables."""
    con = get_connection(read_only=True)
    try:
        rows = []
        for table in (
            "orders",
            "daily_demand",
            "daily_demand_department",
            "demand_features",
            "department_features",
        ):
            count, first, last = con.execute(
                f"SELECT COUNT(*), MIN(order_date), MAX(order_date) FROM {table}"
            ).fetchone()
            rows.append({"table": table, "rows": count, "first_date": first, "last_date": last})
        return pd.DataFrame(rows)
    finally:
        con.close()


def null_rates() -> pd.DataFrame:
    """Share of NULLs in every column of the orders table."""
    con = get_connection(read_only=True)
    try:
        columns = [r[0] for r in con.execute("DESCRIBE orders").fetchall()]
        counts = con.execute(
            "SELECT COUNT(*), "
            + ", ".join(f'COUNT(*) - COUNT("{c}")' for c in columns)
            + " FROM orders"
        ).fetchone()
        total = counts[0]
        df = pd.DataFrame({"column": columns, "null_rows": counts[1:]})
        df["null_pct"] = (100 * df["null_rows"] / total).round(2) if total else 0.0
        return df.sort_values("null_rows", ascending=False).reset_index(drop=True)
    finally:
        con.close()
