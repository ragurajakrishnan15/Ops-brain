"""
Ingest raw supply chain CSV into DuckDB warehouse.

Expects the DataCo Smart Supply Chain dataset CSV in data/raw/.
Download from: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
"""

from pathlib import Path

import duckdb
import pandas as pd

from src.utils.config import DUCKDB_PATH, RAW_CSV_PATH

# Personally identifying columns — dropped at ingest, never stored in the warehouse
PII_COLUMNS = [
    "customer_email",
    "customer_password",
    "customer_fname",
    "customer_lname",
    "customer_street",
]

# Order statuses that don't represent real demand
NON_DEMAND_STATUSES = ("CANCELED", "SUSPECTED_FRAUD")

# DataCo timestamps look like '1/31/2018 22:56'
DATACO_TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M"


def standardize_column_name(col: str) -> str:
    """'order date (DateOrders)' -> 'order_date_dateorders'."""
    return col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")


def ingest_supply_chain(csv_path: Path | None = None) -> int:
    """
    Load the supply chain CSV into DuckDB as the 'orders_raw' table.

    Returns the number of rows loaded.
    """
    csv_path = Path(csv_path) if csv_path is not None else RAW_CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Download it from Kaggle and place DataCoSupplyChainDataset.csv in data/raw/ "
            "(or set RAW_CSV_PATH)."
        )

    con = duckdb.connect(DUCKDB_PATH)
    try:
        # pandas handles the file's latin-1 encoding; DuckDB's reader rejects it
        df = pd.read_csv(csv_path, encoding="latin-1")
        df.columns = [standardize_column_name(col) for col in df.columns]
        df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])

        con.execute("DROP TABLE IF EXISTS orders_raw")
        con.execute("CREATE TABLE orders_raw AS SELECT * FROM df")

        row_count = con.execute("SELECT COUNT(*) FROM orders_raw").fetchone()[0]
        print(f"Loaded {row_count:,} rows into orders_raw")

        return row_count
    finally:
        con.close()


def create_clean_orders_table() -> int:
    """
    Create a cleaned 'orders' table with standardized types and useful columns.

    This is the table all downstream queries should use. Rows with a
    non-demand status are kept but flagged with is_valid_demand = FALSE.
    """
    non_demand = ", ".join(f"'{s}'" for s in NON_DEMAND_STATUSES)

    con = duckdb.connect(DUCKDB_PATH)
    try:
        con.execute("DROP TABLE IF EXISTS orders")
        con.execute(rf"""
            CREATE TABLE orders AS
            SELECT
                -- Order identifiers
                order_id,
                order_item_id,

                -- Dates (DataCo uses M/D/YYYY H:MM, which a plain CAST can't parse)
                CAST(try_strptime(order_date_dateorders, '{DATACO_TIMESTAMP_FORMAT}') AS DATE)
                    AS order_date,
                CAST(try_strptime(shipping_date_dateorders, '{DATACO_TIMESTAMP_FORMAT}') AS DATE)
                    AS ship_date,

                -- Customer
                customer_id,
                customer_segment,
                TRIM(customer_city) AS customer_city,
                TRIM(customer_state) AS customer_state,
                TRIM(customer_country) AS customer_country,

                -- Product (raw names have trailing spaces, e.g. 'Cameras ')
                TRIM(product_name) AS product_name,
                TRIM(category_name) AS category_name,
                TRIM(department_name) AS department_name,
                product_price AS unit_price,

                -- Order details
                order_status,
                order_status NOT IN ({non_demand}) AS is_valid_demand,
                order_item_quantity AS quantity,
                order_item_total AS revenue,
                order_item_discount_rate AS discount_rate,
                order_item_profit_ratio AS profit_ratio,
                order_profit_per_order AS profit,

                -- Shipping
                TRIM(shipping_mode) AS shipping_mode,
                days_for_shipping_real AS actual_ship_days,
                days_for_shipment_scheduled AS planned_ship_days,
                late_delivery_risk,
                TRIM(delivery_status) AS delivery_status,

                -- Market
                TRIM(market) AS market,
                -- DataCo has stray/double spaces, e.g. 'South of  USA '
                regexp_replace(TRIM(order_region), '\s+', ' ', 'g') AS order_region

            FROM orders_raw
            WHERE order_id IS NOT NULL
        """)

        row_count, null_dates, invalid = con.execute("""
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE order_date IS NULL),
                COUNT(*) FILTER (WHERE NOT is_valid_demand)
            FROM orders
        """).fetchone()
        print(
            f"Created orders table with {row_count:,} rows "
            f"({invalid:,} canceled/fraud flagged, {null_dates:,} unparseable dates)"
        )
        if null_dates:
            raise ValueError(f"{null_dates:,} order dates failed to parse — check the date format")
        return row_count
    finally:
        con.close()


if __name__ == "__main__":
    ingest_supply_chain()
    create_clean_orders_table()
