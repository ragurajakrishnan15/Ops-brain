"""
End-to-end tests for the ingest + transform SQL.

Builds a small CSV in the raw DataCo format (original column names and
'M/D/YYYY H:MM' timestamps), runs the real pipeline into a temporary
DuckDB file, and checks the resulting tables.
"""

from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from src.data_pipeline import ingest, transforms

START = date(2017, 1, 1)
N_DAYS = 40
GAP_DAY = 10  # Men's Footwear has no orders on this day
WAREHOUSE = {}  # path of the module's temporary warehouse


def _row(order_id, item_id, day, category, department, qty, status="COMPLETE", late=0):
    d = START + timedelta(days=day)
    ts = f"{d.month}/{d.day}/{d.year} 14:05"
    return {
        "Type": "DEBIT",
        "Days for shipping (real)": 3,
        "Days for shipment (scheduled)": 4,
        "Delivery Status": "Late delivery" if late else "Shipping on time",
        "Late_delivery_risk": late,
        "Category Name": category,
        "Customer City": "Caguas",
        "Customer Country": "Puerto Rico",
        "Customer Email": "XXXXXXXXX",
        "Customer Fname": "Ann",
        "Customer Id": order_id % 7,
        "Customer Lname": "Lee",
        "Customer Password": "XXXXXXXXX",
        "Customer Segment": "Consumer",
        "Customer State": "PR",
        "Customer Street": "1 Main St",
        "Department Name": department,
        "Market": "LATAM",
        "order date (DateOrders)": ts,
        "Order Id": order_id,
        "Order Item Discount Rate": 0.05,
        "Order Item Id": item_id,
        "Order Item Profit Ratio": 0.2,
        "Order Item Quantity": qty,
        "Order Item Total": qty * 10.0,
        "Order Profit Per Order": qty * 2.0,
        "Order Region": "Caribbean",
        "Order Status": status,
        "Product Name": f"{category} item",
        "Product Price": 10.0,
        "shipping date (DateOrders)": ts,
        "Shipping Mode": "Standard Class",
    }


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    rows, order_id, item_id = [], 1, 1
    for day in range(N_DAYS):
        if day != GAP_DAY:
            # Demand = day + 1 units, every 5th order is late
            rows.append(
                _row(
                    order_id,
                    item_id,
                    day,
                    "Men's Footwear",
                    "Apparel",
                    day + 1,
                    late=int(day % 5 == 0),
                )
            )
            order_id += 1
            item_id += 1
        # Electronics is split across two departments; Footwear is the majority
        dept = "Footwear" if day % 3 else "Technology"
        rows.append(
            _row(order_id, item_id, day, "Electronics ", dept, 2)
        )  # raw names can have trailing spaces
        order_id += 1
        item_id += 1
    # A canceled order that must not count as demand
    rows.append(_row(order_id, item_id, 5, "Men's Footwear", "Apparel", 100, status="CANCELED"))

    csv_path = tmp / "dataco.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="latin-1")
    db_path = str(tmp / "warehouse.duckdb")

    mp = pytest.MonkeyPatch()
    mp.setattr(ingest, "DUCKDB_PATH", db_path)
    mp.setattr(transforms, "DUCKDB_PATH", db_path)
    ingest.ingest_supply_chain(csv_path)
    ingest.create_clean_orders_table()
    transforms.build_daily_demand_table()
    transforms.build_feature_table()
    transforms.build_department_tables()
    transforms.build_inventory_risk_view()
    mp.undo()

    WAREHOUSE["path"] = db_path
    con = duckdb.connect(db_path, read_only=True)
    yield con
    con.close()


def test_dates_parse(warehouse):
    nulls, first = warehouse.execute(
        "SELECT COUNT(*) FILTER (WHERE order_date IS NULL), MIN(order_date) FROM orders"
    ).fetchone()
    assert nulls == 0
    assert first == START


def test_pii_columns_dropped(warehouse):
    cols = {r[0] for r in warehouse.execute("DESCRIBE orders_raw").fetchall()}
    assert not cols & set(ingest.PII_COLUMNS)


def test_canceled_orders_excluded(warehouse):
    units = warehouse.execute(
        """
        SELECT total_units FROM daily_demand
        WHERE category_name = 'Men''s Footwear' AND order_date = ?
    """,
        [START + timedelta(days=5)],
    ).fetchone()[0]
    assert units == 6  # day 5 -> 6 units; the 100-unit canceled order is ignored


def test_calendar_is_densified(warehouse):
    units, filled = warehouse.execute(
        """
        SELECT total_units, is_filled FROM daily_demand
        WHERE category_name = 'Men''s Footwear' AND order_date = ?
    """,
        [START + timedelta(days=GAP_DAY)],
    ).fetchone()
    assert units == 0
    assert filled
    per_category = warehouse.execute(
        "SELECT category_name, COUNT(*) FROM daily_demand GROUP BY 1"
    ).fetchall()
    assert all(n == N_DAYS for _, n in per_category)


def test_one_department_per_category(warehouse):
    rows = warehouse.execute("""
        SELECT category_name, COUNT(DISTINCT department_name), ANY_VALUE(department_name)
        FROM daily_demand GROUP BY 1
    """).fetchall()
    by_cat = {cat: (n, dept) for cat, n, dept in rows}
    assert by_cat["Electronics"] == (1, "Footwear")


def test_features_use_only_past_days(warehouse):
    df = warehouse.execute("""
        SELECT order_date, total_units, lag_1d_units, lag_7d_units, rolling_7d_avg_units
        FROM demand_features
        WHERE category_name = 'Men''s Footwear'
        ORDER BY order_date
    """).fetchdf()
    daily = (
        warehouse.execute("""
        SELECT order_date, total_units FROM daily_demand
        WHERE category_name = 'Men''s Footwear' ORDER BY order_date
    """)
        .fetchdf()
        .set_index("order_date")["total_units"]
    )

    assert len(df) == N_DAYS - 28
    for _, row in df.iterrows():
        t = pd.Timestamp(row["order_date"])
        past = daily.loc[t - pd.Timedelta(days=7) : t - pd.Timedelta(days=1)]
        assert row["lag_1d_units"] == daily.loc[t - pd.Timedelta(days=1)]
        assert row["lag_7d_units"] == daily.loc[t - pd.Timedelta(days=7)]
        assert row["rolling_7d_avg_units"] == pytest.approx(past.mean())


def test_feature_cols_exist_and_exclude_same_day_values(warehouse):
    from src.forecasting.train import FEATURE_COLS, TARGET_COL

    cols = {r[0] for r in warehouse.execute("DESCRIBE demand_features").fetchall()}
    assert set(FEATURE_COLS) <= cols
    same_day = {TARGET_COL, "total_revenue", "order_count", "is_filled"}
    assert not same_day & set(FEATURE_COLS)


def test_inventory_risk_percentages(warehouse):
    rows = warehouse.execute(
        "SELECT category_name, late_delivery_pct, risk_score FROM inventory_risk"
    ).fetchall()
    risk = {cat: (pct, score) for cat, pct, score in rows}
    # Footwear: 39 order lines, late on days 0,5,15,...,35 (day 10 is the gap) = 7
    assert risk["Men's Footwear"][0] == round(7 * 100 / 39, 1)
    assert risk["Electronics"][0] == 0.0
    assert all(0 <= pct <= 100 and 0 <= score <= 100 for pct, score in risk.values())


def test_python_features_match_sql(warehouse):
    """features_for_day() must reproduce the SQL feature table exactly."""
    from src.forecasting.features import FEATURE_COLS, features_for_day

    for table, key, name in [
        ("demand_features", "category_name", "Men's Footwear"),
        ("department_features", "department_name", "Footwear"),
    ]:
        daily_table = "daily_demand" if table == "demand_features" else "daily_demand_department"
        units = (
            warehouse.execute(
                f"SELECT order_date, total_units FROM {daily_table} WHERE {key} = ? ORDER BY 1",
                [name],
            )
            .fetchdf()
            .set_index("order_date")["total_units"]
            .astype(float)
        )
        sql = warehouse.execute(
            f"SELECT * FROM {table} WHERE {key} = ? ORDER BY order_date", [name]
        ).fetchdf()
        assert len(sql) > 0
        for _, row in sql.iterrows():
            day = pd.Timestamp(row["order_date"])
            history = units[units.index < day].to_numpy()
            py = features_for_day(history, day)
            for col in FEATURE_COLS:
                assert py[col] == pytest.approx(float(row[col])), (table, day.date(), col)


def test_department_series(warehouse):
    rows = dict(
        warehouse.execute("""
        SELECT department_name, SUM(total_units) FROM daily_demand_department GROUP BY 1
    """).fetchall()
    )
    category_totals = dict(
        warehouse.execute("""
        SELECT department_name, SUM(total_units) FROM daily_demand GROUP BY 1
    """).fetchall()
    )
    assert rows == category_totals
    # Apparel (Men's Footwear) has a gap day; it's filled with zero at department level too
    filled = warehouse.execute(
        """
        SELECT total_units, is_filled FROM daily_demand_department
        WHERE department_name = 'Apparel' AND order_date = ?
    """,
        [START + timedelta(days=GAP_DAY)],
    ).fetchone()
    assert filled == (0, True)


def test_quality_checks(warehouse, monkeypatch):
    from src.data_pipeline import quality

    monkeypatch.setattr("src.utils.db.DUCKDB_PATH", WAREHOUSE["path"])
    checks = quality.run_quality_checks()
    assert len(checks) == len(quality.CHECKS)
    by_rule = checks.set_index("rule")
    # The synthetic orders are clean
    assert (checks.loc[checks["section"] != "Outliers", "status"] == "pass").all()
    assert by_rule.loc["Order date is a valid date", "total_rows"] == 2 * N_DAYS
    # Outlier rule cross-checked with pandas: |units - mean(prev 28)| > 3 * std(prev 28)
    daily = warehouse.execute(
        "SELECT category_name, order_date, total_units FROM daily_demand ORDER BY 1, 2"
    ).fetchdf()
    expected = 0
    for _, g in daily.groupby("category_name"):
        u = g["total_units"].astype(float).reset_index(drop=True)
        mean = u.shift(1).rolling(28, min_periods=1).mean()
        std = u.shift(1).rolling(28, min_periods=2).std()
        expected += int(((std > 0) & ((u - mean).abs() > 3 * std)).sum())
    rule = "Daily category demand within 3 std of its rolling 28-day mean"
    assert by_rule.loc[rule, "failing_rows"] == expected

    profile = quality.table_profile().set_index("table")
    assert profile.loc["daily_demand", "rows"] == 2 * N_DAYS
    nulls = quality.null_rates()
    assert nulls.loc[nulls["column"] == "order_date", "null_rows"].iloc[0] == 0
