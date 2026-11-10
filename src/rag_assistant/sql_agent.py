"""
Natural-language to SQL agent.

Translates user questions about structured data into DuckDB queries,
runs them in a sandboxed read-only connection, and summarizes the
results. Uses Gemini as the LLM backbone.

Safety: generated SQL must be a single SELECT statement. It runs on a
read-only connection with external access disabled (no reading or
writing local files, no network), the configuration locked so the query
can't re-enable it, and results capped at MAX_RESULT_ROWS.
"""

import duckdb
import pandas as pd
from langchain_core.prompts import PromptTemplate

from src.utils.config import DUCKDB_PATH
from src.utils.llm import get_llm, response_text

MAX_RESULT_ROWS = 200

# Schema description for the LLM
SCHEMA_CONTEXT = """
You are a SQL analyst. Write DuckDB-compatible SQL to answer the user's question.

Data covers orders from 2015-01-01 to 2018-01-31.

Available tables:

1. orders — Individual order lines (one row per item in an order)
   Columns: order_id, order_item_id, order_date, ship_date, customer_id, customer_segment,
   customer_city, customer_state, customer_country, product_name, category_name,
   department_name, unit_price, order_status, is_valid_demand (FALSE for canceled/fraud),
   quantity, revenue, discount_rate, profit_ratio, profit, shipping_mode,
   actual_ship_days, planned_ship_days, late_delivery_risk (1 = late), delivery_status,
   market, order_region
   Note: filter on is_valid_demand for demand, sales or revenue questions.

2. daily_demand — Daily demand by category (one row per category per day, zero-filled;
   excludes canceled/fraud orders)
   Columns: order_date, category_name, department_name, order_count, order_lines, total_units,
   total_revenue, avg_unit_price, avg_discount, late_delivery_count, unique_customers,
   is_filled (TRUE on days with no orders)

3. daily_demand_department — Same as daily_demand, summed per department
   Columns: order_date, department_name, order_count, order_lines, total_units,
   total_revenue, late_delivery_count, is_filled

4. demand_features — ML feature table (category level) with lags and rolling averages
   Columns: order_date, category_name, is_filled, total_units, total_revenue, order_count,
   day_of_week (0 = Sunday), month, quarter, is_weekend, lag_1d_units, lag_7d_units,
   lag_14d_units, lag_28d_units, rolling_7d_avg_units, rolling_14d_avg_units,
   rolling_28d_avg_units, rolling_14d_std_units

5. inventory_risk (VIEW) — Risk scores by category
   Columns: category_name, department_name, total_orders, total_order_lines, total_late_deliveries,
   late_delivery_pct, avg_daily_units, demand_volatility, avg_discount_rate, risk_score

Rules:
- Write ONLY one SELECT query (a WITH clause is fine), no explanations
- Use DuckDB syntax (it's PostgreSQL-like)
- Always LIMIT results to 50 rows unless the user asks for all
- For date filters, use ISO format: '2018-01-15'
- Round decimals to 2 places
- Return the SQL wrapped in ```sql``` markers
"""

SUMMARY_PROMPT = """You are an operations analytics assistant. A SQL query was run to answer
the user's question. Using ONLY the query results below, answer the question in 1-3 sentences.
Mention the key numbers. If the results look empty or don't answer the question, say so.

Question: {question}

SQL:
{sql}

Results ({row_count} rows{truncated}):
{table}"""


class UnsafeQueryError(ValueError):
    """Raised when generated SQL is not a single read-only SELECT."""


def extract_sql(text: str) -> str:
    """Pull the SQL out of a ```sql``` block (or return the text as-is)."""
    if "```sql" in text:
        text = text.split("```sql")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip().rstrip(";").strip()


def validate_sql(sql: str) -> str:
    """Ensure `sql` is exactly one SELECT statement. Returns the cleaned SQL."""
    sql = sql.strip().rstrip(";").strip()
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as e:
        raise UnsafeQueryError(f"Could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeQueryError(f"Expected exactly one statement, got {len(statements)}")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise UnsafeQueryError(f"Only SELECT queries are allowed, got {statements[0].type.name}")
    return sql


def run_safe_query(sql: str, db_path: str | None = None) -> tuple[pd.DataFrame, bool]:
    """
    Run a validated SELECT in a sandboxed connection.

    Returns (results, truncated) where truncated is True if more than
    MAX_RESULT_ROWS rows matched.
    """
    sql = validate_sql(sql)
    con = duckdb.connect(
        db_path or DUCKDB_PATH,
        read_only=True,
        config={"enable_external_access": False, "lock_configuration": True},
    )
    try:
        df = con.execute(f"SELECT * FROM ({sql}) AS q LIMIT {MAX_RESULT_ROWS + 1}").fetchdf()
    finally:
        con.close()
    return df.head(MAX_RESULT_ROWS), len(df) > MAX_RESULT_ROWS


def text_to_sql(question: str, previous_sql: str | None = None, error: str | None = None) -> str:
    """
    Convert a natural-language question to a SQL query using Gemini.

    If `previous_sql` and `error` are given, the model is asked to fix that query.
    Returns the SQL string (not executed yet).
    """
    llm = get_llm("answer", temperature=0.0)

    template = SCHEMA_CONTEXT + "\n\nQuestion: {question}\n"
    values = {"question": question}
    if previous_sql and error:
        template += (
            "\nThis query failed:\n```sql\n{previous_sql}\n```\n"
            "Error: {error}\nWrite a corrected query.\n"
        )
        values.update(previous_sql=previous_sql, error=error)
    template += "\nSQL:"

    prompt = PromptTemplate(input_variables=list(values), template=template)
    response = llm.invoke(prompt.format(**values))
    return extract_sql(response_text(response))


def summarize_results(question: str, sql: str, df: pd.DataFrame, truncated: bool) -> str:
    """Ask Gemini for a short natural-language answer grounded in the results."""
    llm = get_llm("answer", temperature=0.1)
    prompt = SUMMARY_PROMPT.format(
        question=question,
        sql=sql,
        row_count=len(df),
        truncated=f", showing first {MAX_RESULT_ROWS}" if truncated else "",
        table=df.head(50).to_markdown(index=False) if len(df) else "(no rows)",
    )
    return response_text(llm.invoke(prompt)).strip()


def ask_data(question: str, max_attempts: int = 2, summarize: bool = True) -> dict:
    """
    End-to-end: question -> SQL -> validate -> execute -> summarize.

    If a query fails (invalid or rejected SQL), the error is sent back to the
    model for one more attempt. Returns a dict with the SQL, a pandas DataFrame
    of results, a short summary, and any error.
    """
    sql, error = None, None
    for _ in range(max_attempts):
        sql = text_to_sql(question, previous_sql=sql, error=error)
        try:
            df, truncated = run_safe_query(sql)
        except (UnsafeQueryError, duckdb.Error) as e:
            error = str(e)
            continue

        return {
            "sql": sql,
            "results": df,
            "truncated": truncated,
            "summary": summarize_results(question, sql, df, truncated) if summarize else None,
            "error": None,
            "row_count": len(df),
        }

    return {
        "sql": sql,
        "results": None,
        "truncated": False,
        "summary": None,
        "error": error,
        "row_count": 0,
    }
