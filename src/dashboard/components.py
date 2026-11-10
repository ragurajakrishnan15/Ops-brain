"""
Reusable chart components for the Streamlit dashboard.

All charts use Plotly for interactivity and consistent styling.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Consistent color palette
COLORS = {
    "primary": "#2563eb",
    "secondary": "#7c3aed",
    "success": "#059669",
    "warning": "#d97706",
    "danger": "#dc2626",
    "gray": "#6b7280",
    "light_blue": "rgba(37, 99, 235, 0.15)",
    "light_purple": "rgba(124, 58, 237, 0.15)",
}


def forecast_chart(df: pd.DataFrame, category: str) -> go.Figure:
    """
    Line chart showing actual vs forecast with prediction intervals.

    Expects forecast_series() output: history (actuals only), backtest
    (actual + forecast on the holdout) and future (forecast only). The shaded
    band shows the 95% prediction interval wherever one exists.
    """
    fig = go.Figure()

    # Prediction interval band, drawn separately for backtest and future
    for i, (_, part) in enumerate(
        df.dropna(subset=["lower_bound", "upper_bound"]).groupby("segment")
    ):
        fig.add_trace(
            go.Scatter(
                x=pd.concat([part["order_date"], part["order_date"][::-1]]),
                y=pd.concat([part["upper_bound"], part["lower_bound"][::-1]]),
                fill="toself",
                fillcolor=COLORS["light_blue"],
                line=dict(color="rgba(0,0,0,0)"),
                name="95% interval",
                legendgroup="interval",
                showlegend=i == 0,
                hoverinfo="skip",
            )
        )

    # Actual values
    actual = df.dropna(subset=["actual"])
    fig.add_trace(
        go.Scatter(
            x=actual["order_date"],
            y=actual["actual"],
            mode="lines",
            name="Actual",
            line=dict(color=COLORS["gray"], width=1.5),
        )
    )

    # Forecast (holdout backtest + future)
    forecast = df.dropna(subset=["forecast"])
    fig.add_trace(
        go.Scatter(
            x=forecast["order_date"],
            y=forecast["forecast"],
            mode="lines",
            name="Forecast",
            line=dict(color=COLORS["primary"], width=2),
        )
    )

    for segment, label in (("backtest", "Holdout"), ("future", "Future")):
        rows = df[df["segment"] == segment]
        if len(rows):
            fig.add_vline(
                x=rows["order_date"].min(), line=dict(color=COLORS["gray"], dash="dot", width=1)
            )
            fig.add_annotation(
                x=rows["order_date"].min(),
                y=1,
                yref="paper",
                text=label,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=11, color=COLORS["gray"]),
            )

    fig.update_layout(
        title=f"Demand Forecast — {category}",
        xaxis_title="Date",
        yaxis_title="Units",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
    )

    return fig


def risk_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap showing inventory risk scores by category and department.
    """
    pivot = df.pivot_table(
        values="risk_score",
        index="category_name",
        columns="department_name",
        aggfunc="mean",
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",  # Red = high risk, Green = low risk
            text=np.round(pivot.values, 1),
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(title="Risk Score"),
        )
    )

    fig.update_layout(
        title="Inventory Risk by Category & Department",
        template="plotly_white",
        height=max(300, len(pivot) * 35 + 100),
        margin=dict(l=150, r=20, t=40, b=80),
    )

    return fig


def metric_card(label: str, value: str, delta: str | None = None) -> str:
    """
    Return HTML for a single metric card (used in st.markdown).
    """
    delta_html = ""
    if delta:
        color = (
            COLORS["success"] if delta.startswith("+") or delta.startswith("-") else COLORS["gray"]
        )
        delta_html = f'<div style="color: {color}; font-size: 14px;">{delta}</div>'

    return f"""
    <div style="background: #f8fafc; border-radius: 8px; padding: 16px; text-align: center;">
        <div style="color: #6b7280; font-size: 13px; text-transform: uppercase;">{label}</div>
        <div style="font-size: 28px; font-weight: 700; color: #111827;">{value}</div>
        {delta_html}
    </div>
    """


FEATURE_LABELS = {
    "day_of_week": "Day of week",
    "month": "Month",
    "quarter": "Quarter",
    "is_weekend": "Weekend",
    "lag_1d_units": "Units yesterday",
    "lag_7d_units": "Units 7 days ago",
    "lag_14d_units": "Units 14 days ago",
    "lag_28d_units": "Units 28 days ago",
    "rolling_7d_avg_units": "7-day average",
    "rolling_14d_avg_units": "14-day average",
    "rolling_28d_avg_units": "28-day average",
    "rolling_14d_std_units": "14-day volatility",
}


def shap_waterfall(explanation: dict, max_features: int = 6) -> go.Figure:
    """
    Waterfall from the 28-day average to the model's prediction.

    `explanation` is explain_prediction() output with all features in
    top_drivers. The first bar is the starting point (28-day average plus the
    model's average correction); each feature then pushes the forecast up
    (blue) or down (red); the remaining features are grouped.
    """
    drivers = explanation["top_drivers"]
    shown, rest = drivers[:max_features], drivers[max_features:]
    start = explanation["baseline_units"] + explanation["base_value"]

    labels = ["Starting point"] + [FEATURE_LABELS.get(d["feature"], d["feature"]) for d in shown]
    values = [start] + [d["contribution"] for d in shown]
    measures = ["absolute"] + ["relative"] * len(shown)
    if rest:
        labels.append(f"{len(rest)} other features")
        values.append(sum(d["contribution"] for d in rest))
        measures.append("relative")
    labels.append("Prediction")
    values.append(0)
    measures.append("total")

    # Zoom the axis to where the forecast actually moves; from zero the feature bars are slivers
    path = np.cumsum(values[:-1])
    lo, hi = (
        float(min(path.min(), explanation["predicted_value"])),
        float(max(path.max(), explanation["predicted_value"])),
    )
    pad = max((hi - lo) * 0.35, 0.5)

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measures,
            y=labels,
            x=values,
            connector={"line": {"color": "#e5e7eb"}},
            decreasing={"marker": {"color": COLORS["danger"]}},
            increasing={"marker": {"color": COLORS["primary"]}},
            totals={"marker": {"color": COLORS["gray"]}},
            textposition="outside",
            cliponaxis=False,
            text=[f"{start:.1f}"]
            + [f"{v:+.2f}" for v in values[1:-1]]
            + [f"{explanation['predicted_value']:.1f}"],
        )
    )

    fig.update_layout(
        title=f"Why the model predicted {explanation['predicted_value']:.1f} units "
        f"(28-day average: {explanation['baseline_units']:.1f})",
        template="plotly_white",
        height=60 + 38 * len(labels),
        margin=dict(l=160, r=70, t=50, b=40),
        xaxis=dict(range=[lo - pad, hi + pad], title="Units (axis zoomed to the change)"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )

    return fig


def importance_bar(importance: pd.DataFrame) -> go.Figure:
    """Horizontal bar of mean |SHAP| per feature (global importance)."""
    df = importance.sort_values("mean_abs_shap")
    fig = go.Figure(
        go.Bar(
            x=df["mean_abs_shap"],
            y=[FEATURE_LABELS.get(f, f) for f in df["feature"]],
            orientation="h",
            marker_color=COLORS["primary"],
            hovertemplate="%{y}: %{x:.2f} units<extra></extra>",
        )
    )
    fig.update_layout(
        title="Average impact on the forecast (mean |SHAP|, units)",
        template="plotly_white",
        height=60 + 28 * len(df),
        margin=dict(l=160, r=20, t=50, b=30),
    )
    return fig


def risk_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Late-delivery rate vs. demand volatility per category.

    Point size is average daily units; color is the risk score. Categories in
    the top right are both unreliable to deliver and hard to forecast.
    """
    cv = (df["demand_volatility"] / df["avg_daily_units"].where(df["avg_daily_units"] > 0)) * 100
    fig = go.Figure(
        go.Scatter(
            x=df["late_delivery_pct"],
            y=cv.clip(upper=100),
            mode="markers",
            marker=dict(
                size=(df["avg_daily_units"].clip(lower=0.1) ** 0.5) * 6 + 6,
                color=df["risk_score"],
                colorscale="RdYlGn_r",
                colorbar=dict(title="Risk score"),
                line=dict(width=0.5, color="white"),
            ),
            text=df["category_name"],
            customdata=df[["department_name", "avg_daily_units", "risk_score"]],
            hovertemplate=(
                "<b>%{text}</b> (%{customdata[0]})<br>Late deliveries: %{x:.1f}%"
                "<br>Volatility (CV): %{y:.0f}%<br>Avg units/day: %{customdata[1]:.1f}"
                "<br>Risk score: %{customdata[2]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Delivery reliability vs. demand volatility (size = avg daily units)",
        xaxis_title="Late delivery rate (%)",
        yaxis_title="Demand volatility (CV %, capped at 100)",
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig
