"""
SHAP-based model explanations.

Generates feature importance plots and per-prediction explanations
that the planning team can use to understand forecast drivers.

The XGBoost model predicts the deviation from the trailing 28-day average,
so SHAP values explain how far each feature moves the forecast away from
that average (in units).
"""

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from src.forecasting.features import BASELINE_COL, FEATURE_COLS, predict_units


def _matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    return X[FEATURE_COLS].to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)


def get_global_importance(model: XGBRegressor, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """
    Compute global SHAP feature importance.

    Returns a DataFrame sorted by mean |SHAP value|, showing which
    features matter most across all predictions.
    """
    shap_values = shap.TreeExplainer(model).shap_values(_matrix(X))

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    )
    importance = importance.sort_values("mean_abs_shap", ascending=False)
    importance["rank"] = range(1, len(importance) + 1)

    return importance


def explain_prediction(model: XGBRegressor, X_row: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Explain a single prediction using SHAP.

    X_row is a one-row DataFrame with FEATURE_COLS. Returns the predicted units,
    the 28-day average it starts from, the model's average correction
    (base_value), and the features that moved it most
    (positive = pushes the forecast up). The pieces add up:
    predicted ≈ baseline_units + base_value + sum(all contributions), before
    clipping at zero.
    """
    X_row = X_row.iloc[[0]] if len(X_row) > 1 else X_row
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(_matrix(X_row))[0]

    contributions = sorted(
        zip(FEATURE_COLS, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return {
        "predicted_value": float(predict_units(model, X_row)[0]),
        "baseline_units": float(X_row[BASELINE_COL].iloc[0]),
        "base_value": float(np.ravel(explainer.expected_value)[0]),
        "top_drivers": [
            {"feature": name, "contribution": float(val)} for name, val in contributions[:top_n]
        ],
    }


def get_shap_summary_plot_data(
    model: XGBRegressor,
    X: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for a summary plot.

    Returns (shap_values, X) to be passed to shap.summary_plot()
    or used in a custom Plotly visualization.
    """
    X = _matrix(X)
    return shap.TreeExplainer(model).shap_values(X), X
