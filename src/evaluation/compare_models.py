"""Model comparison helpers for ANFIS, ARIMA, and MLP outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ModelComparisonResult:
    metrics_table: pd.DataFrame
    predictions_table: pd.DataFrame
    best_model: str


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def compare_model_predictions(
    y_true: np.ndarray,
    predictions_by_model: Mapping[str, np.ndarray],
) -> ModelComparisonResult:
    """Build a tidy comparison table from model predictions."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    metrics_rows: list[dict[str, float | str]] = []
    prediction_columns: dict[str, np.ndarray] = {"actual": actual}

    for model_name, raw_predictions in predictions_by_model.items():
        predictions = np.asarray(raw_predictions, dtype=float).reshape(-1)
        if len(predictions) != len(actual):
            raise ValueError(f"Prediction length mismatch for model '{model_name}'")
        prediction_columns[model_name] = predictions
        metrics_rows.append({"model": model_name, **calculate_metrics(actual, predictions)})

    metrics_table = pd.DataFrame(metrics_rows).sort_values(by=["rmse", "mae", "mape"]).reset_index(drop=True)
    predictions_table = pd.DataFrame(prediction_columns)
    best_model = str(metrics_table.iloc[0]["model"])
    return ModelComparisonResult(
        metrics_table=metrics_table,
        predictions_table=predictions_table,
        best_model=best_model,
    )
