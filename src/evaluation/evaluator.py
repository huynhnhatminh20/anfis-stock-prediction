"""Evaluation utilities for forecast experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.evaluation.compare_models import calculate_metrics


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    split_name: str
    metrics: dict[str, float]
    predictions: pd.DataFrame


def evaluate_predictions(
    actual: pd.Series,
    predicted: pd.Series,
    split_name: str,
) -> EvaluationResult:
    frame = pd.DataFrame(
        {
            "actual": actual.to_numpy(),
            "predicted": predicted.to_numpy(),
            "residual": actual.to_numpy() - predicted.to_numpy(),
        }
    )
    metrics = calculate_metrics(frame["actual"].to_numpy(), frame["predicted"].to_numpy())
    return EvaluationResult(split_name=split_name, metrics=metrics, predictions=frame)


def save_evaluation(result: EvaluationResult, output_dir: str | Path) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = target_dir / f"{result.split_name}_metrics.csv"
    predictions_path = target_dir / f"{result.split_name}_predictions.csv"

    pd.DataFrame([{**{"split": result.split_name}, **result.metrics}]).to_csv(metrics_path, index=False)
    result.predictions.to_csv(predictions_path, index=False)
    return metrics_path, predictions_path
