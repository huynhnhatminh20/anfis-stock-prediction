"""Command-line runner for the ANFIS experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from src.config import load_config
from src.data.data_pipeline import StockDataPipeline
from src.evaluation.evaluator import evaluate_predictions, save_evaluation
from src.models.anfis_model import ANFIS
from src.models.anfis_train import AnfisTrainer


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    raw_path = Path(args.raw_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path, parse_dates=["date"])
    pipeline = StockDataPipeline(source=config.dataset.source)
    pipeline_output = pipeline.fit_transform_from_df(raw_df, symbol=config.dataset.symbol, save_processed=False)

    feature_columns = [col for col in config.dataset.feature_columns if col in pipeline_output.full.columns]
    if config.dataset.target_column in feature_columns:
        feature_columns.remove(config.dataset.target_column)

    train_features, train_targets = build_supervised_split(pipeline_output.train, feature_columns, config.dataset.target_column)
    val_features, val_targets = build_supervised_split(pipeline_output.val, feature_columns, config.dataset.target_column)
    test_features, test_targets = build_supervised_split(pipeline_output.test, feature_columns, config.dataset.target_column)

    anfis_config = config.anfis
    model = ANFIS(input_dim=len(feature_columns), num_memberships=anfis_config.num_memberships)
    trainer = AnfisTrainer(anfis_config)
    artifacts = trainer.fit(model, train_features, train_targets, val_features, val_targets)

    train_pred = trainer.predict(artifacts.model, train_features).squeeze(1).cpu().numpy()
    val_pred = trainer.predict(artifacts.model, val_features).squeeze(1).cpu().numpy()
    test_pred = trainer.predict(artifacts.model, test_features).squeeze(1).cpu().numpy()

    results = [
        make_result("train", pipeline, train_targets, train_pred),
        make_result("val", pipeline, val_targets, val_pred),
        make_result("test", pipeline, test_targets, test_pred),
    ]

    summary_rows: list[dict[str, float | str]] = []
    for result in results:
        save_evaluation(result, output_dir)
        summary_rows.append({"split": result.split_name, **result.metrics})

    summary_path = output_dir / "anfis_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    metadata_path = output_dir / "anfis_run.json"
    metadata = {
        "config_path": str(Path(args.config).resolve()),
        "raw_csv": str(raw_path.resolve()),
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
        "num_rules": artifacts.model.num_rules,
        "best_epoch": artifacts.history.best_epoch,
        "best_val_loss": artifacts.history.best_val_loss,
        "stopped_early": artifacts.history.stopped_early,
        "history": asdict(artifacts.history),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved summary to: {summary_path}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


def build_supervised_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    supervised = df.copy()
    supervised["target_next_close"] = supervised[target_column].shift(-1)
    supervised = supervised.dropna(subset=["target_next_close"]).reset_index(drop=True)

    features = torch.tensor(supervised[feature_columns].to_numpy(), dtype=torch.float32)
    targets = torch.tensor(supervised["target_next_close"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    return features, targets


def make_result(
    split_name: str,
    pipeline: StockDataPipeline,
    targets: torch.Tensor,
    predictions: list[float] | torch.Tensor | pd.Series | pd.Index | object,
):
    actual_series = pd.Series(pipeline.inverse_transform_close(pd.Series(targets.squeeze(1).cpu().numpy())))
    predicted_series = pd.Series(pipeline.inverse_transform_close(pd.Series(predictions)))
    return evaluate_predictions(actual_series, predicted_series, split_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANFIS experiment on processed stock data.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--raw-csv", default="data/raw/VNM_2020-01-01_2026-03-30_1d.csv", help="Raw OHLCV CSV path.")
    parser.add_argument("--output-dir", default="results/anfis", help="Directory for metrics and predictions.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
