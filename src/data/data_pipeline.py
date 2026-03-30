"""End-to-end data pipeline for collection, feature engineering, and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data_collector import DataCollector
from .data_preprocessor import DataPreprocessor
from .feature_engineer import FeatureEngineer


@dataclass
class PipelineOutput:
    full: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class StockDataPipeline:
    """Chain collector + feature engineering + preprocessing into one interface."""

    def __init__(
        self,
        source: str = "VCI",
        raw_dir: Path = Path("data/raw"),
        processed_dir: Path = Path("data/processed"),
    ) -> None:
        self.collector = DataCollector(source=source, output_dir=raw_dir)
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fit_transform(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        save_processed: bool = True,
    ) -> PipelineOutput:
        """Run full pipeline and return scaled train/val/test splits."""
        raw_df = self.collector.fetch_symbol(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
            save_csv=True,
        )
        return self.fit_transform_from_df(raw_df, symbol=symbol, save_processed=save_processed)

    def fit_transform_from_df(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        save_processed: bool = True,
    ) -> PipelineOutput:
        feat_df = self.feature_engineer.transform(df)
        feat_df = feat_df.dropna().reset_index(drop=True)

        scaled_df = self.preprocessor.fit_transform(feat_df)
        split = self.preprocessor.split(scaled_df)

        if save_processed:
            symbol_name = symbol.upper()
            full_path = self.processed_dir / f"{symbol_name}_processed.csv"
            train_path = self.processed_dir / f"{symbol_name}_train.csv"
            val_path = self.processed_dir / f"{symbol_name}_val.csv"
            test_path = self.processed_dir / f"{symbol_name}_test.csv"

            scaled_df.to_csv(full_path, index=False)
            split.train.to_csv(train_path, index=False)
            split.val.to_csv(val_path, index=False)
            split.test.to_csv(test_path, index=False)

        return PipelineOutput(
            full=scaled_df,
            train=split.train,
            val=split.val,
            test=split.test,
        )

    def inverse_transform_close(self, scaled_close_series: pd.Series) -> pd.Series:
        """Convert scaled close predictions back to original price."""
        values = self.preprocessor.inverse_transform_close(scaled_close_series.to_numpy())
        return pd.Series(values, index=scaled_close_series.index, name="close_inverse")

