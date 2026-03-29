"""Preprocessing utilities for stock forecast datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class SplitDataset:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class DataPreprocessor:
    """Handle missing values, scaling, and time-based split."""

    def __init__(
        self,
        price_columns: Sequence[str] = ("open", "high", "low", "close"),
        volume_column: str = "volume",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

        self.price_columns = list(price_columns)
        self.volume_column = volume_column
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.price_scaler = MinMaxScaler()
        self.volume_scaler = StandardScaler()
        self._is_fitted = False

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill then backfill remaining NaN at dataset head."""
        return df.copy().ffill().bfill()

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        data = self.fill_missing(df)
        self._check_columns(data)

        self.price_scaler.fit(data[self.price_columns])
        self.volume_scaler.fit(data[[self.volume_column]])
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        data = self.fill_missing(df)
        self._check_columns(data)

        out = data.copy()
        out[self.price_columns] = self.price_scaler.transform(out[self.price_columns])
        out[[self.volume_column]] = self.volume_scaler.transform(out[[self.volume_column]])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def inverse_transform_close(self, scaled_close: np.ndarray) -> np.ndarray:
        """Invert close price from MinMax scale back to original price scale."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse transform.")

        close_idx = self.price_columns.index("close")
        tmp = np.zeros((len(scaled_close), len(self.price_columns)), dtype=float)
        tmp[:, close_idx] = scaled_close.reshape(-1)
        inv = self.price_scaler.inverse_transform(tmp)
        return inv[:, close_idx]

    def split(self, df: pd.DataFrame) -> SplitDataset:
        """Split time series into train/val/test without shuffling."""
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return SplitDataset(train=train_df, val=val_df, test=test_df)

    def _check_columns(self, df: pd.DataFrame) -> None:
        required = set(self.price_columns + [self.volume_column])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for preprocessing: {missing}")