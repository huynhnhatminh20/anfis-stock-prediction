"""Feature engineering for stock time-series technical indicators."""

from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    """Build technical features required by the ANFIS stock project."""

    def __init__(self, price_col: str = "close", volume_col: str = "volume") -> None:
        self.price_col = price_col
        self.volume_col = volume_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MA(5/20), RSI(14), MACD, Bollinger Bands."""
        if self.price_col not in df.columns:
            raise ValueError(f"Missing price column: {self.price_col}")

        out = df.copy()
        close = out[self.price_col].astype(float)

        # Moving averages
        out["ma5"] = close.rolling(window=5, min_periods=5).mean()
        out["ma20"] = close.rolling(window=20, min_periods=20).mean()

        # RSI(14) - Wilder smoothing
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        out["rsi14"] = 100 - (100 / (1 + rs))

        # MACD = EMA12 - EMA26, signal(9), histogram
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        out["macd"] = ema12 - ema26
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
        out["macd_hist"] = out["macd"] - out["macd_signal"]

        # Bollinger Bands(20, 2 sigma)
        rolling_mean = close.rolling(window=20, min_periods=20).mean()
        rolling_std = close.rolling(window=20, min_periods=20).std(ddof=0)
        out["bb_mid"] = rolling_mean
        out["bb_upper"] = rolling_mean + 2 * rolling_std
        out["bb_lower"] = rolling_mean - 2 * rolling_std

        return out
