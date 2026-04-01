"""Data collection utilities for Vietnamese stock OHLCV data via vnstock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


SUPPORTED_SOURCES = {"KBS", "VCI", "MSN", "DNSE", "BINANCE", "FMP", "FMARKET"}


@dataclass
class DataCollector:
    """Collect OHLCV data from vnstock and persist raw CSV files."""

    source: str = "VCI"
    output_dir: Path = Path("data/raw")

    def __post_init__(self) -> None:
        self.source = self.source.upper()
        if self.source not in SUPPORTED_SOURCES:
            raise ValueError(f"source must be one of {sorted(SUPPORTED_SOURCES)}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_symbol(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV for one ticker symbol."""
        try:
            from vnstock import Quote
        except Exception as exc:  # pragma: no cover - optional dependency at runtime
            raise ImportError("vnstock is required. Install with: pip install vnstock") from exc

        quote = Quote(symbol=symbol.upper(), source=self.source.lower())
        df = quote.history(start=start, end=end, interval=interval)
        if df is None or len(df) == 0:
            raise ValueError(f"No data returned for symbol={symbol}")

        df = self._normalize_columns(df)
        df["symbol"] = symbol.upper()
        df = df.sort_values("date").reset_index(drop=True)

        if save_csv:
            out_path = self.output_dir / f"{symbol.upper()}_{start}_{end}_{interval}.csv"
            df.to_csv(out_path, index=False)

        return df

    def fetch_many(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        interval: str = "1d",
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """Fetch and concat OHLCV for many symbols."""
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            frames.append(self.fetch_symbol(symbol, start, end, interval=interval, save_csv=save_csv))
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "time": "date",
            "datetime": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        # Lower-case and strip spaces for robust mapping across vnstock versions.
        normalized = {col: col.strip().lower() for col in df.columns}
        df = df.rename(columns=normalized)

        mapped: dict[str, str] = {}
        for col in df.columns:
            if col in rename_map:
                mapped[col] = rename_map[col]
        df = df.rename(columns=mapped)

        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        df = df[required]
        df["date"] = pd.to_datetime(df["date"])
        return df


def collect_to_csv(
    symbol: str,
    start: str,
    end: str,
    source: str = "VCI",
    interval: str = "1d",
    output_dir: Optional[str] = None,
) -> Path:
    """Convenience function for one-shot collection."""
    collector = DataCollector(source=source, output_dir=Path(output_dir) if output_dir else Path("data/raw"))
    collector.fetch_symbol(symbol=symbol, start=start, end=end, interval=interval, save_csv=True)
    return collector.output_dir
