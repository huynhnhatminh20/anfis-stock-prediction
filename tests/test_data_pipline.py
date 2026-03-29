from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_pipeline import StockDataPipeline
from src.data.data_preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer


def _make_sample_df(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    base = np.linspace(100, 150, n)
    # Use larger oscillation so the series has both gains and losses.
    noise = 6 * np.sin(np.linspace(0, 20, n))
    close = base + noise

    df = pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, n),
        }
    )

    # Inject NaN for missing-value handling tests
    df.loc[0, "open"] = np.nan
    df.loc[10, "close"] = np.nan
    df.loc[20, "volume"] = np.nan
    return df


def test_feature_engineer_adds_required_columns() -> None:
    df = _make_sample_df()
    fe = FeatureEngineer()
    out = fe.transform(df)

    required_cols = {
        "ma5",
        "ma20",
        "rsi14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_mid",
        "bb_upper",
        "bb_lower",
    }
    assert required_cols.issubset(set(out.columns))


def test_preprocessor_split_ratio_70_15_15() -> None:
    df = _make_sample_df(100)
    fe = FeatureEngineer()
    with_features = fe.transform(df).dropna().reset_index(drop=True)

    pre = DataPreprocessor()
    scaled = pre.fit_transform(with_features)
    split = pre.split(scaled)

    n = len(scaled)
    assert len(split.train) == int(n * 0.70)
    assert len(split.val) == int(n * 0.15)
    assert len(split.train) + len(split.val) + len(split.test) == n


def test_pipeline_fit_transform_from_df_and_inverse_close() -> None:
    df = _make_sample_df(140)
    pipe = StockDataPipeline(processed_dir=Path("data/processed"))

    output = pipe.fit_transform_from_df(df, symbol="TEST", save_processed=False)

    assert not output.full.empty
    assert not output.train.empty
    assert not output.val.empty
    assert not output.test.empty

    for col in ["open", "high", "low", "close"]:
        assert output.full[col].between(0, 1).all()

    recovered = pipe.inverse_transform_close(output.full["close"])
    assert recovered.notna().all()
