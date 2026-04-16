"""Project configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetConfig:
    symbol: str = "VNM"
    start_date: str = "2020-01-01"
    end_date: str = "2026-03-30"
    interval: str = "1d"
    source: str = "VCI"
    target_column: str = "close"
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ma5",
            "ma20",
            "rsi14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_mid",
            "bb_upper",
            "bb_lower",
        ]
    )


@dataclass(slots=True)
class AnfisConfig:
    input_dim: int = 4
    num_memberships: int = 2
    learning_rate: float = 0.01
    epochs: int = 50
    batch_size: int = 32
    patience: int = 8
    min_delta: float = 1e-4
    membership_type: str = "gaussian"
    use_hybrid_learning: bool = True
    l2_weight_decay: float = 0.0
    random_seed: int = 42


@dataclass(slots=True)
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: ["mae", "rmse", "mape"])
    significance_level: float = 0.05


@dataclass(slots=True)
class ProjectConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    anfis: AnfisConfig = field(default_factory=AnfisConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(path: str | Path = "configs/config.yaml") -> ProjectConfig:
    """Load nested configuration from a YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyYAML is required to load configs/config.yaml") from exc

    config_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return ProjectConfig(
        dataset=_build_dataclass(DatasetConfig, raw.get("dataset", {})),
        anfis=_build_dataclass(AnfisConfig, raw.get("anfis", {})),
        evaluation=_build_dataclass(EvaluationConfig, raw.get("evaluation", {})),
    )


def _build_dataclass(cls: type[Any], values: dict[str, Any]) -> Any:
    allowed = {field.name for field in cls.__dataclass_fields__.values()}
    filtered = {key: value for key, value in values.items() if key in allowed}
    return cls(**filtered)
