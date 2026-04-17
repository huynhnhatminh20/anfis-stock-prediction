"""Statistical comparison tests for forecast residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class StatisticalTestResult:
    statistic: float
    p_value: float
    significant: bool
    test_name: str


def diebold_mariano_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    horizon: int = 1,
    power: int = 2,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Compare two forecast series with a Diebold-Mariano style test."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    forecast_a = np.asarray(pred_a, dtype=float).reshape(-1)
    forecast_b = np.asarray(pred_b, dtype=float).reshape(-1)
    _validate_lengths(actual, forecast_a, forecast_b)

    errors_a = actual - forecast_a
    errors_b = actual - forecast_b
    loss_diff = np.abs(errors_a) ** power - np.abs(errors_b) ** power

    mean_diff = np.mean(loss_diff)
    gamma0 = _autocovariance(loss_diff, lag=0)
    variance = gamma0
    for lag in range(1, horizon):
        gamma_lag = _autocovariance(loss_diff, lag=lag)
        variance += 2 * gamma_lag

    variance = max(variance / len(loss_diff), 1e-12)
    statistic = float(mean_diff / np.sqrt(variance))
    p_value = float(2 * (1 - stats.norm.cdf(abs(statistic))))
    return StatisticalTestResult(
        statistic=statistic,
        p_value=p_value,
        significant=bool(p_value < alpha),
        test_name="diebold_mariano",
    )


def wilcoxon_signed_rank_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Non-parametric paired test on absolute forecast errors."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    forecast_a = np.asarray(pred_a, dtype=float).reshape(-1)
    forecast_b = np.asarray(pred_b, dtype=float).reshape(-1)
    _validate_lengths(actual, forecast_a, forecast_b)

    errors_a = np.abs(actual - forecast_a)
    errors_b = np.abs(actual - forecast_b)
    statistic, p_value = stats.wilcoxon(errors_a, errors_b, zero_method="wilcox")
    return StatisticalTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        test_name="wilcoxon_signed_rank",
    )


def _autocovariance(series: np.ndarray, lag: int) -> float:
    centered = series - np.mean(series)
    if lag == 0:
        return float(np.dot(centered, centered) / len(centered))
    return float(np.dot(centered[lag:], centered[:-lag]) / len(centered))


def _validate_lengths(*arrays: np.ndarray) -> None:
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("All series must have the same length")
