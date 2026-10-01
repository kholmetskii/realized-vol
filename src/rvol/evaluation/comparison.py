"""Statistical comparison of competing volatility forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from rvol.evaluation.losses import log_squared_error, qlike


@dataclass(frozen=True)
class DieboldMarianoResult:
    """One-sided test of whether HAR has lower expected loss than naïve."""

    n_obs: int
    mean_loss_difference: float
    statistic: float
    p_value: float
    hac_lags: int


@dataclass(frozen=True)
class ForecastComparison:
    """Mean losses and their one-sided Diebold-Mariano comparison."""

    metric: str
    n_obs: int
    naive_mean_loss: float
    har_mean_loss: float
    mean_loss_difference: float
    dm_statistic: float
    p_value: float
    hac_lags: int


def _loss_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype="float64")
    if vector.ndim != 1 or len(vector) < 3:
        raise ValueError(f"{name} must contain at least three observations")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _default_hac_lags(n_obs: int) -> int:
    """Newey-West's common sample-size rule, bounded by available lags."""
    rule = int(np.floor(4 * (n_obs / 100) ** (2 / 9)))
    return min(rule, n_obs - 2)


def diebold_mariano(
    naive_loss: ArrayLike,
    har_loss: ArrayLike,
    *,
    hac_lags: int | None = None,
) -> DieboldMarianoResult:
    """Test whether HAR's expected loss is lower than the naïve model's.

    The loss differential is ``naive_loss - har_loss``. The reported p-value
    is one-sided for the alternative that its expectation is positive. A
    Bartlett-kernel Newey-West estimate allows the differential to be serially
    correlated.
    """
    naive = _loss_vector(naive_loss, "naive_loss")
    har = _loss_vector(har_loss, "har_loss")
    if naive.shape != har.shape:
        raise ValueError("naive_loss and har_loss must have the same shape")

    n_obs = len(naive)
    lags = _default_hac_lags(n_obs) if hac_lags is None else hac_lags
    if not 0 <= lags <= n_obs - 2:
        raise ValueError("hac_lags must satisfy 0 <= hac_lags <= n_obs - 2")

    differential = naive - har
    mean_difference = float(np.mean(differential))
    centered = differential - mean_difference
    long_run_variance = float(centered @ centered / n_obs)
    for lag in range(1, lags + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n_obs)
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)

    tolerance = np.finfo("float64").eps * max(1.0, abs(mean_difference))
    if long_run_variance <= tolerance:
        if abs(mean_difference) <= tolerance:
            statistic, p_value = 0.0, 0.5
        elif mean_difference > 0:
            statistic, p_value = float("inf"), 0.0
        else:
            statistic, p_value = float("-inf"), 1.0
    else:
        standard_error = np.sqrt(long_run_variance / n_obs)
        statistic = mean_difference / standard_error
        p_value = float(norm.sf(statistic))

    return DieboldMarianoResult(
        n_obs=n_obs,
        mean_loss_difference=mean_difference,
        statistic=float(statistic),
        p_value=p_value,
        hac_lags=lags,
    )


def _comparison(
    metric: str,
    naive_loss: NDArray[np.float64],
    har_loss: NDArray[np.float64],
    hac_lags: int | None,
) -> ForecastComparison:
    test = diebold_mariano(naive_loss, har_loss, hac_lags=hac_lags)
    return ForecastComparison(
        metric=metric,
        n_obs=test.n_obs,
        naive_mean_loss=float(np.mean(naive_loss)),
        har_mean_loss=float(np.mean(har_loss)),
        mean_loss_difference=test.mean_loss_difference,
        dm_statistic=test.statistic,
        p_value=test.p_value,
        hac_lags=test.hac_lags,
    )


def compare_forecasts(
    forecasts: pd.DataFrame,
    *,
    hac_lags: int | None = None,
) -> tuple[ForecastComparison, ForecastComparison]:
    """Compare naïve and HAR walk-forward forecasts under both loss metrics."""
    required = {
        "actual_log_rv",
        "naive_log_prediction",
        "har_log_prediction",
        "actual_rv",
        "naive_rv_prediction",
        "har_rv_prediction",
    }
    missing = required.difference(forecasts.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"forecasts are missing required columns: {names}")

    qlike_naive = qlike(forecasts["actual_rv"], forecasts["naive_rv_prediction"])
    qlike_har = qlike(forecasts["actual_rv"], forecasts["har_rv_prediction"])
    log_mse_naive = log_squared_error(
        forecasts["actual_log_rv"], forecasts["naive_log_prediction"]
    )
    log_mse_har = log_squared_error(
        forecasts["actual_log_rv"], forecasts["har_log_prediction"]
    )
    return (
        _comparison("QLIKE", qlike_naive, qlike_har, hac_lags),
        _comparison("log-RV MSE", log_mse_naive, log_mse_har, hac_lags),
    )
