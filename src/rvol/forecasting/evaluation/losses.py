"""Loss functions for realized-volatility forecasts."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype="float64")
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def qlike(actual_rv: ArrayLike, forecast_rv: ArrayLike) -> NDArray[np.float64]:
    """Return per-observation QLIKE losses for positive variance forecasts."""
    actual = _finite_vector(actual_rv, "actual_rv")
    forecast = _finite_vector(forecast_rv, "forecast_rv")
    if actual.shape != forecast.shape:
        raise ValueError("actual_rv and forecast_rv must have the same shape")
    if np.any(actual <= 0) or np.any(forecast <= 0):
        raise ValueError("QLIKE requires strictly positive actual and forecast variances")

    ratio = actual / forecast
    # The expression is theoretically non-negative; clipping removes tiny
    # negative values caused only by floating-point cancellation near ratio=1.
    return np.maximum(ratio - np.log(ratio) - 1.0, 0.0)


def log_squared_error(
    actual_log_rv: ArrayLike,
    forecast_log_rv: ArrayLike,
) -> NDArray[np.float64]:
    """Return squared errors on the log realized-variance scale."""
    actual = _finite_vector(actual_log_rv, "actual_log_rv")
    forecast = _finite_vector(forecast_log_rv, "forecast_log_rv")
    if actual.shape != forecast.shape:
        raise ValueError("actual_log_rv and forecast_log_rv must have the same shape")
    return np.square(actual - forecast)
