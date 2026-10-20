"""QLIKE loss for variance forecasts and the QLIKE metric for evaluate()."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rvol.models import FloatArray


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


class QLikeMetric:
    """QLIKE on variance-scale values; pass the output of variance_forecasts."""

    name = "QLIKE"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        return qlike(actual, predicted)
