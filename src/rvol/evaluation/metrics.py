"""Forecast losses: plain per-observation functions and named metrics for evaluate()."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rvol.models.contracts import FloatArray


def _aligned(
    actual: ArrayLike, predicted: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed = np.asarray(actual, dtype="float64")
    forecast = np.asarray(predicted, dtype="float64")
    if observed.ndim != 1 or not len(observed) or observed.shape != forecast.shape:
        raise ValueError("actual and predicted must be non-empty aligned one-dimensional arrays")
    if not np.isfinite(observed).all() or not np.isfinite(forecast).all():
        raise ValueError("actual and predicted must contain only finite values")
    return observed, forecast


def squared_error(actual: ArrayLike, predicted: ArrayLike) -> NDArray[np.float64]:
    observed, forecast = _aligned(actual, predicted)
    return np.square(observed - forecast)


def absolute_error(actual: ArrayLike, predicted: ArrayLike) -> NDArray[np.float64]:
    observed, forecast = _aligned(actual, predicted)
    return np.abs(observed - forecast)


class MSEMetric:
    """Mean squared error in the supplied target scale."""

    name = "MSE"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        return squared_error(actual, predicted)


class MAEMetric:
    """Mean absolute error in the supplied target scale."""

    name = "MAE"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        return absolute_error(actual, predicted)
