"""Array validation shared by the forecast loss metrics."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def aligned_vectors(
    actual: ArrayLike, predicted: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed = np.asarray(actual, dtype="float64")
    forecast = np.asarray(predicted, dtype="float64")
    if observed.ndim != 1 or not len(observed) or observed.shape != forecast.shape:
        raise ValueError("actual and predicted must be non-empty aligned one-dimensional arrays")
    if not np.isfinite(observed).all() or not np.isfinite(forecast).all():
        raise ValueError("actual and predicted must contain only finite values")
    return observed, forecast
