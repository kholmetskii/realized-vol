"""Squared-error loss and the MSE metric for evaluate()."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rvol.evaluation._validation import aligned_vectors
from rvol.models import FloatArray


def squared_error(actual: ArrayLike, predicted: ArrayLike) -> NDArray[np.float64]:
    observed, forecast = aligned_vectors(actual, predicted)
    return np.square(observed - forecast)


class MSEMetric:
    """Mean squared error in the supplied target scale."""

    name = "MSE"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        return squared_error(actual, predicted)
