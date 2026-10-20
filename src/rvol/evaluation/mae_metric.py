"""Absolute-error loss and the MAE metric for evaluate()."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rvol.evaluation._validation import aligned_vectors
from rvol.models import FloatArray


def absolute_error(actual: ArrayLike, predicted: ArrayLike) -> NDArray[np.float64]:
    observed, forecast = aligned_vectors(actual, predicted)
    return np.abs(observed - forecast)


class MAEMetric:
    """Mean absolute error in the supplied target scale."""

    name = "MAE"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        return absolute_error(actual, predicted)
