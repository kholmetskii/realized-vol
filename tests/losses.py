"""Simple losses for exercising the target-agnostic evaluation functions."""

import numpy as np

from rvol.models import FloatArray


def squared_error(actual: FloatArray, predicted: FloatArray) -> FloatArray:
    return np.square(actual - predicted)


def absolute_error(actual: FloatArray, predicted: FloatArray) -> FloatArray:
    return np.abs(actual - predicted)
