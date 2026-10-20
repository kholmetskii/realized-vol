"""Simple loss metrics for exercising the target-agnostic evaluation engine."""

import numpy as np

from rvol.models import FloatArray


class MSEMetric:
    name = "MSE"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        return np.square(actual - predicted)


class MAEMetric:
    name = "MAE"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        return np.abs(actual - predicted)
