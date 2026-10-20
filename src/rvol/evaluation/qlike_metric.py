"""The QLIKE metric for evaluate()."""

from __future__ import annotations

import numpy as np

from rvol.models import FloatArray


class QLikeMetric:
    """QLIKE loss actual/predicted - log(actual/predicted) - 1 on positive values.

    The loss is zero only for a perfect forecast, and under-prediction costs
    more than over-prediction by the same factor. evaluate() supplies finite,
    aligned one-dimensional arrays; this metric adds the positivity check.
    """

    name = "QLIKE"

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
        if np.any(actual <= 0) or np.any(predicted <= 0):
            raise ValueError("QLIKE requires strictly positive actual and predicted values")
        ratio = actual / predicted
        # The expression is theoretically non-negative; clipping removes tiny
        # negative values caused only by floating-point cancellation near ratio=1.
        return np.maximum(ratio - np.log(ratio) - 1.0, 0.0)
