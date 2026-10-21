"""The QLIKE loss for strictly positive forecasts."""

from __future__ import annotations

import numpy as np

from rvol.models import FloatArray


def qlike(actual: FloatArray, predicted: FloatArray) -> FloatArray:
    """Per-forecast losses actual/predicted - log(actual/predicted) - 1.

    The loss is zero only for a perfect forecast, and under-prediction costs
    more than over-prediction by the same factor.
    """
    if np.any(actual <= 0) or np.any(predicted <= 0):
        raise ValueError("QLIKE requires strictly positive actual and predicted values")
    ratio = actual / predicted
    # The expression is theoretically non-negative; clipping removes tiny
    # negative values caused only by floating-point cancellation near ratio=1.
    return np.maximum(ratio - np.log(ratio) - 1.0, 0.0)
