"""The exponential smoothing rule: fixed exponentially decaying weights on past targets."""

from __future__ import annotations

import numpy as np

from rvol.models.arrays import FloatArray


class ExponentialSmoothing:
    """Predict s_n, where s_1 = y_1 and s_k = decay * s_{k-1} + (1 - decay) * y_k.

    The weights are fixed, not estimated. ``decay=0`` gives s_n = y_n, the last
    observed target: the naive forecast that tomorrow equals today.
    """

    def __init__(self, decay: float) -> None:
        if not np.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("decay must lie in [0, 1)")
        self.decay = decay

    def fit(self, features: FloatArray, target: FloatArray) -> ExponentialSmoothing:
        value = float(target[0])
        for observed in target[1:]:
            value = self.decay * value + (1.0 - self.decay) * float(observed)
        self.value = value
        return self

    def predict(self, features: FloatArray) -> FloatArray:
        return np.full(len(features), self.value)
