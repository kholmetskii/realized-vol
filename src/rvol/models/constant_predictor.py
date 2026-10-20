"""Constant prediction state fitted by HistoricalMeanModel."""

from __future__ import annotations

import numpy as np

from rvol.models._validation import validated_features
from rvol.models.arrays import FloatArray
from rvol.models.predictor import Predictor


class ConstantPredictor(Predictor):
    """Return one read-only fitted numerical constant for every row."""

    def __init__(self, value: float, n_features: int = 0) -> None:
        if not np.isfinite(value):
            raise ValueError("prediction constant must be finite")
        if n_features < 0:
            raise ValueError("n_features must be non-negative")
        self._value = value
        self._n_features = n_features

    @property
    def value(self) -> float:
        return self._value

    @property
    def n_features(self) -> int:
        return self._n_features

    def predict(self, features: FloatArray) -> FloatArray:
        matrix = validated_features(features, self.n_features)
        return np.full(len(matrix), self.value, dtype="float64")
