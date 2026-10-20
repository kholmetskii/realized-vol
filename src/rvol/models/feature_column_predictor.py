"""Feature-column prediction state fitted by NaiveModel."""

from __future__ import annotations

import numpy as np

from rvol.models._validation import validated_features
from rvol.models.arrays import FloatArray
from rvol.models.predictor import Predictor


class FeatureColumnPredictor(Predictor):
    """Return a numerical feature using read-only column and offset settings."""

    def __init__(self, n_features: int, column: int = 0, offset: float = 0.0) -> None:
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if not 0 <= column < n_features:
            raise ValueError("column must identify an available feature")
        if not np.isfinite(offset):
            raise ValueError("prediction offset must be finite")
        self._n_features = n_features
        self._column = column
        self._offset = offset

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def column(self) -> int:
        return self._column

    @property
    def offset(self) -> float:
        return self._offset

    def predict(self, features: FloatArray) -> FloatArray:
        matrix = validated_features(features, self.n_features)
        result = matrix[:, self.column].copy()
        return result if self.offset == 0 else result + self.offset
