"""Immutable predictors used by benchmark forecast models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rvol.domain.contracts import FloatArray
from rvol.models._validation import validated_features


@dataclass(frozen=True)
class ConstantPredictor:
    """Predict one fitted constant for every row."""

    value: float
    n_features: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError("prediction constant must be finite")
        if self.n_features < 0:
            raise ValueError("n_features must be non-negative")

    def predict(self, features: FloatArray) -> FloatArray:
        matrix = validated_features(features, self.n_features)
        return np.full(len(matrix), self.value, dtype="float64")


@dataclass(frozen=True)
class FeatureColumnPredictor:
    """Return one selected feature as the prediction."""

    n_features: int
    column: int = 0

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError("n_features must be positive")
        if not 0 <= self.column < self.n_features:
            raise ValueError("column must identify an available feature")

    def predict(self, features: FloatArray) -> FloatArray:
        matrix = validated_features(features, self.n_features)
        return matrix[:, self.column].copy()
