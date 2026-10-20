"""Linear prediction state fitted by LinearRegressionModel."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rvol.models._validation import validated_features
from rvol.models.arrays import FloatArray
from rvol.models.predictor import Predictor


class LinearPredictor(Predictor):
    """Fitted linear predictor with read-only coefficients."""

    def __init__(self, intercept: float, coefficients: Sequence[float]) -> None:
        stored_coefficients = tuple(coefficients)
        if not np.isfinite(intercept) or not np.all(np.isfinite(stored_coefficients)):
            raise ValueError("linear coefficients must be finite")
        self._intercept = intercept
        self._coefficients = stored_coefficients

    @property
    def intercept(self) -> float:
        return self._intercept

    @property
    def coefficients(self) -> tuple[float, ...]:
        return self._coefficients

    def predict(self, features: FloatArray) -> FloatArray:
        matrix = validated_features(features, len(self.coefficients))
        coefficients = np.asarray(self.coefficients, dtype="float64")
        return self.intercept + matrix @ coefficients
