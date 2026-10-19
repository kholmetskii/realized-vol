"""LinearRegressionModel fitting and its numerical prediction state."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rvol.models._validation import validated_features, validated_training_data
from rvol.models.contracts import FloatArray, Predictor


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


class LinearRegressionModel:
    """Fit an intercept and slopes; input columns and units belong to the caller."""

    def __init__(self, name: str = "linear_regression") -> None:
        self.name = name

    def fit(self, features: FloatArray, target: FloatArray) -> LinearPredictor:
        matrix, response = validated_training_data(
            features, target, n_features=None, min_observations=1
        )
        required = matrix.shape[1] + 1
        if len(response) < required:
            raise ValueError(f"model fitting requires at least {required} observations")
        design = np.column_stack([np.ones(len(matrix)), matrix])
        fitted, *_ = np.linalg.lstsq(design, response, rcond=None)
        return LinearPredictor(
            intercept=float(fitted[0]),
            coefficients=tuple(float(value) for value in fitted[1:]),
        )
