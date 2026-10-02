"""Ordinary least-squares fitting and immutable linear prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rvol.domain.contracts import FloatArray
from rvol.models._validation import validated_features, validated_training_data


@dataclass(frozen=True)
class LinearPredictor:
    """Immutable fitted linear regression coefficients."""

    intercept: float
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(self.intercept) or not np.all(np.isfinite(self.coefficients)):
            raise ValueError("linear coefficients must be finite")

    def predict(self, features: FloatArray) -> FloatArray:
        matrix = validated_features(features, len(self.coefficients))
        coefficients = np.asarray(self.coefficients, dtype="float64")
        return self.intercept + matrix @ coefficients


def fit_ols(
    features: FloatArray,
    target: FloatArray,
    *,
    n_features: int,
) -> LinearPredictor:
    """Fit an intercept and slopes by ordinary least squares."""
    matrix, response = validated_training_data(
        features,
        target,
        n_features=n_features,
        min_observations=n_features + 1,
    )
    design = np.column_stack([np.ones(len(matrix)), matrix])
    fitted, *_ = np.linalg.lstsq(design, response, rcond=None)
    return LinearPredictor(
        intercept=float(fitted[0]),
        coefficients=tuple(float(value) for value in fitted[1:]),
    )
