"""Shared immutable predictors and numerical validation for forecast models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rvol.domain.contracts import FloatArray


def validated_features(values: FloatArray, n_features: int) -> FloatArray:
    """Return a finite two-dimensional feature matrix with the expected width."""
    matrix = np.asarray(values, dtype="float64")
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if matrix.shape[1] != n_features:
        raise ValueError(f"features must contain exactly {n_features} columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix


def validated_training_data(
    features: FloatArray,
    target: FloatArray,
    *,
    n_features: int,
    min_observations: int,
) -> tuple[FloatArray, FloatArray]:
    """Validate aligned numerical training arrays without modifying them."""
    matrix = validated_features(features, n_features)
    response = np.asarray(target, dtype="float64")
    if response.ndim != 1:
        raise ValueError("target must be a one-dimensional array")
    if len(matrix) != len(response):
        raise ValueError("features and target must contain the same number of rows")
    if len(response) < min_observations:
        raise ValueError(f"model fitting requires at least {min_observations} observations")
    if not np.all(np.isfinite(response)):
        raise ValueError("target must contain only finite values")
    return matrix, response


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


def fit_linear_regression(
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
