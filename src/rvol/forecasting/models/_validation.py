"""Numerical validation shared by forecast model implementations."""

from __future__ import annotations

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
