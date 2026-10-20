"""Least-squares linear regression on prepared numerical features."""

from __future__ import annotations

import numpy as np

from rvol.models._validation import validated_training_data
from rvol.models.arrays import FloatArray
from rvol.models.forecast_model import ForecastModel
from rvol.models.linear_predictor import LinearPredictor


class LinearRegressionModel(ForecastModel):
    """Fit an intercept and slopes; input columns and units belong to the caller."""

    def __init__(self, name: str = "linear_regression") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

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
