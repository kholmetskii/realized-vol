"""Naive forecast: the current observation, optionally shifted."""

from __future__ import annotations

import math
from typing import ClassVar

from rvol.models._validation import validated_training_data
from rvol.models.arrays import FloatArray
from rvol.models.feature_column_predictor import FeatureColumnPredictor
from rvol.models.forecast_model import ForecastModel


class NaiveModel(ForecastModel):
    """Use a selected current observation, optionally shifted by a fixed offset."""

    name: ClassVar[str] = "naive"

    def __init__(self, offset: float = 0.0) -> None:
        if not math.isfinite(offset):
            raise ValueError("offset must be finite")
        self._offset = offset

    @property
    def offset(self) -> float:
        return self._offset

    def fit(self, features: FloatArray, target: FloatArray) -> FeatureColumnPredictor:
        validated_training_data(features, target, n_features=1, min_observations=1)
        return FeatureColumnPredictor(n_features=1, offset=self.offset)
