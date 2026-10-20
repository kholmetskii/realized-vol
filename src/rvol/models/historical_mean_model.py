"""Historical mean of the supplied training targets."""

from __future__ import annotations

import numpy as np

from rvol.models._validation import validated_training_data
from rvol.models.arrays import FloatArray
from rvol.models.constant_predictor import ConstantPredictor
from rvol.models.forecast_model import ForecastModel


class HistoricalMeanModel(ForecastModel):
    """Predict the arithmetic mean of the supplied training targets."""

    name = "historical_mean"

    def fit(self, features: FloatArray, target: FloatArray) -> ConstantPredictor:
        _, response = validated_training_data(
            features,
            target,
            n_features=0,
            min_observations=1,
        )
        return ConstantPredictor(value=float(np.mean(response)))
