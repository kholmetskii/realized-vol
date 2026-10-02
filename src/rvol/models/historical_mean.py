"""Expanding historical-mean realized-volatility benchmark."""

from __future__ import annotations

import numpy as np

from rvol.domain.contracts import FloatArray
from rvol.models._validation import validated_training_data
from rvol.models.predictors import ConstantPredictor


class HistoricalMeanForecaster:
    """Predict the mean log RV observed in the training sample."""

    name = "historical_mean"
    feature_names: tuple[str, ...] = ()

    def fit(self, features: FloatArray, target: FloatArray) -> ConstantPredictor:
        _, response = validated_training_data(
            features,
            target,
            n_features=0,
            min_observations=1,
        )
        return ConstantPredictor(value=float(np.mean(response)))
