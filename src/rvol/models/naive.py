"""Persistence benchmark for realized-volatility forecasts."""

from __future__ import annotations

from rvol.domain.contracts import FloatArray
from rvol.models.base import FeatureColumnPredictor, validated_training_data


class NaiveForecaster:
    """Predict that the next session's log RV equals the current session's."""

    name = "naive"
    feature_names = ("rv_daily",)

    def fit(self, features: FloatArray, target: FloatArray) -> FeatureColumnPredictor:
        validated_training_data(
            features,
            target,
            n_features=len(self.feature_names),
            min_observations=1,
        )
        return FeatureColumnPredictor(n_features=len(self.feature_names))
