"""Persistence benchmark for realized-volatility forecasts."""

from __future__ import annotations

from rvol.domain.contracts import FloatArray
from rvol.domain.specifications import ComponentSpecification
from rvol.models._validation import validated_training_data
from rvol.models.predictors import FeatureColumnPredictor


class NaiveForecaster:
    """Predict that the next session's log RV equals the current session's."""

    name = "naive"
    feature_names = ("rv_daily",)

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name=self.name,
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            feature_names=self.feature_names,
        )

    def fit(self, features: FloatArray, target: FloatArray) -> FeatureColumnPredictor:
        validated_training_data(
            features,
            target,
            n_features=len(self.feature_names),
            min_observations=1,
        )
        return FeatureColumnPredictor(n_features=len(self.feature_names))
