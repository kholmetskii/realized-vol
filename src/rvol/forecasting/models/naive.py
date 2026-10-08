"""Persistence benchmark for realized-volatility forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from rvol.domain.contracts import FloatArray
from rvol.domain.specifications import ComponentSpecification
from rvol.forecasting.models._validation import validated_training_data
from rvol.forecasting.models.predictors import FeatureColumnPredictor


@dataclass(frozen=True)
class NaiveForecaster:
    """Persist current variance, with an explicit variance-scale multiplier."""

    variance_multiplier: float = 1.0
    name: ClassVar[str] = "naive"
    feature_names: ClassVar[tuple[str, ...]] = ("rv_daily",)

    def __post_init__(self) -> None:
        if not math.isfinite(self.variance_multiplier) or self.variance_multiplier <= 0:
            raise ValueError("variance_multiplier must be finite and positive")

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name=self.name,
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            feature_names=self.feature_names,
            parameters=(("variance_multiplier", self.variance_multiplier),),
        )

    def fit(self, features: FloatArray, target: FloatArray) -> FeatureColumnPredictor:
        validated_training_data(
            features,
            target,
            n_features=len(self.feature_names),
            min_observations=1,
        )
        return FeatureColumnPredictor(
            n_features=len(self.feature_names), offset=math.log(self.variance_multiplier),
        )
