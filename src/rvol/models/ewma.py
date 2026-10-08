"""Exponentially weighted realized-volatility forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from rvol.domain.contracts import FloatArray
from rvol.domain.specifications import ComponentSpecification
from rvol.models._validation import validated_training_data
from rvol.models.predictors import ConstantPredictor


@dataclass(frozen=True)
class EWMAForecaster:
    """Forecast variance with a fixed-decay RiskMetrics-style EWMA.

    Training targets are log realized variances. The recursion is evaluated in
    log space for numerical stability, but it averages on the variance scale.
    The fixed decay avoids tuning the model on the evaluation period.
    """

    decay: float = 0.94

    name: ClassVar[str] = "EWMA"
    feature_names: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.decay) or not 0.0 < self.decay < 1.0:
            raise ValueError("EWMA decay must be strictly between zero and one")

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name=self.name,
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            feature_names=self.feature_names,
            parameters=(("decay", self.decay),),
        )

    def fit(self, features: FloatArray, target: FloatArray) -> ConstantPredictor:
        _, response = validated_training_data(
            features,
            target,
            n_features=0,
            min_observations=1,
        )
        forecast = float(response[0])
        log_decay = float(np.log(self.decay))
        log_update = float(np.log1p(-self.decay))
        for observed_log_rv in response[1:]:
            forecast = float(np.logaddexp(
                log_decay + forecast,
                log_update + observed_log_rv,
            ))
        return ConstantPredictor(value=forecast)
