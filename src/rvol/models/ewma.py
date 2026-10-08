"""Exponentially weighted realized-volatility forecasts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Self

import numpy as np

from rvol.domain.contracts import FloatArray
from rvol.domain.specifications import ComponentSpecification
from rvol.models._validation import validated_features, validated_training_data


def _validate_decay(decay: float) -> None:
    if not np.isfinite(decay) or not 0.0 < decay < 1.0:
        raise ValueError("EWMA decay must be strictly between zero and one")


@dataclass(frozen=True)
class EWMAPredictor:
    """An immutable EWMA log-variance state with a fixed decay.

    Prediction reads the current state. Update consumes newly observed log
    variances in chronological order and returns a new state with the same
    decay. The caller determines which observations are new and available.
    """

    value: float
    decay: float

    def __post_init__(self) -> None:
        _validate_decay(self.decay)
        if not np.isfinite(self.value):
            raise ValueError("EWMA state must be finite")

    def predict(self, features: FloatArray) -> FloatArray:
        matrix = validated_features(features, n_features=0)
        return np.full(len(matrix), self.value, dtype="float64")

    def update(self, features: FloatArray, target: FloatArray) -> Self:
        _, response = validated_training_data(
            features, target, n_features=0, min_observations=0,
        )
        if not len(response):
            return self

        forecast = self.value
        log_decay = float(np.log(self.decay))
        log_update = float(np.log1p(-self.decay))
        for observed_log_rv in response:
            forecast = float(np.logaddexp(
                log_decay + forecast,
                log_update + observed_log_rv,
            ))
        return replace(self, value=forecast)


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
        _validate_decay(self.decay)

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name=self.name,
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            feature_names=self.feature_names,
            parameters=(("decay", self.decay),),
        )

    def fit(self, features: FloatArray, target: FloatArray) -> EWMAPredictor:
        matrix, response = validated_training_data(
            features,
            target,
            n_features=0,
            min_observations=1,
        )
        initial = EWMAPredictor(value=float(response[0]), decay=self.decay)
        return initial.update(matrix[1:], response[1:])
