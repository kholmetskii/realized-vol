"""EWMAModel fitting and its numerical prediction state."""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np

from rvol.models._validation import validate_decay, validated_features, validated_training_data
from rvol.models.contracts import FloatArray


class EWMAPredictor:
    """Read-only exponentially weighted state; updates return a new predictor."""

    def __init__(self, value: float, decay: float) -> None:
        validate_decay(decay)
        if not np.isfinite(value):
            raise ValueError("EWMA state must be finite")
        self._value = value
        self._decay = decay

    @property
    def value(self) -> float:
        return self._value

    @property
    def decay(self) -> float:
        return self._decay

    def predict(self, features: FloatArray) -> FloatArray:
        matrix = validated_features(features, n_features=0)
        return np.full(len(matrix), self.value, dtype="float64")

    def update(self, features: FloatArray, target: FloatArray) -> Self:
        _, response = validated_training_data(
            features,
            target,
            n_features=0,
            min_observations=0,
        )
        if not len(response):
            return self

        forecast = self.value
        for observed in response:
            forecast = self.decay * forecast + (1.0 - self.decay) * float(observed)
        return type(self)(value=forecast, decay=self.decay)


class EWMAModel:
    """Smooth supplied target values with a fixed decay."""

    name: ClassVar[str] = "EWMA"

    def __init__(self, decay: float = 0.94) -> None:
        validate_decay(decay)
        self._decay = decay

    @property
    def decay(self) -> float:
        return self._decay

    def fit(self, features: FloatArray, target: FloatArray) -> EWMAPredictor:
        matrix, response = validated_training_data(
            features,
            target,
            n_features=0,
            min_observations=1,
        )
        initial = EWMAPredictor(
            value=float(response[0]),
            decay=self.decay,
        )
        return initial.update(matrix[1:], response[1:])
