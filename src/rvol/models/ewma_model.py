"""Exponentially weighted moving average of the supplied targets."""

from __future__ import annotations

from typing import ClassVar

from rvol.models._validation import validate_decay, validated_training_data
from rvol.models.arrays import FloatArray
from rvol.models.ewma_predictor import EWMAPredictor
from rvol.models.forecast_model import ForecastModel


class EWMAModel(ForecastModel):
    """Smooth supplied target values with a fixed decay."""

    name: ClassVar[str] = "EWMA"

    def __init__(self, decay: float) -> None:
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
