"""The interface for predictors that consume new observations between refits."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from rvol.models.arrays import FloatArray
from rvol.models.predictor import Predictor


@runtime_checkable
class UpdatablePredictor(Predictor, Protocol):
    """Consume newly observable targets and return new fitted state."""

    def update(self, features: FloatArray, target: FloatArray) -> Self: ...
