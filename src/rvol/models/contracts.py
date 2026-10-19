"""Small numerical interfaces for interchangeable models and predictors."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Predictor(Protocol):
    """Return one finite prediction per feature row without changing inputs."""

    def predict(self, features: FloatArray) -> FloatArray: ...


@runtime_checkable
class UpdatablePredictor(Predictor, Protocol):
    """Consume newly observable targets and return new fitted state."""

    def update(self, features: FloatArray, target: FloatArray) -> Self: ...


class ForecastModel(Protocol):
    """Fit numerical arrays without choosing columns, dates or target units."""

    @property
    def name(self) -> str: ...

    def fit(self, features: FloatArray, target: FloatArray) -> Predictor: ...
