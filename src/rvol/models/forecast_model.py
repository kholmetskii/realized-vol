"""The interface every forecast model implements."""

from __future__ import annotations

from typing import Protocol

from rvol.models.arrays import FloatArray
from rvol.models.predictor import Predictor


class ForecastModel(Protocol):
    """Fit numerical arrays without choosing columns, dates or target units."""

    @property
    def name(self) -> str: ...

    def fit(self, features: FloatArray, target: FloatArray) -> Predictor: ...
