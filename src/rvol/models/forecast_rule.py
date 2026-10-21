"""The interface every forecasting rule implements."""

from __future__ import annotations

from typing import Protocol

from rvol.models.arrays import FloatArray


class ForecastRule(Protocol):
    """How a forecast is computed from regressors: estimate with fit(), then predict().

    ``fit`` returns the rule so calls can be chained. Rules receive plain arrays
    and know nothing about column names, dates or units.
    """

    def fit(self, features: FloatArray, target: FloatArray) -> ForecastRule: ...

    def predict(self, features: FloatArray) -> FloatArray: ...
