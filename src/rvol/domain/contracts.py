"""Structural contracts implemented by application and adapter objects."""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ForecastScale = Literal["log_variance", "variance"]
Dataset = TypeVar("Dataset")


@runtime_checkable
class FittedForecaster(Protocol):
    """A fitted model capable of predicting from a numerical feature matrix."""

    def predict(self, features: FloatArray) -> FloatArray: ...


@runtime_checkable
class Forecaster(Protocol):
    """An unfitted forecasting strategy used by a walk-forward experiment."""

    @property
    def name(self) -> str: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    def fit(self, features: FloatArray, target: FloatArray) -> FittedForecaster: ...


@runtime_checkable
class ForecastMetric(Protocol):
    """A per-observation loss metric on one declared forecast scale."""

    @property
    def name(self) -> str: ...

    @property
    def scale(self) -> ForecastScale: ...

    def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray: ...


@runtime_checkable
class DatasetRepository(Protocol[Dataset]):
    """Persistence boundary for a dataset representation chosen by an adapter."""

    def load(self) -> Dataset: ...

    def save(self, dataset: Dataset) -> None: ...
