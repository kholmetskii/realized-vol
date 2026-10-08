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
    """A fitted model returning forecasts in the canonical log-variance scale.

    Predictions use the feature order declared by the originating forecaster.
    Log variance is the output representation, not necessarily the statistical
    target: log(mean variance) and mean(log variance) are different forecasts.
    Each model documents which quantity its fitting procedure estimates.
    """

    def predict(self, features: FloatArray) -> FloatArray:
        """Return one finite log variance per row without modifying features.

        The result has shape (n_rows,); exponentiating each value must yield
        a finite positive variance.
        """
        ...


@runtime_checkable
class Forecaster(Protocol):
    """Fit a forecasting strategy using numerical arrays without mutating them.

    Features have shape (n_rows, n_features), with columns in feature_names
    order. Targets have shape (n_rows,) and contain observed log realized
    variances. Models may transform these targets internally during fitting.
    """

    @property
    def name(self) -> str: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    def fit(self, features: FloatArray, target: FloatArray) -> FittedForecaster:
        """Fit aligned training rows and return a log-variance predictor."""
        ...


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
