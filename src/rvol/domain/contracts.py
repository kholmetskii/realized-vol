"""Structural contracts implemented by application and adapter objects."""

from __future__ import annotations

from typing import Literal, Protocol, Self, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from rvol.domain.specifications import ComponentSpecification

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
class UpdatablePredictor(FittedForecaster, Protocol):
    """A fitted predictor that can consume new observations between fits."""

    def update(self, features: FloatArray, target: FloatArray) -> Self:
        """Return updated state without modifying this predictor or its inputs.

        Rows are new, aligned observations in chronological order, with the
        originating model's feature order and log-variance targets. The caller
        supplies only outcomes available at the forecast origin and consumes
        each observation once. An empty batch leaves the state unchanged.
        Updating state retains the configured model parameters.
        """
        ...


@runtime_checkable
class TrainingWindowPolicy(Protocol):
    """Select a contiguous window from chronological, already eligible rows."""

    def select(self, n_available: int) -> slice:
        """Return a slice within [0, n_available), without selecting future rows."""
        ...


@runtime_checkable
class RefitSchedule(Protocol):
    """Decide when to fit using the dataset's eligible forecast-origin index."""

    def should_refit(self, eligible_origin_index: int) -> bool:
        """Index zero is the first origin satisfying the training requirement.

        Count observed eligible origins, including those before the reporting
        start date, rather than calendar days or rows in a filtered report.
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

    @property
    def specification(self) -> ComponentSpecification:
        """Describe this configured model, including its actual parameter values."""
        ...

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


@runtime_checkable
class FeatureBuilder(Protocol[Dataset]):
    """Prepare forecast features and describe the settings used to build them."""

    @property
    def specification(self) -> ComponentSpecification: ...

    def __call__(self, daily: Dataset) -> Dataset: ...
