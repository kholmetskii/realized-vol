"""Stable domain contracts and value objects for volatility research."""

from rvol.domain.config import ExperimentConfig
from rvol.domain.contracts import (
    DatasetRepository,
    FeatureBuilder,
    FittedForecaster,
    Forecaster,
    ForecastMetric,
)
from rvol.domain.results import ExperimentResult, ForecastRecord
from rvol.domain.specifications import ComponentSpecification, ExperimentSpecification

__all__ = [
    "ComponentSpecification",
    "DatasetRepository",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentSpecification",
    "FeatureBuilder",
    "FittedForecaster",
    "Forecaster",
    "ForecastMetric",
    "ForecastRecord",
]
