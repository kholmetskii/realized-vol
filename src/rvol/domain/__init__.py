"""Stable domain contracts and value objects for volatility research."""

from rvol.domain.config import ExperimentConfig
from rvol.domain.contracts import (
    DatasetRepository,
    DescribedExecutionSettings,
    ExperimentExecutor,
    ExperimentExecutorFactory,
    FeatureBuilder,
    FittedForecaster,
    Forecaster,
    ForecastMetric,
    RefitSchedule,
    TrainingWindowPolicy,
    UpdatablePredictor,
)
from rvol.domain.policies import (
    EveryNSessions,
    ExpandingWindow,
    RollingWindow,
)
from rvol.domain.results import ExperimentResult, ForecastRecord
from rvol.domain.settings import HoldoutSettings, WalkForwardSettings
from rvol.domain.specifications import (
    ComponentSpecification,
    ExecutionMetadata,
    ExecutionSpecification,
    ExperimentSpecification,
)

__all__ = [
    "ComponentSpecification",
    "DatasetRepository",
    "DescribedExecutionSettings",
    "EveryNSessions",
    "ExecutionMetadata",
    "ExecutionSpecification",
    "ExpandingWindow",
    "ExperimentConfig",
    "ExperimentExecutor",
    "ExperimentExecutorFactory",
    "ExperimentResult",
    "ExperimentSpecification",
    "FeatureBuilder",
    "FittedForecaster",
    "Forecaster",
    "ForecastMetric",
    "ForecastRecord",
    "HoldoutSettings",
    "RefitSchedule",
    "RollingWindow",
    "TrainingWindowPolicy",
    "UpdatablePredictor",
    "WalkForwardSettings",
]
