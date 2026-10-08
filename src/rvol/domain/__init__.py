"""Stable domain contracts and value objects for volatility research."""

from rvol.domain.comparisons import ComparisonOutcome
from rvol.domain.config import ExperimentConfig
from rvol.domain.contracts import (
    DatasetRepository,
    DescribedExecutionSettings,
    ExperimentExecutor,
    ExperimentExecutorFactory,
    FeatureBuilder,
    FittedForecaster,
    ForecastComparisonTest,
    Forecaster,
    ForecastMetric,
    RefitSchedule,
    TargetBuilder,
    TrainingWindowPolicy,
    UpdatablePredictor,
)
from rvol.domain.identity import RunIdentity
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
from rvol.domain.targets import TargetHorizon

__all__ = [
    "ComponentSpecification",
    "ComparisonOutcome",
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
    "ForecastComparisonTest",
    "ForecastMetric",
    "ForecastRecord",
    "HoldoutSettings",
    "RefitSchedule",
    "RollingWindow",
    "RunIdentity",
    "TrainingWindowPolicy",
    "TargetHorizon",
    "TargetBuilder",
    "UpdatablePredictor",
    "WalkForwardSettings",
]
