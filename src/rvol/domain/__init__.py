"""Stable domain contracts and value objects for volatility research."""

from rvol.domain.comparisons import ComparisonOutcome
from rvol.domain.config import ExperimentConfig
from rvol.domain.contracts import (
    FeatureBuilder,
    ForecastComparisonTest,
    ForecastMetric,
    ForecastModel,
    Predictor,
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
    "EveryNSessions",
    "ExecutionMetadata",
    "ExecutionSpecification",
    "ExpandingWindow",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentSpecification",
    "FeatureBuilder",
    "ForecastComparisonTest",
    "ForecastMetric",
    "ForecastModel",
    "ForecastRecord",
    "HoldoutSettings",
    "Predictor",
    "RefitSchedule",
    "RollingWindow",
    "RunIdentity",
    "TrainingWindowPolicy",
    "TargetHorizon",
    "TargetBuilder",
    "UpdatablePredictor",
    "WalkForwardSettings",
]
