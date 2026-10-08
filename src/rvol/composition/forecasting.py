"""Compose the standard realized-volatility forecast experiment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Generic, TypeVar, overload

import pandas as pd

from rvol.application import HoldoutExperiment, WalkForwardExperiment, prepare_forecast_data
from rvol.domain import (
    DescribedExecutionSettings,
    ExperimentConfig,
    ExperimentExecutorFactory,
    ExperimentResult,
    ExperimentSpecification,
    FeatureBuilder,
    Forecaster,
    HoldoutSettings,
    TargetBuilder,
    TargetHorizon,
    WalkForwardSettings,
)
from rvol.domain.contracts import ForecastComparisonTest, ForecastMetric
from rvol.domain.specifications import ComponentSpecification
from rvol.evaluation import (
    DiagnosticCheck,
    DieboldMarianoTest,
    ForecastEvaluator,
    ForecastRobustnessAnalyzer,
    LagSensitivityCheck,
    LargestErrorsCheck,
    LogMSEMetric,
    MonthlyLossCheck,
    QLikeMetric,
    WinRateCheck,
)
from rvol.features.forecasting import HARFeatureBuilder
from rvol.models import (
    AR1Forecaster,
    EWMAForecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)
from rvol.targets import RealizedVarianceTargetBuilder

Settings = TypeVar("Settings")
ROBUSTNESS_CHECK_NAMES = ("hac_sensitivity", "win_rates", "monthly_losses", "largest_errors")


def standard_diagnostic_checks(
    names: Sequence[str] | None = None, *,
    hac_lags: Sequence[int] = tuple(range(6)), top_errors: int = 5,
) -> tuple[DiagnosticCheck, ...]:
    selected = ROBUSTNESS_CHECK_NAMES if names is None else tuple(names)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("diagnostic check names must be non-empty and unique")
    factories: dict[str, Callable[[], DiagnosticCheck]] = {
        "hac_sensitivity": lambda: LagSensitivityCheck(tuple(hac_lags)),
        "win_rates": WinRateCheck,
        "monthly_losses": MonthlyLossCheck,
        "largest_errors": lambda: LargestErrorsCheck(top_errors),
    }
    if set(selected).difference(factories):
        raise ValueError("unknown diagnostic check")
    return tuple(factories[name]() for name in selected)


@dataclass(frozen=True)
class ExperimentDefinition(Generic[Settings]):
    """Select the components and comparison defaults for a forecast workflow."""

    models: tuple[Forecaster, ...]
    feature_builder: FeatureBuilder[pd.DataFrame]
    target_builder: TargetBuilder[pd.DataFrame]
    executor_factory: ExperimentExecutorFactory[Settings, pd.DataFrame]
    metrics: tuple[ForecastMetric, ...]
    comparison_pairs: tuple[tuple[str, str], ...]
    candidate_model: str
    baseline_models: tuple[str, ...]
    comparison_test: ForecastComparisonTest = field(default_factory=DieboldMarianoTest)

    def __post_init__(self) -> None:
        names = [model.name for model in self.models]
        if not names or len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise ValueError("models must have non-empty, unique names")
        if not self.metrics:
            raise ValueError("at least one metric is required")
        if len(set(self.comparison_pairs)) != len(self.comparison_pairs):
            raise ValueError("comparison pairs must be unique")
        references = {self.candidate_model, *self.baseline_models}
        references.update(name for pair in self.comparison_pairs for name in pair)
        missing = references.difference(names)
        if missing:
            raise ValueError(f"definition references unknown models: {', '.join(sorted(missing))}")

    @property
    def specification(self) -> ExperimentSpecification:
        """Snapshot settings from the configured components, in model-name order."""
        configured = sorted(self.models, key=lambda model: model.name)
        models = tuple(model.specification for model in configured)
        for model, spec in zip(configured, models, strict=True):
            if spec.name != model.name or spec.feature_names != model.feature_names:
                raise ValueError("model specification must match its name and feature order")
        return ExperimentSpecification(
            features=self.feature_builder.specification, models=models,
            target=self.target_builder.specification,
        )

    @property
    def all_comparison_pairs(self) -> tuple[tuple[str, str], ...]:
        """Compare every other configured model with the selected candidate."""
        return tuple(
            (name, self.candidate_model)
            for name in sorted(model.name for model in self.models)
            if name != self.candidate_model
        )

    def prepare(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Build independent inputs and outcomes, then align by forecast origin."""
        targets = self.target_builder(daily)
        return prepare_forecast_data(
            self.feature_builder(daily), targets, expected_horizon=self.target_builder.horizon,
        )

    def run(self, daily: pd.DataFrame, config: ExperimentConfig[Settings]) -> ExperimentResult:
        """Create an independent executor and delegate the aligned forecast data."""
        executor = self.executor_factory(self.models, config)
        return executor.run(self.prepare(daily))

    def evaluator(
        self,
        *,
        comparison_pairs: Sequence[tuple[str, str]] | None = None,
        hac_lags: int | None = None,
    ) -> ForecastEvaluator:
        """Build an evaluator with defaults or explicit comparison overrides."""
        test = self.comparison_test
        if hac_lags is not None:
            if not isinstance(test, DieboldMarianoTest):
                raise ValueError(
                    "hac_lags requires a DieboldMarianoTest; configure your test directly"
                )
            test = replace(test, hac_lags=hac_lags)
        return ForecastEvaluator(
            metrics=self.metrics,
            comparison_pairs=(
                self.comparison_pairs if comparison_pairs is None else comparison_pairs
            ),
            comparison_test=test,
        )

    def robustness_analyzer(
        self,
        *,
        hac_lags: Sequence[int] | None = None,
        top_errors: int | None = None,
        checks: Sequence[DiagnosticCheck] | None = None,
    ) -> ForecastRobustnessAnalyzer:
        """Build robustness checks for the configured candidate and baselines."""
        if checks is not None and (hac_lags is not None or top_errors is not None):
            raise ValueError("configure settings on the injected diagnostic checks")
        return ForecastRobustnessAnalyzer(
            checks=(
                standard_diagnostic_checks(
                    hac_lags=tuple(range(6)) if hac_lags is None else hac_lags,
                    top_errors=5 if top_errors is None else top_errors,
                )
                if checks is None else checks
            ),
            candidate_model=self.candidate_model,
            baseline_models=self.baseline_models,
            metrics=self.metrics,
        )


@dataclass(frozen=True)
class ExperimentRun(Generic[Settings]):
    """Bind an experiment definition to compatible settings for repeated execution."""

    definition: ExperimentDefinition[Settings]
    config: ExperimentConfig[Settings]
    diagnostic_checks: tuple[DiagnosticCheck, ...] = field(
        default_factory=standard_diagnostic_checks,
    )
    plot_metric: str | None = None

    def __post_init__(self) -> None:
        metric_names = [metric.name for metric in self.definition.metrics]
        if self.plot_metric is None:
            object.__setattr__(self, "plot_metric", metric_names[0])
        if self.plot_metric not in metric_names:
            raise ValueError("plot metric must be one of the selected evaluation metrics")
        self.definition.evaluator()
        # Validate model inputs and diagnostic selection before loading a dataset.
        _ = self.definition.specification
        self.definition.robustness_analyzer(checks=self.diagnostic_checks)

    @property
    def resolved_configuration(self) -> dict[str, object]:
        """Describe the actual components and settings, independent of their input source."""
        settings = self.config.execution
        if not isinstance(settings, DescribedExecutionSettings):
            raise ValueError("resolved configuration requires described execution settings")

        def component(value: ComponentSpecification) -> dict[str, object]:
            return asdict(value) | {"parameters": dict(value.parameters)}

        definition = self.definition
        specification = definition.specification
        return {
            "dataset": {"instrument": self.config.instrument},
            "run": {
                **asdict(self.config.identity),
                "forecast_start": (
                    None if self.config.forecast_start is None
                    else self.config.forecast_start.isoformat()
                ),
                "forecast_end": (
                    None if self.config.forecast_end is None
                    else self.config.forecast_end.isoformat()
                ),
            },
            "execution": {
                "settings": component(settings.specification.component),
                "policies": {name: component(policy)
                             for name, policy in settings.specification.policies},
            },
            "features": component(specification.features),
            "target": component(specification.target),
            "models": [component(model) for model in specification.models],
            "evaluation": {
                "metrics": [{"name": metric.name, "scale": metric.scale}
                            for metric in definition.metrics],
                "comparison_test": component(definition.comparison_test.specification),
                "comparison_pairs": [list(pair) for pair in definition.comparison_pairs],
                "candidate_model": definition.candidate_model,
                "baseline_models": list(definition.baseline_models),
            },
            "robustness": [component(check.specification) for check in self.diagnostic_checks],
            "plot": {"metric": self.plot_metric},
        }

    def run(self, daily: pd.DataFrame) -> ExperimentResult:
        return self.definition.run(daily, self.config)


def standard_forecasters(*, target_horizon: TargetHorizon | None = None) -> tuple[Forecaster, ...]:
    """Return fresh instances of the models used by standard project workflows."""
    horizon = TargetHorizon() if target_horizon is None else target_horizon
    return (
        HistoricalMeanForecaster(),
        NaiveForecaster(
            variance_multiplier=float(horizon.sessions) if horizon.kind == "cumulative" else 1.0,
        ),
        EWMAForecaster(),
        AR1Forecaster(),
        HARForecaster(),
    )


@overload
def standard_experiment_definition(
    *, target_horizon: TargetHorizon | None = None,
) -> ExperimentDefinition[WalkForwardSettings]: ...


@overload
def standard_experiment_definition(
    *, executor_factory: ExperimentExecutorFactory[Settings, pd.DataFrame],
    target_horizon: TargetHorizon | None = None,
) -> ExperimentDefinition[Settings]: ...


def standard_experiment_definition(
    *, executor_factory: ExperimentExecutorFactory[Settings, pd.DataFrame] | None = None,
    target_horizon: TargetHorizon | None = None,
) -> ExperimentDefinition[Settings] | ExperimentDefinition[WalkForwardSettings]:
    """Assemble fresh standard components with the selected execution method."""
    if executor_factory is None:
        return standard_experiment_definition(
            executor_factory=WalkForwardExperiment, target_horizon=target_horizon,
        )
    horizon = TargetHorizon() if target_horizon is None else target_horizon
    return ExperimentDefinition(
        models=standard_forecasters(target_horizon=horizon),
        feature_builder=HARFeatureBuilder(),
        target_builder=RealizedVarianceTargetBuilder(horizon),
        executor_factory=executor_factory,
        metrics=(QLikeMetric(), LogMSEMetric()),
        comparison_pairs=(("naive", "HAR"),),
        candidate_model="HAR",
        baseline_models=("naive", "EWMA", "AR1", "historical_mean"),
        comparison_test=DieboldMarianoTest(min_hac_lags=horizon.sessions - 1),
    )


def holdout_experiment_definition(
    *, target_horizon: TargetHorizon | None = None,
) -> ExperimentDefinition[HoldoutSettings]:
    """Assemble the standard components for fixed-cutoff holdout execution."""
    return standard_experiment_definition(
        executor_factory=HoldoutExperiment, target_horizon=target_horizon,
    )


def run_standard_forecast_experiment(
    daily: pd.DataFrame,
    config: ExperimentConfig[WalkForwardSettings],
    *, target_horizon: TargetHorizon | None = None,
) -> ExperimentResult:
    """Build HAR features and run the standard model set on common windows."""
    return standard_experiment_definition(target_horizon=target_horizon).run(daily, config)
