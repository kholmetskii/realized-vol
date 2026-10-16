"""Configure and run realized-volatility experiments with explicit workflow steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from rvol.domain import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentSpecification,
    FeatureBuilder,
    ForecastModel,
    HoldoutSettings,
    TargetBuilder,
    TargetHorizon,
    WalkForwardSettings,
)
from rvol.domain.contracts import ForecastComparisonTest, ForecastMetric
from rvol.domain.specifications import ComponentSpecification
from rvol.forecasting.evaluation import (
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
from rvol.forecasting.execution import (
    HoldoutExperiment,
    WalkForwardExperiment,
    prepare_forecast_data,
)
from rvol.forecasting.features import HARFeatureBuilder
from rvol.forecasting.models import (
    AR1Model,
    EWMAModel,
    HARModel,
    HistoricalMeanModel,
    NaiveModel,
)
from rvol.forecasting.targets import RealizedVarianceTargetBuilder


@dataclass(frozen=True)
class Experiment:
    """The settings and components for one repeatable forecast experiment.

    Preparation, execution, evaluation and export remain separate steps. The
    execution settings select either the walk-forward or fixed-cutoff workflow.
    """

    config: ExperimentConfig
    models: tuple[ForecastModel, ...]
    feature_builder: FeatureBuilder[pd.DataFrame]
    target_builder: TargetBuilder[pd.DataFrame]
    metrics: tuple[ForecastMetric, ...]
    comparison_pairs: tuple[tuple[str, str], ...]
    candidate_model: str
    baseline_models: tuple[str, ...]
    comparison_test: ForecastComparisonTest = field(default_factory=DieboldMarianoTest)
    diagnostic_checks: tuple[DiagnosticCheck, ...] = field(
        default_factory=lambda: (
            LagSensitivityCheck(), WinRateCheck(), MonthlyLossCheck(), LargestErrorsCheck(),
        ),
    )
    plot_metric: str | None = None

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
            raise ValueError(f"experiment references unknown models: {', '.join(sorted(missing))}")
        metric_names = [metric.name for metric in self.metrics]
        if self.plot_metric is None:
            object.__setattr__(self, "plot_metric", metric_names[0])
        if self.plot_metric not in metric_names:
            raise ValueError("plot metric must be one of the selected evaluation metrics")
        # Reject invalid selections before loading data or fitting models.
        self.evaluator()
        _ = self.specification
        self.robustness_analyzer()

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

    def run(self, daily: pd.DataFrame) -> ExperimentResult:
        """Prepare inputs and run the configured chronological execution method."""
        executor: HoldoutExperiment | WalkForwardExperiment
        if isinstance(self.config.execution, HoldoutSettings):
            executor = HoldoutExperiment(self.models, self.config)
        else:
            executor = WalkForwardExperiment(self.models, self.config)
        return executor.run(self.prepare(daily))

    def evaluator(self) -> ForecastEvaluator:
        """Build an evaluator from the experiment's metrics and comparison settings."""
        return ForecastEvaluator(
            metrics=self.metrics,
            comparison_pairs=self.comparison_pairs,
            comparison_test=self.comparison_test,
        )

    def robustness_analyzer(self) -> ForecastRobustnessAnalyzer:
        """Build an analyzer from the experiment's checks, candidate and baselines."""
        return ForecastRobustnessAnalyzer(
            checks=self.diagnostic_checks,
            candidate_model=self.candidate_model,
            baseline_models=self.baseline_models,
            metrics=self.metrics,
        )

    @property
    def resolved_configuration(self) -> dict[str, object]:
        """Describe the actual components and settings, independent of their input source."""
        settings = self.config.execution

        def component(value: ComponentSpecification) -> dict[str, object]:
            return asdict(value) | {"parameters": dict(value.parameters)}

        specification = self.specification
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
                            for metric in self.metrics],
                "comparison_test": component(self.comparison_test.specification),
                "comparison_pairs": [list(pair) for pair in self.comparison_pairs],
                "candidate_model": self.candidate_model,
                "baseline_models": list(self.baseline_models),
            },
            "robustness": [component(check.specification) for check in self.diagnostic_checks],
            "plot": {"metric": self.plot_metric},
        }


def standard_models(*, target_horizon: TargetHorizon | None = None) -> tuple[ForecastModel, ...]:
    """Return fresh instances of the models used by standard project workflows."""
    horizon = TargetHorizon() if target_horizon is None else target_horizon
    return (
        HistoricalMeanModel(),
        NaiveModel(
            variance_multiplier=float(horizon.sessions) if horizon.kind == "cumulative" else 1.0,
        ),
        EWMAModel(),
        AR1Model(),
        HARModel(),
    )


def standard_experiment(
    config: ExperimentConfig | None = None, *,
    target_horizon: TargetHorizon | None = None,
) -> Experiment:
    """Select fresh standard components and bind them to execution settings."""
    horizon = TargetHorizon() if target_horizon is None else target_horizon
    return Experiment(
        config=ExperimentConfig(execution=WalkForwardSettings()) if config is None else config,
        models=standard_models(target_horizon=horizon),
        feature_builder=HARFeatureBuilder(),
        target_builder=RealizedVarianceTargetBuilder(horizon),
        metrics=(QLikeMetric(), LogMSEMetric()),
        comparison_pairs=(("naive", "HAR"),),
        candidate_model="HAR",
        baseline_models=("naive", "EWMA", "AR1", "historical_mean"),
        comparison_test=DieboldMarianoTest(min_hac_lags=horizon.sessions - 1),
    )
