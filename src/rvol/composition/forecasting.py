"""Compose the standard realized-volatility forecast experiment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from rvol.application import WalkForwardExperiment
from rvol.domain import ExperimentConfig, ExperimentResult, Forecaster
from rvol.domain.contracts import ForecastMetric
from rvol.evaluation import (
    ForecastEvaluator,
    ForecastRobustnessAnalyzer,
    LogMSEMetric,
    QLikeMetric,
)
from rvol.features.forecasting import build_har_features
from rvol.models import (
    AR1Forecaster,
    EWMAForecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)


@dataclass(frozen=True)
class ExperimentDefinition:
    """Select the components and comparison defaults for a forecast workflow."""

    models: tuple[Forecaster, ...]
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame]
    metrics: tuple[ForecastMetric, ...]
    comparison_pairs: tuple[tuple[str, str], ...]
    candidate_model: str
    baseline_models: tuple[str, ...]

    def __post_init__(self) -> None:
        names = [model.name for model in self.models]
        if not names or len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise ValueError("models must have non-empty, unique names")
        if not self.metrics:
            raise ValueError("at least one metric is required")
        references = {self.candidate_model, *self.baseline_models}
        references.update(name for pair in self.comparison_pairs for name in pair)
        missing = references.difference(names)
        if missing:
            raise ValueError(f"definition references unknown models: {', '.join(sorted(missing))}")

    @property
    def all_comparison_pairs(self) -> tuple[tuple[str, str], ...]:
        """Compare every other configured model with the selected candidate."""
        return tuple(
            (name, self.candidate_model)
            for name in sorted(model.name for model in self.models)
            if name != self.candidate_model
        )

    def run(self, daily: pd.DataFrame, config: ExperimentConfig) -> ExperimentResult:
        """Prepare features and run the selected models on common windows."""
        return WalkForwardExperiment(models=self.models, config=config).run(
            self.feature_builder(daily),
        )

    def evaluator(
        self,
        *,
        comparison_pairs: Sequence[tuple[str, str]] | None = None,
        hac_lags: int | None = None,
    ) -> ForecastEvaluator:
        """Build an evaluator with defaults or explicit comparison overrides."""
        return ForecastEvaluator(
            metrics=self.metrics,
            comparison_pairs=(
                self.comparison_pairs if comparison_pairs is None else comparison_pairs
            ),
            hac_lags=hac_lags,
        )

    def robustness_analyzer(
        self,
        *,
        hac_lags: Sequence[int] = tuple(range(6)),
        top_errors: int = 5,
    ) -> ForecastRobustnessAnalyzer:
        """Build robustness checks for the configured candidate and baselines."""
        return ForecastRobustnessAnalyzer(
            candidate_model=self.candidate_model,
            baseline_models=self.baseline_models,
            metrics=self.metrics,
            hac_lags=hac_lags,
            top_errors=top_errors,
        )


def standard_forecasters() -> tuple[Forecaster, ...]:
    """Return fresh instances of the models used by standard project workflows."""
    return (
        HistoricalMeanForecaster(),
        NaiveForecaster(),
        EWMAForecaster(),
        AR1Forecaster(),
        HARForecaster(),
    )


def standard_experiment_definition() -> ExperimentDefinition:
    """Return fresh components and the standard commands' comparison choices."""
    return ExperimentDefinition(
        models=standard_forecasters(),
        feature_builder=build_har_features,
        metrics=(QLikeMetric(), LogMSEMetric()),
        comparison_pairs=(("naive", "HAR"),),
        candidate_model="HAR",
        baseline_models=("naive", "EWMA", "AR1", "historical_mean"),
    )


def run_standard_forecast_experiment(
    daily: pd.DataFrame,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Build HAR features and run the standard model set on common windows."""
    return standard_experiment_definition().run(daily, config)
