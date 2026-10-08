"""Model-agnostic evaluation of normalized forecast experiment results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from rvol.domain.contracts import ForecastComparisonTest, ForecastMetric
from rvol.domain.identity import RunIdentity
from rvol.domain.results import ExperimentResult, ForecastRecord
from rvol.domain.specifications import ComponentSpecification, ParameterValue
from rvol.domain.targets import TargetHorizon
from rvol.evaluation.comparison import DieboldMarianoTest
from rvol.evaluation.metrics import LogMSEMetric, QLikeMetric

if TYPE_CHECKING:
    from rvol.evaluation.validation import LossAggregation, ValidationResult


@dataclass(frozen=True)
class LossRecord:
    """One model's loss for one metric and target session."""

    model: str
    metric: str
    target_date: date
    loss: float
    origin_date: date | None = None
    target_start_date: date | None = None


@dataclass(frozen=True)
class ModelLossSummary:
    """Mean out-of-sample loss for one model and metric."""

    model: str
    metric: str
    n_obs: int
    mean_loss: float

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if self.n_obs < 1:
            raise ValueError("n_obs must be positive")
        if not np.isfinite(self.mean_loss) or self.mean_loss < 0:
            raise ValueError("mean_loss must be finite and non-negative")


@dataclass(frozen=True)
class PairwiseComparison:
    """One-sided test that a candidate has lower loss than a baseline."""

    metric: str
    baseline_model: str
    candidate_model: str
    n_obs: int
    baseline_mean_loss: float
    candidate_mean_loss: float
    mean_loss_difference: float
    test_name: str
    statistic: float
    p_value: float
    test_parameters: tuple[tuple[str, ParameterValue], ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Per-date losses, model summaries, and requested comparisons."""

    losses: tuple[LossRecord, ...]
    summaries: tuple[ModelLossSummary, ...]
    comparisons: tuple[PairwiseComparison, ...]
    comparison_test: ComponentSpecification | None = None
    identity: RunIdentity = field(default_factory=RunIdentity)
    target_horizon: TargetHorizon = field(default_factory=TargetHorizon)
    instrument: str | None = None


class ForecastEvaluator:
    """Evaluate all models on aligned targets under injected metrics."""

    def __init__(
        self,
        metrics: Sequence[ForecastMetric] | None = None,
        *,
        comparison_pairs: Sequence[tuple[str, str]] = (),
        hac_lags: int | None = None,
        comparison_test: ForecastComparisonTest | None = None,
    ) -> None:
        selected_metrics = (
            (QLikeMetric(), LogMSEMetric())
            if metrics is None
            else tuple(metrics)
        )
        if not selected_metrics:
            raise ValueError("at least one forecast metric is required")
        metric_names = [metric.name for metric in selected_metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("forecast metric names must be unique")
        if any(metric.scale not in {"variance", "log_variance"} for metric in selected_metrics):
            raise ValueError("forecast metric scale must be variance or log_variance")
        pairs = tuple(comparison_pairs)
        if any(baseline == candidate for baseline, candidate in pairs):
            raise ValueError("comparison models must be different")
        self.metrics = selected_metrics
        self.comparison_pairs = pairs
        if comparison_test is not None and hac_lags is not None:
            raise ValueError("configure hac_lags on the injected comparison test")
        self.comparison_test = (
            DieboldMarianoTest(hac_lags) if comparison_test is None else comparison_test
        )
        self._uses_default_test = comparison_test is None
        self._default_hac_lags = hac_lags

    def evaluate_runs(
        self, experiments: Sequence[ExperimentResult], *,
        aggregation: LossAggregation = "equal_run",
    ) -> ValidationResult:
        """Evaluate each identified run/fold independently, then aggregate losses."""
        from rvol.evaluation.validation import evaluate_runs

        return evaluate_runs(self, experiments, aggregation=aggregation)

    @staticmethod
    def _aligned_records(
        experiment: ExperimentResult,
    ) -> dict[str, list[ForecastRecord]]:
        if not experiment.records:
            raise ValueError("experiment must contain forecast records")
        grouped: dict[str, list[ForecastRecord]] = defaultdict(list)
        for record in experiment.records:
            grouped[record.model].append(record)
        for records in grouped.values():
            records.sort(key=lambda record: record.target_date)
            dates = [record.target_date for record in records]
            if len(dates) != len(set(dates)):
                raise ValueError("each model must have one forecast per target date")

        reference_model = sorted(grouped)[0]
        reference = grouped[reference_model]
        reference_dates = [record.target_date for record in reference]
        reference_actual_log = np.array([record.actual_log_rv for record in reference])
        for model, records in grouped.items():
            if [record.target_date for record in records] != reference_dates:
                raise ValueError("all models must forecast identical target dates")
            if [(r.origin_date, r.target_start) for r in records] != [
                (r.origin_date, r.target_start) for r in reference
            ]:
                raise ValueError("all models must use identical forecast origins and target starts")
            actual_log = np.array([record.actual_log_rv for record in records])
            if not np.array_equal(actual_log, reference_actual_log):
                raise ValueError(f"actual values disagree across models for {model}")
        return dict(grouped)

    def evaluate(self, experiment: ExperimentResult) -> EvaluationResult:
        """Calculate losses and requested pairwise model comparisons."""
        grouped = self._aligned_records(experiment)
        comparison_test = self.comparison_test
        if self._uses_default_test:
            comparison_test = DieboldMarianoTest(
                hac_lags=self._default_hac_lags,
                min_hac_lags=experiment.target_horizon.sessions - 1,
            )
        test_specification = comparison_test.specification
        model_names = set(grouped)
        for baseline, candidate in self.comparison_pairs:
            missing = {baseline, candidate}.difference(model_names)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"comparison references unknown models: {names}")

        loss_records: list[LossRecord] = []
        summaries: list[ModelLossSummary] = []
        losses_by_key: dict[tuple[str, str], np.ndarray] = {}
        for metric in self.metrics:
            for model in sorted(grouped):
                records = grouped[model]
                if metric.scale == "variance":
                    actual = np.array([record.actual_rv for record in records])
                    predicted = np.array([record.predicted_rv for record in records])
                else:
                    actual = np.array([record.actual_log_rv for record in records])
                    predicted = np.array([record.predicted_log_rv for record in records])
                losses = np.asarray(metric.losses(actual, predicted), dtype="float64")
                if losses.shape != (len(records),) or not np.all(np.isfinite(losses)):
                    raise ValueError(f"{metric.name} must return one finite loss per forecast")
                if np.any(losses < 0):
                    raise ValueError(f"{metric.name} must not return negative losses")
                losses_by_key[(metric.name, model)] = losses
                summaries.append(ModelLossSummary(
                    model=model,
                    metric=metric.name,
                    n_obs=len(records),
                    mean_loss=float(np.mean(losses)),
                ))
                loss_records.extend(
                    LossRecord(
                        model, metric.name, record.target_date, float(loss),
                        origin_date=record.origin_date, target_start_date=record.target_start,
                    )
                    for record, loss in zip(records, losses, strict=True)
                )

        comparisons: list[PairwiseComparison] = []
        for metric in self.metrics:
            for baseline, candidate in self.comparison_pairs:
                baseline_loss = losses_by_key[(metric.name, baseline)]
                candidate_loss = losses_by_key[(metric.name, candidate)]
                test = comparison_test.compare(baseline_loss.copy(), candidate_loss.copy())
                comparisons.append(PairwiseComparison(
                    metric=metric.name,
                    baseline_model=baseline,
                    candidate_model=candidate,
                    n_obs=len(baseline_loss),
                    baseline_mean_loss=float(np.mean(baseline_loss)),
                    candidate_mean_loss=float(np.mean(candidate_loss)),
                    mean_loss_difference=float(np.mean(baseline_loss - candidate_loss)),
                    test_name=test_specification.name,
                    statistic=test.statistic,
                    p_value=test.p_value,
                    test_parameters=test.parameters,
                ))

        return EvaluationResult(
            losses=tuple(loss_records),
            summaries=tuple(summaries),
            comparisons=tuple(comparisons),
            comparison_test=test_specification if comparisons else None,
            identity=experiment.identity,
            target_horizon=experiment.target_horizon,
            instrument=None if experiment.config is None else experiment.config.instrument,
        )
