"""Model-agnostic evaluation of normalized forecast experiment results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from rvol.domain.contracts import ForecastMetric
from rvol.domain.results import ExperimentResult, ForecastRecord, ModelLossSummary
from rvol.evaluation.comparison import diebold_mariano
from rvol.evaluation.metrics import LogMSEMetric, QLikeMetric


@dataclass(frozen=True)
class LossRecord:
    """One model's loss for one metric and target session."""

    model: str
    metric: str
    target_date: date
    loss: float


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
    dm_statistic: float
    p_value: float
    hac_lags: int


@dataclass(frozen=True)
class EvaluationResult:
    """Per-date losses, model summaries, and requested comparisons."""

    losses: tuple[LossRecord, ...]
    summaries: tuple[ModelLossSummary, ...]
    comparisons: tuple[PairwiseComparison, ...]


class ForecastEvaluator:
    """Evaluate all models on aligned targets under injected metrics."""

    def __init__(
        self,
        metrics: Sequence[ForecastMetric] | None = None,
        *,
        comparison_pairs: Sequence[tuple[str, str]] = (("naive", "HAR"),),
        hac_lags: int | None = None,
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
        self.hac_lags = hac_lags

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
        reference_actual = np.array([record.actual_rv for record in reference])
        for model, records in grouped.items():
            if [record.target_date for record in records] != reference_dates:
                raise ValueError("all models must forecast identical target dates")
            actual_log = np.array([record.actual_log_rv for record in records])
            actual = np.array([record.actual_rv for record in records])
            if not np.array_equal(actual_log, reference_actual_log) or not np.array_equal(
                actual, reference_actual
            ):
                raise ValueError(f"actual values disagree across models for {model}")
        return dict(grouped)

    def evaluate(self, experiment: ExperimentResult) -> EvaluationResult:
        """Calculate losses and requested pairwise model comparisons."""
        grouped = self._aligned_records(experiment)
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
                    LossRecord(model, metric.name, record.target_date, float(loss))
                    for record, loss in zip(records, losses, strict=True)
                )

        comparisons: list[PairwiseComparison] = []
        for metric in self.metrics:
            for baseline, candidate in self.comparison_pairs:
                baseline_loss = losses_by_key[(metric.name, baseline)]
                candidate_loss = losses_by_key[(metric.name, candidate)]
                test = diebold_mariano(
                    baseline_loss,
                    candidate_loss,
                    hac_lags=self.hac_lags,
                )
                comparisons.append(PairwiseComparison(
                    metric=metric.name,
                    baseline_model=baseline,
                    candidate_model=candidate,
                    n_obs=test.n_obs,
                    baseline_mean_loss=float(np.mean(baseline_loss)),
                    candidate_mean_loss=float(np.mean(candidate_loss)),
                    mean_loss_difference=test.mean_loss_difference,
                    dm_statistic=test.statistic,
                    p_value=test.p_value,
                    hac_lags=test.hac_lags,
                ))

        return EvaluationResult(
            losses=tuple(loss_records),
            summaries=tuple(summaries),
            comparisons=tuple(comparisons),
        )
