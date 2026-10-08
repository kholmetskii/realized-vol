"""Score independently identified runs without flattening overlapping folds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from rvol.domain import ExperimentResult
from rvol.evaluation.evaluator import EvaluationResult

if TYPE_CHECKING:
    from rvol.evaluation.evaluator import ForecastEvaluator

LossAggregation = Literal["equal_run", "pooled"]


@dataclass(frozen=True)
class AggregateLossSummary:
    model: str
    metric: str
    n_runs: int
    n_forecasts: int
    mean_loss: float


@dataclass(frozen=True)
class ValidationResult:
    """Per-run evaluation plus descriptive aggregate losses.

    Every run/fold retains its own date alignment and statistical tests. No
    pooled statistical test is performed across overlapping or dependent folds.
    n_forecasts counts forecast cases across folds, not independent observations.
    """

    runs: tuple[EvaluationResult, ...]
    summaries: tuple[AggregateLossSummary, ...]
    aggregation: LossAggregation


def evaluate_runs(
    evaluator: ForecastEvaluator, experiments: Sequence[ExperimentResult], *,
    aggregation: LossAggregation = "equal_run",
) -> ValidationResult:
    if aggregation not in ("equal_run", "pooled"):
        raise ValueError("aggregation must be equal_run or pooled")
    identities = [experiment.identity for experiment in experiments]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("validation runs must be non-empty and have unique run/fold identities")
    if len({experiment.target_horizon for experiment in experiments}) != 1:
        raise ValueError("validation runs must use identical target horizons")
    ordered = sorted(experiments, key=lambda item: (
        item.identity.run_id, item.identity.fold_id or "",
    ))
    runs = tuple(evaluator.evaluate(experiment) for experiment in ordered)
    keys = {(item.model, item.metric) for item in runs[0].summaries}
    if any({(item.model, item.metric) for item in run.summaries} != keys for run in runs):
        raise ValueError("validation runs must contain identical models and metrics")
    summaries = []
    for model, metric in sorted(keys):
        selected = [
            item for run in runs for item in run.summaries
            if (item.model, item.metric) == (model, metric)
        ]
        means = np.array([item.mean_loss for item in selected])
        counts = np.array([item.n_obs for item in selected])
        mean = float(
            np.mean(means) if aggregation == "equal_run" else np.average(means, weights=counts)
        )
        summaries.append(AggregateLossSummary(model, metric, len(runs), int(counts.sum()), mean))
    return ValidationResult(runs, tuple(summaries), aggregation)
