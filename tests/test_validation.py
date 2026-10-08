import json
from dataclasses import FrozenInstanceError
from datetime import date, timedelta
from typing import Any, cast

import pandas as pd
import pytest

from rvol.domain import ExperimentResult, ForecastRecord, RunIdentity
from rvol.forecasting.evaluation import ForecastEvaluator, LogMSEMetric
from rvol.presentation.reporting import ValidationArtifactWriter


def run(fold: str, start: int, count: int, error: float) -> ExperimentResult:
    records = []
    for offset in range(start, start + count):
        day = date(2024, 1, 1) + timedelta(days=offset)
        for model, prediction in (("baseline", -10 + error), ("candidate", -10.0)):
            records.append(ForecastRecord(
                model, day - timedelta(days=1), day, 10, -10.0, prediction,
            ))
    return ExperimentResult(tuple(records), identity=RunIdentity("study", fold))


def evaluator() -> ForecastEvaluator:
    return ForecastEvaluator(
        metrics=(LogMSEMetric(),), comparison_pairs=(("baseline", "candidate"),), hac_lags=0,
    )


def test_overlapping_folds_are_scored_separately_with_explicit_aggregation_weights():
    first, second = run("a", 0, 4, 1), run("b", 2, 6, 3)
    equal = evaluator().evaluate_runs((second, first))
    pooled = evaluator().evaluate_runs((first, second), aggregation="pooled")

    assert [result.identity.fold_id for result in equal.runs] == ["a", "b"]
    assert equal.runs[0] == evaluator().evaluate(first)
    assert equal.runs[1] == evaluator().evaluate(second)
    assert [r.comparisons[0].n_obs for r in equal.runs] == [4, 6]
    baseline_equal = next(item for item in equal.summaries if item.model == "baseline")
    baseline_pooled = next(item for item in pooled.summaries if item.model == "baseline")
    assert baseline_equal.mean_loss == 5.0
    assert baseline_pooled.mean_loss == 5.8
    assert baseline_equal.n_runs == 2
    assert baseline_equal.n_forecasts == 10


def test_validation_artifacts_keep_overlapping_cases_and_are_deterministic(tmp_path):
    first, second = run("a", 0, 4, 1), run("b", 2, 6, 3)
    writer = ValidationArtifactWriter(tmp_path)
    paths = writer.write(evaluator().evaluate_runs((first, second)))
    original = {path.name: path.read_bytes() for path in paths.all()}
    losses = pd.read_csv(paths.losses)
    overlap = losses[(losses.target_date == "2024-01-03") & (losses.model == "baseline")]

    assert set(overlap.fold_id) == {"a", "b"}
    assert set(overlap.loss) == {1.0, 9.0}
    assert set(losses.run_id) == {"study"}
    assert len(pd.read_csv(paths.run_comparisons)) == 2
    metadata = json.loads(paths.metadata.read_text())
    assert metadata["aggregation"] == "equal_run"
    assert metadata["statistical_tests"] == "within_each_run_fold_only"
    assert [item["forecast_cases"] for item in metadata["runs"]] == [8, 12]
    writer.write(evaluator().evaluate_runs((second, first)))
    assert {path.name: path.read_bytes() for path in paths.all()} == original


def test_validation_rejects_duplicate_identities_and_incompatible_model_sets():
    first = run("a", 0, 4, 1)
    with pytest.raises(ValueError, match="unique run/fold identities"):
        evaluator().evaluate_runs((first, first))
    with pytest.raises(ValueError, match="non-empty"):
        evaluator().evaluate_runs(())
    different = ExperimentResult(
        tuple(record for record in run("b", 2, 6, 3).records if record.model == "baseline"),
        identity=RunIdentity("study", "b"),
    )
    with pytest.raises(ValueError, match="identical models"):
        ForecastEvaluator().evaluate_runs((first, different))


@pytest.mark.parametrize("aggregation", ["unknown"])
def test_validation_requires_a_known_aggregation_rule(aggregation):
    with pytest.raises(ValueError, match="aggregation must be"):
        evaluator().evaluate_runs((run("a", 0, 4, 1),), aggregation=aggregation)


@pytest.mark.parametrize("run_id, fold_id", [("", None), ("study", ""), (" ", "fold")])
def test_run_identities_validate_names(run_id, fold_id):
    with pytest.raises(ValueError):
        RunIdentity(run_id, fold_id)


def test_run_identities_are_immutable():
    identity = RunIdentity("study", "fold")
    with pytest.raises(FrozenInstanceError):
        cast(Any, identity).run_id = "changed"
