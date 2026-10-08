from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from rvol.application import WalkForwardExperiment
from rvol.domain import ExperimentConfig, ExperimentResult
from rvol.evaluation import ForecastEvaluator, ModelLossSummary
from rvol.models import (
    AR1Forecaster,
    EWMAForecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)


def feature_sample(n: int = 50, seed: int = 29) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n + 1)
    matrix = rng.normal(loc=-10.0, scale=0.5, size=(n, 3))
    target = -0.4 + matrix @ np.array([0.5, 0.3, 0.15])
    return pd.DataFrame({
        "origin_date": dates[:-1],
        "target_date": dates[1:],
        "rv_daily": matrix[:, 0],
        "rv_weekly": matrix[:, 1],
        "rv_monthly": matrix[:, 2],
        "target": target,
    })


def experiment_result(
    *,
    forecast_start: date | None = None,
    forecast_end: date | None = None,
) -> ExperimentResult:
    experiment = WalkForwardExperiment(
        models=(
            HistoricalMeanForecaster(),
            NaiveForecaster(),
            EWMAForecaster(),
            AR1Forecaster(),
            HARForecaster(),
        ),
        config=ExperimentConfig(
            min_train_size=20,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
        ),
    )
    return experiment.run(feature_sample())


def test_evaluator_calculates_both_metrics_for_every_model():
    evaluation = ForecastEvaluator(hac_lags=3).evaluate(experiment_result())

    assert len(evaluation.losses) == 300
    assert len(evaluation.summaries) == 10
    assert {
        (summary.model, summary.metric, summary.n_obs)
        for summary in evaluation.summaries
    } == {
        (model, metric, 30)
        for model in ("AR1", "EWMA", "HAR", "historical_mean", "naive")
        for metric in ("QLIKE", "log-RV MSE")
    }
    assert len(evaluation.comparisons) == 2
    for comparison in evaluation.comparisons:
        assert comparison.baseline_model == "naive"
        assert comparison.candidate_model == "HAR"
        assert comparison.candidate_mean_loss < comparison.baseline_mean_loss
        assert comparison.mean_loss_difference > 0
        assert comparison.p_value < 0.01
        assert comparison.hac_lags == 3


def test_evaluator_supports_additional_pairwise_comparisons():
    evaluator = ForecastEvaluator(
        comparison_pairs=(("naive", "HAR"), ("AR1", "HAR")),
        hac_lags=2,
    )

    evaluation = evaluator.evaluate(experiment_result())

    assert len(evaluation.comparisons) == 4
    assert {
        (comparison.baseline_model, comparison.candidate_model)
        for comparison in evaluation.comparisons
    } == {("naive", "HAR"), ("AR1", "HAR")}


def test_validation_and_test_evaluations_remain_date_disjoint():
    features = feature_sample()
    validation_start = cast(pd.Timestamp, features.loc[25, "target_date"]).date()
    validation_end = cast(pd.Timestamp, features.loc[34, "target_date"]).date()
    test_start = cast(pd.Timestamp, features.loc[35, "target_date"]).date()
    validation = ForecastEvaluator().evaluate(
        experiment_result(forecast_start=validation_start, forecast_end=validation_end)
    )
    final_test = ForecastEvaluator().evaluate(
        experiment_result(forecast_start=test_start)
    )

    validation_dates = {record.target_date for record in validation.losses}
    test_dates = {record.target_date for record in final_test.losses}
    assert max(validation_dates) < min(test_dates)
    assert validation_dates.isdisjoint(test_dates)


def test_evaluator_rejects_misaligned_models_and_unknown_comparison_names():
    complete = experiment_result()
    missing_one = ExperimentResult(records=complete.records[:-1])

    with pytest.raises(ValueError, match="identical target dates"):
        ForecastEvaluator().evaluate(missing_one)
    with pytest.raises(ValueError, match="unknown models"):
        ForecastEvaluator(comparison_pairs=(("naive", "missing"),)).evaluate(complete)


def test_evaluator_rejects_an_explicitly_empty_metric_collection():
    with pytest.raises(ValueError, match="at least one"):
        ForecastEvaluator(metrics=())


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"model": ""}, "model"),
        ({"metric": ""}, "metric"),
        ({"n_obs": 0}, "positive"),
        ({"mean_loss": -1.0}, "non-negative"),
        ({"mean_loss": float("nan")}, "finite"),
    ],
)
def test_model_loss_summary_rejects_invalid_evaluation_values(changes, message):
    values = {"model": "HAR", "metric": "QLIKE", "n_obs": 20, "mean_loss": 0.1}
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        ModelLossSummary(**cast(Any, values))
