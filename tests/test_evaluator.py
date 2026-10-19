import numpy as np
import pandas as pd
import pytest

from rvol.evaluation import (
    ComparisonOutcome,
    DieboldMarianoTest,
    MAEMetric,
    MSEMetric,
    evaluate,
)
from rvol.evaluation.comparison import diebold_mariano
from rvol.evaluation.evaluator import validate_evaluation


def sample_forecasts(n=20):
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "model": model,
                    "origin_date": dates - pd.Timedelta(days=1),
                    "target_date": dates,
                    "actual": np.linspace(-2, 3, n),
                    "predicted": np.linspace(-2, 3, n) + error,
                }
            )
            for model, error in (("baseline", 0.5), ("candidate", 0.2))
        ],
        ignore_index=True,
    )


def test_evaluation_returns_tables_with_untransformed_per_case_losses():
    forecasts = sample_forecasts()
    unchanged = forecasts.copy(deep=True)
    result = evaluate(forecasts, metrics=(MSEMetric(), MAEMetric()))
    assert isinstance(result.losses, pd.DataFrame)
    assert isinstance(result.summaries, pd.DataFrame)
    expected = {
        (model, metric): value
        for model, error in (("baseline", 0.5), ("candidate", 0.2))
        for metric, value in (("MSE", error**2), ("MAE", error))
    }
    for row in result.summaries.itertuples():
        assert row.mean_loss == pytest.approx(expected[(str(row.model), str(row.metric))])
        assert row.n_obs == 20
    assert result.comparisons.empty
    pd.testing.assert_frame_equal(forecasts, unchanged)


def test_minimal_custom_metric_and_comparison_need_no_specification():
    class AbsoluteError:
        name = "absolute"

        def losses(self, actual, predicted):
            return np.abs(actual - predicted)

    class CustomTest:
        name = "custom"

        def compare(self, baseline_loss, candidate_loss):
            baseline_loss[:] = 0
            candidate_loss[:] = 0
            return ComparisonOutcome(1.2, 0.1, {"option": 2})

    result = evaluate(
        sample_forecasts(),
        metrics=(AbsoluteError(),),
        comparison_pairs=(("baseline", "candidate"),),
        comparison_test=CustomTest(),
    )
    assert result.comparisons.test_name.tolist() == ["custom"]
    assert result.comparisons.test_parameters.iloc[0] == {"option": 2}
    assert result.losses.loss.min() > 0
    assert not hasattr(AbsoluteError(), "specification")


def test_pairwise_test_matches_the_existing_dm_formula():
    forecasts = sample_forecasts()
    result = evaluate(
        forecasts,
        comparison_pairs=(("baseline", "candidate"),),
        comparison_test=DieboldMarianoTest(hac_lags=0),
    )
    pivot = result.losses.pivot(index="target_date", columns="model", values="loss")
    expected = diebold_mariano(pivot.baseline, pivot.candidate, hac_lags=0)
    actual = result.comparisons.iloc[0]
    assert actual.statistic == expected.statistic
    assert actual.p_value == expected.p_value
    assert actual.mean_loss_difference == pytest.approx((pivot.baseline - pivot.candidate).mean())


def test_default_dm_lag_selection_and_explicit_settings_match_the_numerical_test():
    forecasts = sample_forecasts(20)
    result = evaluate(forecasts, comparison_pairs=(("baseline", "candidate"),))
    pivot = result.losses.pivot(index="target_date", columns="model", values="loss")
    expected = diebold_mariano(pivot.baseline, pivot.candidate)
    assert result.comparisons.test_parameters.iloc[0] == expected.parameters
    explicit = evaluate(
        forecasts,
        comparison_pairs=(("baseline", "candidate"),),
        comparison_test=DieboldMarianoTest(hac_lags=0),
    )
    assert explicit.comparisons.test_parameters.iloc[0]["hac_lags"] == 0


@pytest.mark.parametrize(
    "values",
    [np.array([np.nan]), np.array([-1.0]), np.empty(0), np.ones((20, 1)), np.full(20, np.inf)],
)
def test_invalid_metric_outputs_are_rejected(values):
    class Invalid:
        name = "invalid"

        def losses(self, actual, predicted):
            return values

    with pytest.raises(ValueError, match="finite non-negative"):
        evaluate(sample_forecasts(), metrics=(Invalid(),))


@pytest.mark.parametrize(
    "options",
    [
        {"metrics": ()},
        {"metrics": (MSEMetric(), MSEMetric())},
        {"comparison_pairs": (("missing", "candidate"),)},
        {"comparison_pairs": (("candidate", "candidate"),)},
        {"comparison_pairs": (("baseline", "candidate"), ("baseline", "candidate"))},
    ],
)
def test_invalid_metric_and_comparison_choices_are_rejected(options):
    with pytest.raises(ValueError):
        evaluate(sample_forecasts(), **options)


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate",
        "missing_date",
        "different_actual",
        "different_origin",
        "nonfinite_actual",
        "nonfinite_prediction",
        "empty_model",
        "mixed_asset",
        "invalid_interval",
        "partial_fit_dates",
        "future_fit",
        "empty",
    ],
)
def test_invalid_or_misaligned_forecast_tables_are_rejected(corruption):
    frame = sample_forecasts()
    if corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif corruption == "missing_date":
        frame = frame.drop(index=0)
    elif corruption == "different_actual":
        frame.loc[0, "actual"] += 1
    elif corruption == "different_origin":
        frame.loc[0, "origin_date"] -= pd.Timedelta(days=1)
    elif corruption == "nonfinite_actual":
        frame.loc[0, "actual"] = np.nan
    elif corruption == "nonfinite_prediction":
        frame.loc[0, "predicted"] = np.inf
    elif corruption == "empty_model":
        frame.loc[0, "model"] = ""
    elif corruption == "mixed_asset":
        frame["asset"] = "one"
        frame.loc[0, "asset"] = "two"
    elif corruption == "invalid_interval":
        frame.loc[0, "origin_date"] = frame.loc[0, "target_date"]
    elif corruption == "partial_fit_dates":
        frame["fit_date"] = frame.origin_date
    elif corruption == "future_fit":
        frame["train_start_date"], frame["train_end_date"] = frame.origin_date, frame.origin_date
        frame["fit_date"] = frame.target_date
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        evaluate(frame)


@pytest.mark.parametrize("consumer", ["evaluate", "validate_evaluation"])
@pytest.mark.parametrize("problem", ["unaligned", "mixed_asset"])
def test_evaluation_consumers_enforce_one_asset_and_a_common_sample(consumer, problem):
    frame = sample_forecasts()
    evaluation = evaluate(frame)
    if problem == "unaligned":
        selected = frame.model == "candidate"
        frame.loc[selected, ["origin_date", "target_date"]] += pd.Timedelta(days=30)
        message = "identical dates and actual values"
    else:
        frame["asset"] = "EURUSD"
        frame.loc[0, "asset"] = "GBPUSD"
        message = "each asset separately"

    with pytest.raises(ValueError, match=message):
        if consumer == "evaluate":
            evaluate(frame)
        else:
            validate_evaluation(frame, evaluation)


@pytest.mark.parametrize(
    "corruption", ["prediction", "tiny_prediction", "missing_loss", "duplicate_loss"]
)
def test_reports_cannot_reuse_an_evaluation_of_different_forecasts(corruption):
    forecasts = sample_forecasts()
    evaluation = evaluate(forecasts, comparison_pairs=(("baseline", "candidate"),))
    if corruption == "prediction":
        forecasts.predicted += 1
    elif corruption == "tiny_prediction":
        forecasts.predicted += 1e-10
    elif corruption == "missing_loss":
        evaluation.losses = evaluation.losses.iloc[1:]
    else:
        evaluation.losses = pd.concat([evaluation.losses, evaluation.losses.iloc[:1]])
    with pytest.raises(ValueError):
        validate_evaluation(forecasts, evaluation)
