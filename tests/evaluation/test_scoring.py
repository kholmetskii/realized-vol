import numpy as np
import pandas as pd
import pytest

from rvol.evaluation import compare_models, default_hac_lags, diebold_mariano, forecast_losses
from tests.losses import absolute_error, squared_error


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


@pytest.mark.parametrize(
    "loss, expected", [(squared_error, (0.25, 0.04)), (absolute_error, (0.5, 0.2))]
)
def test_losses_are_computed_per_forecast_without_transformations(loss, expected):
    forecasts = sample_forecasts()
    unchanged = forecasts.copy(deep=True)

    losses = forecast_losses(forecasts, loss)

    assert len(losses) == len(forecasts)
    means = losses.groupby("model").loss.mean()
    assert means["baseline"] == pytest.approx(expected[0])
    assert means["candidate"] == pytest.approx(expected[1])
    assert {"model", "origin_date", "target_date", "series", "actual", "predicted"} <= set(losses)
    pd.testing.assert_frame_equal(forecasts, unchanged)


@pytest.mark.parametrize(
    "values",
    [np.array([np.nan]), np.array([-1.0]), np.empty(0), np.ones((40, 1)), np.full(40, np.inf)],
)
def test_invalid_loss_outputs_are_rejected(values):
    with pytest.raises(ValueError, match="finite non-negative"):
        forecast_losses(sample_forecasts(), lambda actual, predicted: values)


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
        "invalid_interval",
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
    elif corruption == "invalid_interval":
        frame.loc[0, "origin_date"] = frame.loc[0, "target_date"]
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        forecast_losses(frame, squared_error)


@pytest.mark.parametrize("problem", ["unaligned", "mixed_series"])
def test_losses_require_one_series_and_a_common_sample(problem):
    frame = sample_forecasts()
    if problem == "unaligned":
        selected = frame.model == "candidate"
        frame.loc[selected, ["origin_date", "target_date"]] += pd.Timedelta(days=30)
        message = "identical dates and actual values"
    else:
        frame["series"] = "one"
        frame.loc[0, "series"] = "two"
        message = "each series separately"

    with pytest.raises(ValueError, match=message):
        forecast_losses(frame, squared_error)


def test_comparison_reports_mean_losses_and_the_dm_test_against_the_candidate():
    losses = forecast_losses(sample_forecasts(), squared_error)

    summary = compare_models(losses, "candidate", hac_lags=0).set_index("model")

    pivot = losses.pivot(index="target_date", columns="model", values="loss")
    statistic, p_value = diebold_mariano(pivot.baseline, pivot.candidate, hac_lags=0)
    assert summary.loc["baseline", "dm_statistic"] == statistic
    assert summary.loc["baseline", "p_value"] == p_value
    assert summary.loc["baseline", "mean_loss"] == pytest.approx(0.25)
    assert summary.loc["candidate", "mean_loss"] == pytest.approx(0.04)
    assert summary.loc["candidate", ["dm_statistic", "p_value"]].isna().all()
    assert (summary.n_obs == 20).all()


def test_comparison_uses_the_default_hac_lags_unless_told_otherwise():
    rng = np.random.default_rng(9)
    forecasts = sample_forecasts(60)
    forecasts["predicted"] += rng.normal(0, 0.3, len(forecasts))
    losses = forecast_losses(forecasts, squared_error)
    pivot = losses.pivot(index="target_date", columns="model", values="loss")

    default = compare_models(losses, "candidate").set_index("model")

    _, expected = diebold_mariano(pivot.baseline, pivot.candidate, hac_lags=default_hac_lags(60))
    assert default.loc["baseline", "p_value"] == expected


def test_comparison_needs_a_known_candidate():
    with pytest.raises(ValueError, match="unknown candidate"):
        compare_models(forecast_losses(sample_forecasts(), squared_error), "missing")
