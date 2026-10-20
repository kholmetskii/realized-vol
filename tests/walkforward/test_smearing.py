import numpy as np
import pandas as pd
import pytest

from rvol.evaluation import QLikeMetric, evaluate
from rvol.walkforward import smeared_forecasts


def log_forecasts(actual, predicted, *, model: str = "HAR") -> pd.DataFrame:
    """One-step forecasts on consecutive days: target i is the origin of i + 1."""
    dates = pd.bdate_range("2024-01-01", periods=len(actual) + 1)
    return pd.DataFrame(
        {
            "model": model,
            "origin_date": dates[:-1],
            "target_date": dates[1:],
            "actual": np.asarray(actual, dtype="float64"),
            "predicted": np.asarray(predicted, dtype="float64"),
        }
    )


def test_factor_averages_only_errors_observable_at_each_origin():
    rng = np.random.default_rng(3)
    actual = rng.normal(-10, 0.5, 30)
    predicted = actual - rng.normal(0, 0.5, 30)

    result = smeared_forecasts(log_forecasts(actual, predicted), min_errors=1)

    exp_errors = np.exp(actual - predicted)
    expected = [exp_errors[:i].mean() for i in range(1, 30)]
    assert np.allclose(result["smearing_factor"], expected)
    assert np.allclose(result["actual"], np.exp(actual[1:]))
    assert np.allclose(result["predicted"], np.exp(predicted[1:]) * np.array(expected))


def test_changing_a_later_outcome_leaves_earlier_factors_unchanged():
    actual = np.linspace(-10, -9, 25)
    predicted = actual - 0.1
    changed = actual.copy()
    changed[-1] += 3.0

    original = smeared_forecasts(log_forecasts(actual, predicted), min_errors=5)
    shifted = smeared_forecasts(log_forecasts(changed, predicted), min_errors=5)

    pd.testing.assert_series_equal(original["smearing_factor"], shifted["smearing_factor"])


def test_constant_error_is_removed_exactly():
    actual = np.linspace(-11, -9, 12)
    predicted = actual - 0.3

    result = smeared_forecasts(log_forecasts(actual, predicted), min_errors=3)

    assert np.allclose(result["smearing_factor"], np.exp(0.3))
    assert np.allclose(result["predicted"], result["actual"])


def test_rows_without_enough_observable_errors_are_dropped():
    actual = np.zeros(10)

    result = smeared_forecasts(log_forecasts(actual, actual), min_errors=4)

    assert len(result) == 6
    assert result["target_date"].min() == log_forecasts(actual, actual)["target_date"].iloc[4]


def test_each_model_uses_its_own_errors():
    actual = np.zeros(8)
    frame = pd.concat(
        [
            log_forecasts(actual, actual - 0.2, model="low"),
            log_forecasts(actual, actual + 0.2, model="high"),
        ],
        ignore_index=True,
    )

    result = smeared_forecasts(frame, min_errors=2)

    factors = result.groupby("model")["smearing_factor"].unique()
    assert np.allclose(factors["low"], np.exp(0.2))
    assert np.allclose(factors["high"], np.exp(-0.2))


def test_smearing_recovers_the_mean_of_lognormal_variance():
    sigma, n = 0.5, 20_000
    rng = np.random.default_rng(11)
    mean_log = -10.0
    actual = mean_log + rng.normal(0, sigma, n)
    predicted = np.full(n, mean_log)

    result = smeared_forecasts(log_forecasts(actual, predicted), min_errors=1_000)

    true_mean = np.exp(mean_log + sigma**2 / 2)
    assert np.exp(mean_log) / true_mean == pytest.approx(np.exp(-(sigma**2) / 2))
    assert result["predicted"].iloc[-1] == pytest.approx(true_mean, rel=0.02)
    assert result["actual"].mean() == pytest.approx(true_mean, rel=0.02)


def test_smearing_lowers_qlike_for_unbiased_log_forecasts():
    rng = np.random.default_rng(5)
    actual = -10 + rng.normal(0, 0.5, 400)
    frame = log_forecasts(actual, np.full(400, -10.0))

    smeared = smeared_forecasts(frame, min_errors=50)
    naive = smeared["predicted"] / smeared["smearing_factor"]

    actual = smeared["actual"].to_numpy()
    qlike = QLikeMetric()
    assert qlike.losses(actual, smeared["predicted"].to_numpy()).mean() < (
        qlike.losses(actual, naive.to_numpy()).mean()
    )


def test_output_can_be_evaluated_with_variance_scale_qlike():
    actual = np.linspace(-10, -9, 30)
    frame = pd.concat(
        [
            log_forecasts(actual, actual - 0.1, model="A"),
            log_forecasts(actual, actual + 0.2, model="B"),
        ],
        ignore_index=True,
    )

    smeared = smeared_forecasts(frame, min_errors=5)
    evaluation = evaluate(smeared, metrics=(QLikeMetric(),), comparison_pairs=[("A", "B")])

    actual_values = smeared["actual"].to_numpy()
    expected = QLikeMetric().losses(actual_values, smeared["predicted"].to_numpy())
    assert np.allclose(evaluation.losses.sort_values(["model", "target_date"])["loss"], expected)


@pytest.mark.parametrize("min_errors", [0, -1, 1.5, True])
def test_min_errors_must_be_a_positive_integer(min_errors):
    actual = np.zeros(5)
    with pytest.raises(ValueError, match="min_errors"):
        smeared_forecasts(log_forecasts(actual, actual), min_errors=min_errors)


def test_too_short_history_is_rejected():
    actual = np.zeros(5)
    with pytest.raises(ValueError, match="enough observable errors"):
        smeared_forecasts(log_forecasts(actual, actual), min_errors=10)
