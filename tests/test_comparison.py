import numpy as np
import pandas as pd

from rvol.evaluation.comparison import compare_forecasts, diebold_mariano


def forecast_sample(n: int = 120, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    actual_log = rng.normal(-10.0, 0.6, n)
    common_noise = rng.normal(0.0, 0.15, n)
    naive_log = actual_log + common_noise + rng.normal(0.0, 0.45, n)
    har_log = actual_log + common_noise
    return pd.DataFrame({
        "actual_log_rv": actual_log,
        "naive_log_prediction": naive_log,
        "har_log_prediction": har_log,
        "actual_rv": np.exp(actual_log),
        "naive_rv_prediction": np.exp(naive_log),
        "har_rv_prediction": np.exp(har_log),
    })


def test_comparison_detects_a_better_har_forecast():
    comparisons = compare_forecasts(forecast_sample(), hac_lags=3)

    assert {result.metric for result in comparisons} == {"QLIKE", "log-RV MSE"}
    for result in comparisons:
        assert result.har_mean_loss < result.naive_mean_loss
        assert result.mean_loss_difference > 0
        assert result.dm_statistic > 0
        assert result.p_value < 0.01
        assert result.hac_lags == 3


def test_identical_loss_sequences_give_no_evidence_of_a_difference():
    losses = np.linspace(0.1, 1.0, 20)

    result = diebold_mariano(losses, losses)

    assert result.mean_loss_difference == 0
    assert result.statistic == 0
    assert result.p_value == 0.5


def test_zero_lag_statistic_matches_the_iid_formula():
    naive = np.array([0.8, 1.2, 0.7, 1.5, 0.9, 1.1])
    har = np.array([0.6, 0.9, 0.8, 1.0, 0.7, 0.8])
    differential = naive - har
    expected = differential.mean() / (differential.std(ddof=0) / np.sqrt(len(differential)))

    result = diebold_mariano(naive, har, hac_lags=0)

    assert np.isclose(result.statistic, expected)
