"""Characterization tests for the pre-refactor forecasting pipeline.

These assertions intentionally lock the current end-to-end behavior. Update
the golden values only when a numerical or alignment change is deliberate.
"""

import numpy as np
import pandas as pd
import pytest

from rvol.evaluation.comparison import compare_forecasts
from rvol.evaluation.walk_forward import FORECAST_COLUMNS, walk_forward_forecasts
from rvol.features.forecasting import build_har_features


@pytest.fixture(scope="module")
def characterized_pipeline():
    n_sessions = 90
    step = np.arange(n_sessions, dtype="float64")
    log_rv = (
        -10.0
        + 0.3 * np.sin(step / 3)
        + 0.15 * np.cos(step / 7)
        + 0.002 * step
    )
    daily = pd.DataFrame({
        "date": pd.bdate_range("2020-01-02", periods=n_sessions),
        "rv_5min": np.exp(log_rv),
        "log_rv": log_rv,
        "observation_count": np.full(n_sessions, 288),
    })
    features = build_har_features(daily)
    forecasts = walk_forward_forecasts(features, min_train_size=25)
    comparisons = compare_forecasts(forecasts, hac_lags=3)
    return features, forecasts, comparisons


def test_walk_forward_pipeline_matches_current_alignment_and_predictions(
    characterized_pipeline,
):
    features, forecasts, _ = characterized_pipeline

    assert len(features) == 68
    assert features.loc[0, "origin_date"] == pd.Timestamp("2020-01-31")
    assert features.loc[0, "target_date"] == pd.Timestamp("2020-02-03")
    assert features.loc[67, "origin_date"] == pd.Timestamp("2020-05-05")
    assert features.loc[67, "target_date"] == pd.Timestamp("2020-05-06")

    assert list(forecasts.columns) == FORECAST_COLUMNS
    assert len(forecasts) == 43
    assert forecasts.loc[0, "origin_date"] == pd.Timestamp("2020-03-06")
    assert forecasts.loc[0, "target_date"] == pd.Timestamp("2020-03-09")
    assert forecasts.loc[42, "origin_date"] == pd.Timestamp("2020-05-05")
    assert forecasts.loc[42, "target_date"] == pd.Timestamp("2020-05-06")
    np.testing.assert_array_equal(forecasts["n_train"], np.arange(25, 68))

    selected = forecasts.loc[
        [0, 21, 42],
        ["actual_log_rv", "naive_log_prediction", "har_log_prediction"],
    ].to_numpy()
    expected = np.array([
        [-9.757338569918984, -9.654409842525581, -9.812760345626966],
        [-10.195348190910776, -10.115053704270585, -10.1806092207242],
        [-9.968873206979659, -9.9355616903085, -10.009376565223356],
    ])
    np.testing.assert_allclose(selected, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        [
            forecasts["har_log_prediction"].sum(),
            forecasts["naive_log_prediction"].sum(),
            forecasts["actual_log_rv"].sum(),
        ],
        [-425.47912683545536, -424.8527856107647, -425.16724897521885],
        rtol=1e-10,
        atol=1e-10,
    )


def test_statistical_comparison_matches_current_pipeline(characterized_pipeline):
    _, _, comparisons = characterized_pipeline
    by_metric = {result.metric: result for result in comparisons}

    qlike = by_metric["QLIKE"]
    assert qlike.n_obs == 43
    assert qlike.hac_lags == 3
    np.testing.assert_allclose(
        [
            qlike.naive_mean_loss,
            qlike.har_mean_loss,
            qlike.mean_loss_difference,
            qlike.dm_statistic,
            qlike.p_value,
        ],
        [
            0.002657592022619984,
            0.0002929614467149504,
            0.0023646305759050337,
            4.77821783149408,
            8.842786335854546e-07,
        ],
        rtol=1e-9,
        atol=1e-12,
    )

    log_mse = by_metric["log-RV MSE"]
    assert log_mse.n_obs == 43
    assert log_mse.hac_lags == 3
    np.testing.assert_allclose(
        [
            log_mse.naive_mean_loss,
            log_mse.har_mean_loss,
            log_mse.mean_loss_difference,
            log_mse.dm_statistic,
            log_mse.p_value,
        ],
        [
            0.005315602810634525,
            0.0005811821692075719,
            0.004734420641426953,
            4.915974016022684,
            4.417105771101206e-07,
        ],
        rtol=1e-9,
        atol=1e-12,
    )
