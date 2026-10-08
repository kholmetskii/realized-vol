"""Characterization tests for the canonical forecasting pipeline.

These assertions intentionally lock the current end-to-end behavior. Update
the golden values only when a numerical or alignment change is deliberate.
HAR expectations include the variance correction from each training window.
"""

import numpy as np
import pandas as pd
import pytest

from rvol.application import WalkForwardExperiment
from rvol.domain import ExperimentConfig
from rvol.evaluation import ForecastEvaluator
from rvol.features.forecasting import build_har_features
from rvol.models import HARForecaster, NaiveForecaster


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
    experiment = WalkForwardExperiment(
        models=(NaiveForecaster(), HARForecaster()),
        config=ExperimentConfig(min_train_size=25),
    ).run(features)
    evaluation = ForecastEvaluator(
        comparison_pairs=(("naive", "HAR"),), hac_lags=3,
    ).evaluate(experiment)
    return features, experiment, evaluation


def test_walk_forward_pipeline_matches_current_alignment_and_predictions(
    characterized_pipeline,
):
    features, experiment, _ = characterized_pipeline

    assert len(features) == 68
    assert features.loc[0, "origin_date"] == pd.Timestamp("2020-01-31")
    assert features.loc[0, "target_date"] == pd.Timestamp("2020-02-03")
    assert features.loc[67, "origin_date"] == pd.Timestamp("2020-05-05")
    assert features.loc[67, "target_date"] == pd.Timestamp("2020-05-06")

    by_model = {
        model: [record for record in experiment.records if record.model == model]
        for model in experiment.models
    }
    naive = by_model["naive"]
    har = by_model["HAR"]

    assert experiment.n_forecasts == 86
    assert len(naive) == len(har) == 43
    assert har[0].origin_date == pd.Timestamp("2020-03-06").date()
    assert har[0].target_date == pd.Timestamp("2020-03-09").date()
    assert har[42].origin_date == pd.Timestamp("2020-05-05").date()
    assert har[42].target_date == pd.Timestamp("2020-05-06").date()
    np.testing.assert_array_equal([record.n_train for record in har], np.arange(25, 68))

    selected = np.array([
        [har[index].actual_log_rv, naive[index].predicted_log_rv, har[index].predicted_log_rv]
        for index in (0, 21, 42)
    ])
    expected = np.array([
        [-9.757338569918984, -9.654409842525581, -9.81262938523659],
        [-10.195348190910776, -10.115053704270585, -10.18040187219842],
        [-9.968873206979659, -9.9355616903085, -10.009187806732454],
    ])
    np.testing.assert_allclose(selected, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        [
            sum(record.predicted_log_rv for record in har),
            sum(record.predicted_log_rv for record in naive),
            sum(record.actual_log_rv for record in har),
        ],
        [-425.4709423658551, -424.8527856107647, -425.16724897521885],
        rtol=1e-10,
        atol=1e-10,
    )


def test_statistical_comparison_matches_current_pipeline(characterized_pipeline):
    _, _, evaluation = characterized_pipeline
    by_metric = {result.metric: result for result in evaluation.comparisons}

    qlike = by_metric["QLIKE"]
    assert qlike.n_obs == 43
    assert qlike.hac_lags == 3
    np.testing.assert_allclose(
        [
            qlike.baseline_mean_loss,
            qlike.candidate_mean_loss,
            qlike.mean_loss_difference,
            qlike.dm_statistic,
            qlike.p_value,
        ],
        [
            0.002657592022619984,
            0.00029172650971693394,
            0.00236586551290305,
            4.779358110184867,
            8.792786297935597e-07,
        ],
        rtol=1e-9,
        atol=1e-12,
    )

    log_mse = by_metric["log-RV MSE"]
    assert log_mse.n_obs == 43
    assert log_mse.hac_lags == 3
    np.testing.assert_allclose(
        [
            log_mse.baseline_mean_loss,
            log_mse.candidate_mean_loss,
            log_mse.mean_loss_difference,
            log_mse.dm_statistic,
            log_mse.p_value,
        ],
        [
            0.005315602810634525,
            0.0005788184668746995,
            0.004736784343759825,
            4.916822639473957,
            4.398008759862686e-07,
        ],
        rtol=1e-9,
        atol=1e-12,
    )
