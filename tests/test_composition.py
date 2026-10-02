import numpy as np
import pandas as pd

from rvol.composition import run_standard_forecast_experiment, standard_forecasters
from rvol.domain import ExperimentConfig


def daily_sample(n_sessions: int = 50) -> pd.DataFrame:
    step = np.arange(n_sessions, dtype="float64")
    log_rv = -10.0 + 0.2 * np.sin(step / 4) + 0.002 * step
    return pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=n_sessions),
        "rv_5min": np.exp(log_rv),
        "log_rv": log_rv,
        "observation_count": np.full(n_sessions, 288),
    })


def test_standard_forecasters_define_the_project_model_set():
    assert [model.name for model in standard_forecasters()] == [
        "historical_mean",
        "naive",
        "AR1",
        "HAR",
    ]


def test_standard_experiment_builds_features_and_runs_every_model():
    result = run_standard_forecast_experiment(
        daily_sample(),
        ExperimentConfig(min_train_size=10),
    )

    assert result.models == ("AR1", "HAR", "historical_mean", "naive")
    assert result.n_forecasts == 72
    assert {record.n_train for record in result.records} == set(range(10, 28))
