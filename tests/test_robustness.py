import numpy as np
import pandas as pd
import pytest

from rvol.application import WalkForwardExperiment
from rvol.domain import ExperimentConfig
from rvol.evaluation import ForecastRobustnessAnalyzer
from rvol.models import (
    AR1Forecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)
from rvol.reporting import RobustnessArtifactWriter


def experiment_result():
    n_rows = 55
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n_rows + 1)
    matrix = rng.normal(loc=-10.0, scale=0.5, size=(n_rows, 3))
    target = -0.3 + matrix @ np.array([0.5, 0.3, 0.17])
    features = pd.DataFrame({
        "origin_date": dates[:-1],
        "target_date": dates[1:],
        "rv_daily": matrix[:, 0],
        "rv_weekly": matrix[:, 1],
        "rv_monthly": matrix[:, 2],
        "target": target,
    })
    return WalkForwardExperiment(
        models=(
            HistoricalMeanForecaster(),
            NaiveForecaster(),
            AR1Forecaster(),
            HARForecaster(),
        ),
        config=ExperimentConfig(min_train_size=20),
    ).run(features)


def test_robustness_analyzer_calculates_all_predefined_checks():
    result = ForecastRobustnessAnalyzer(
        hac_lags=(0, 1),
        top_errors=2,
    ).analyze(experiment_result())

    assert len(result.hac_comparisons) == 12
    assert {item.hac_lags for item in result.hac_comparisons} == {0, 1}
    assert len(result.win_rates) == 6
    assert all(item.n_obs == 35 for item in result.win_rates)
    assert all(0 <= item.candidate_win_rate <= 1 for item in result.win_rates)
    assert all(item.candidate_wins + item.ties <= item.n_obs for item in result.win_rates)
    assert len(result.monthly_losses) >= 8
    assert {item.metric for item in result.monthly_losses} == {"QLIKE", "log-RV MSE"}
    assert len(result.largest_errors) == 8
    assert {item.model for item in result.largest_errors} == {
        "AR1",
        "HAR",
        "historical_mean",
        "naive",
    }


def test_robustness_writer_is_deterministic(tmp_path):
    result = ForecastRobustnessAnalyzer(
        hac_lags=(0, 1),
        top_errors=2,
    ).analyze(experiment_result())
    writer = RobustnessArtifactWriter(tmp_path)

    paths = writer.write(result)
    first_run = {path.name: path.read_bytes() for path in paths.all()}
    repeated = writer.write(result)

    assert {path.name: path.read_bytes() for path in repeated.all()} == first_run
    assert len(pd.read_csv(paths.hac_sensitivity)) == 12
    assert len(pd.read_csv(paths.win_rates)) == 6
    assert len(pd.read_csv(paths.largest_errors)) == 8
    report = paths.report.read_text()
    assert "# Forecast robustness report" in report
    assert "## Newey–West lag sensitivity" in report
    assert "## Monthly forecast losses" in report


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"baseline_models": ()}, "baseline_models"),
        ({"baseline_models": ("HAR",)}, "must not also"),
        ({"hac_lags": ()}, "hac_lags"),
        ({"hac_lags": (-1,)}, "hac_lags"),
        ({"top_errors": 0}, "top_errors"),
    ],
)
def test_robustness_analyzer_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ForecastRobustnessAnalyzer(**kwargs)
