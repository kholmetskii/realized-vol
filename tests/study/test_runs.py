import pandas as pd
import pytest

from rvol import study
from rvol.data.layout import DataLayout
from tests.integration.test_forecasting_pipeline import daily_series, specification


def test_a_run_is_the_study_computed_step_by_step():
    daily = daily_series()

    run = study.study_run(daily, asset="test")

    prepared = study.prepare(daily)
    calibration = study.ridge_calibration(prepared)
    models = study.calibrated_models(calibration)
    expected = study.forecast(prepared, models, asset="test")
    assert run.asset == "test" and run.daily is daily
    pd.testing.assert_frame_equal(run.prepared, prepared)
    pd.testing.assert_series_equal(run.calibration, calibration)
    assert specification(run.models) == specification(models)
    pd.testing.assert_frame_equal(run.forecasts, expected)
    pd.testing.assert_frame_equal(run.comparison.summary, study.compare(expected).summary)


def test_the_chosen_ridge_penalty_is_the_calibration_minimum():
    run = study.study_run(daily_series(), asset="test")

    ridge = next(model for model in run.models if model.name == "HAR-ridge")

    assert vars(ridge.rule)["ridge"] == run.calibration.idxmin()


def test_running_an_asset_needs_its_tick_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="no tick files"):
        study.run_study("EURUSD", DataLayout(tmp_path))
