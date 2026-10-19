from datetime import date

import numpy as np
import pandas as pd
import pytest

from rvol.models import AR1Model, EWMAModel, LinearRegressionModel
from rvol.walkforward import WalkForwardPlanner, WalkForwardRunner
from rvol.walkforward.planning import WalkForwardPlan, WalkForwardStep
from tests.forecasting_helpers import benchmark_models, prepared_variance, reference_predictions


@pytest.mark.parametrize("window", ["expanding", "rolling"])
@pytest.mark.parametrize("refit_every", [1, 5, 11])
def test_walk_forward_matches_rebuilt_state_at_every_origin(window, refit_every):
    data = prepared_variance()
    unchanged = data.copy(deep=True)
    size = 25 if window == "rolling" else None
    runner = WalkForwardRunner(
        WalkForwardPlanner(
            window=window,
            window_size=size,
            min_train_size=15,
            refit_every=refit_every,
        )
    )
    for model, columns in benchmark_models():
        actual = runner.run(data, model=model, feature_columns=columns)
        expected = reference_predictions(
            data,
            model,
            feature_columns=columns,
            window_size=size,
            min_train_size=15,
            refit_every=refit_every,
        )
        pd.testing.assert_frame_equal(
            actual[["target_date", "fit_date", "n_train"]],
            expected[["target_date", "fit_date", "n_train"]],
            check_dtype=False,
        )
        np.testing.assert_allclose(actual.predicted, expected.predicted, rtol=1e-13, atol=1e-13)
    pd.testing.assert_frame_equal(data, unchanged)


@pytest.mark.parametrize("window", ["expanding", "rolling"])
def test_reporting_bounds_preserve_the_full_history_refit_anchor(window):
    data = prepared_variance()
    for model, columns in benchmark_models():
        options = dict(
            window=window,
            window_size=25 if window == "rolling" else None,
            min_train_size=15,
            refit_every=5,
        )
        full_runner = WalkForwardRunner(WalkForwardPlanner(**options))
        full = full_runner.run(data, model=model, feature_columns=columns)
        start, end = full.target_date.iloc[7].date(), full.target_date.iloc[-4].date()
        bounded_runner = WalkForwardRunner(
            WalkForwardPlanner(forecast_start=start, forecast_end=end, **options)
        )
        bounded = bounded_runner.run(data, model=model, feature_columns=columns)
        expected = full.loc[full.target_date.between(pd.Timestamp(start), pd.Timestamp(end))]
        pd.testing.assert_frame_equal(bounded, expected.reset_index(drop=True))
        assert (
            bounded_runner.plan(data).execution["schedule_anchor"]
            == full_runner.plan(data).execution["schedule_anchor"]
        )


@pytest.mark.parametrize("changes", [{"asset": ""}, {"asset": " "}])
def test_invalid_asset_labels_are_rejected_without_fitting(changes):
    class NoFit(LinearRegressionModel):
        def fit(self, *args):
            raise AssertionError("invalid options must be rejected before fitting")

    with pytest.raises(ValueError):
        WalkForwardRunner(WalkForwardPlanner(min_train_size=15)).run(
            prepared_variance(),
            model=NoFit(),
            feature_columns=("log_rv_daily",),
            **changes,
        )


def test_rolling_window_can_grow_from_minimum_to_capacity():
    data = prepared_variance(feature_columns=("log_rv_daily",))
    runner = WalkForwardRunner(
        WalkForwardPlanner(
            window="rolling",
            window_size=25,
            min_train_size=10,
        )
    )
    forecasts = runner.run(data, model=AR1Model(), feature_columns=("log_rv_daily",))
    assert forecasts.n_train.iloc[0] == 10
    assert forecasts.n_train.max() == 25
    assert list(forecasts.n_train.iloc[:16]) == list(range(10, 26))


def test_state_updates_do_not_change_recorded_fit_dates_and_training_boundaries():
    data = prepared_variance(feature_columns=())
    result = WalkForwardRunner(WalkForwardPlanner(min_train_size=10, refit_every=5)).run(
        data, model=EWMAModel()
    )
    first = result.iloc[:5]
    assert (
        first.fit_date.nunique() == first.n_train.nunique() == first.train_end_date.nunique() == 1
    )
    assert first.predicted.nunique() > 1


def test_no_forecast_period_returns_an_empty_table_with_defined_columns():
    result = WalkForwardRunner(
        WalkForwardPlanner(
            min_train_size=15,
            forecast_start=date(2050, 1, 1),
        )
    ).run(
        prepared_variance(),
        model=AR1Model(),
        feature_columns=("log_rv_daily",),
    )
    assert result.empty
    assert {"actual", "predicted", "fit_date", "n_train"}.issubset(result.columns)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_predictions_are_rejected(value):
    class Invalid:
        name = "invalid"

        def fit(self, features, target):
            return self

        def predict(self, features):
            return np.full(len(features), value)

    with pytest.raises(ValueError, match="finite prediction"):
        WalkForwardRunner(WalkForwardPlanner(min_train_size=15)).run(
            prepared_variance(), model=Invalid()
        )


def test_target_columns_cannot_be_used_as_model_inputs():
    with pytest.raises(ValueError, match="future targets"):
        WalkForwardRunner(WalkForwardPlanner(min_train_size=15)).run(
            prepared_variance(), model=LinearRegressionModel(), feature_columns=("target",)
        )


def test_reused_runner_keeps_asset_labels_specific_to_each_call():
    runner = WalkForwardRunner(WalkForwardPlanner(min_train_size=15, refit_every=5))
    data = prepared_variance(feature_columns=())
    first = runner.run(data, model=EWMAModel(), asset="EURUSD")
    second = runner.run(data, model=EWMAModel(), asset="GBPUSD")
    default = runner.run(data, model=EWMAModel())

    np.testing.assert_array_equal(first.predicted, second.predicted)
    np.testing.assert_array_equal(first.predicted, default.predicted)
    assert set(first.asset) == {"EURUSD"}
    assert set(second.asset) == {"GBPUSD"}
    assert set(default.asset) == {"default"}


def test_edited_planner_settings_are_applied_and_validated_before_execution():
    data = prepared_variance(feature_columns=("log_rv_daily",))
    model = AR1Model()
    planner = WalkForwardPlanner(min_train_size=10)
    runner = WalkForwardRunner(planner)
    planner.window = "rolling"
    planner.window_size = 25

    actual = runner.run(data, model=model, feature_columns=("log_rv_daily",))
    expected = WalkForwardRunner(
        WalkForwardPlanner(window="rolling", window_size=25, min_train_size=10)
    ).run(data, model=model, feature_columns=("log_rv_daily",))
    pd.testing.assert_frame_equal(actual, expected)

    planner.window_size = 5
    with pytest.raises(ValueError, match="at least min_train_size"):
        runner.run(data, model=model, feature_columns=("log_rv_daily",))


def test_runner_executes_the_supplied_plan_including_warmup_and_observation_updates():
    class FixedPlanner(WalkForwardPlanner):
        def build(self, origins, targets):
            return WalkForwardPlan(
                (
                    WalkForwardStep(
                        position=3,
                        training_rows=slice(1, 3),
                        update_rows=slice(0, 0),
                        refit=True,
                        emit_forecast=False,
                    ),
                    WalkForwardStep(
                        position=4,
                        training_rows=slice(1, 4),
                        update_rows=slice(3, 4),
                        refit=False,
                        emit_forecast=True,
                    ),
                ),
                execution={"method": "fixed_plan"},
            )

    ready = pd.DataFrame(
        {
            "origin_date": pd.date_range("2026-10-01", periods=5),
            "target_date": pd.date_range("2026-10-02", periods=5),
            "target": [2.0, 4.0, 8.0, 16.0, 32.0],
        }
    )
    runner = WalkForwardRunner(FixedPlanner())

    forecasts = runner.run(ready, model=EWMAModel(decay=0.5))

    assert len(forecasts) == 1
    forecast = forecasts.iloc[0]
    assert forecast.predicted == 11.0
    assert forecast.origin_date == pd.Timestamp("2026-10-05")
    assert forecast.target_date == pd.Timestamp("2026-10-06")
    assert forecast.fit_date == pd.Timestamp("2026-10-04")
    assert forecast.n_train == 2
    assert forecast.train_start_date == pd.Timestamp("2026-10-03")
    assert forecast.train_end_date == pd.Timestamp("2026-10-04")
    assert runner.plan(ready).execution == {"method": "fixed_plan"}
    assert forecasts.attrs == {}
