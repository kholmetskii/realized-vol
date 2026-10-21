from datetime import date

import numpy as np
import pandas as pd
import pytest

from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.models import ExponentialSmoothing, ForecastModel, LinearRegression
from rvol.walkforward import FORECAST_COLUMNS, walk_forward
from tests.forecasting_helpers import (
    ALL_FEATURES,
    benchmark_models,
    daily_sample,
    har_features,
    prepared_variance,
    reference_predictions,
)


def test_forecasts_match_a_slow_rebuild_from_date_filters():
    data = prepared_variance()
    unchanged = data.copy(deep=True)
    for model in benchmark_models():
        actual = walk_forward(data, model, min_train_size=15)
        expected = reference_predictions(data, model, min_train_size=15)
        assert actual.target_date.tolist() == expected.target_date.tolist()
        assert actual.n_train.tolist() == expected.n_train.tolist()
        np.testing.assert_allclose(actual.predicted, expected.predicted, rtol=1e-13, atol=1e-13)
    pd.testing.assert_frame_equal(data, unchanged)


def test_changing_future_data_cannot_change_earlier_forecasts():
    daily = daily_sample()
    changed = daily.copy()
    changed.loc[65:, "log_rv"] += 1
    cutoff = daily.date.iloc[65]
    for model in benchmark_models():

        def run(frame, model=model):
            prepared = prepare_forecast_data(
                har_features(model.regressors)(frame),
                build_next_observation_targets(frame, value_col="log_rv"),
            )
            return walk_forward(prepared, model, min_train_size=15)

        first, second = run(daily), run(changed)
        pd.testing.assert_series_equal(
            first.loc[first.origin_date < cutoff, "predicted"],
            second.loc[second.origin_date < cutoff, "predicted"],
        )


def test_training_uses_exactly_the_outcomes_observed_at_each_origin():
    data = prepared_variance(feature_columns=("log_rv_daily",))

    ar1 = ForecastModel("AR1", LinearRegression(), ("log_rv_daily",))
    forecasts = walk_forward(data, ar1, min_train_size=10)

    assert forecasts.n_train.iloc[0] == 10
    for row in forecasts.itertuples():
        assert row.n_train == (data.target_date <= row.origin_date).sum()


def test_bounds_select_target_dates_without_changing_forecasts():
    data = prepared_variance()
    model = ForecastModel("HAR", LinearRegression(), ALL_FEATURES)
    full = walk_forward(data, model, min_train_size=15)
    start, end = full.target_date.iloc[7].date(), full.target_date.iloc[-4].date()

    bounded = walk_forward(data, model, min_train_size=15, start=start, end=end)

    expected = full.loc[full.target_date.between(pd.Timestamp(start), pd.Timestamp(end))]
    pd.testing.assert_frame_equal(bounded, expected.reset_index(drop=True))


def test_no_forecast_period_returns_an_empty_table_with_the_forecast_columns():
    result = walk_forward(
        prepared_variance(),
        ForecastModel("historical_mean", LinearRegression()),
        min_train_size=15,
        start=date(2050, 1, 1),
    )

    assert result.empty
    assert list(result.columns) == list(FORECAST_COLUMNS)


def test_rows_are_labelled_with_the_model_and_series():
    ewma = ForecastModel("EWMA", ExponentialSmoothing(decay=0.94))
    forecasts = walk_forward(prepared_variance(), ewma, min_train_size=15, series="EURUSD")

    assert set(forecasts.model) == {"EWMA"}
    assert set(forecasts.series) == {"EURUSD"}


def test_regressors_reach_the_rule_in_the_model_s_order():
    class Nonlinear:
        def fit(self, features, target):
            residuals = target - features[:, 0] ** 2 - 2 * features[:, 1]
            self.intercept = float(np.mean(residuals))
            return self

        def predict(self, features):
            return self.intercept + features[:, 0] ** 2 + 2 * features[:, 1]

    dates = pd.bdate_range("2024-01-01", periods=41)
    first, second = np.random.default_rng(12).normal(size=(2, 40))
    ready = pd.DataFrame(
        {
            "origin_date": dates[:-1],
            "target_date": dates[1:],
            "target": 7 + second**2 + 2 * first,
            "first": first,
            "second": second,
        }
    )

    model = ForecastModel("nonlinear", Nonlinear(), ("second", "first"))
    forecasts = walk_forward(ready, model, min_train_size=5)

    np.testing.assert_allclose(forecasts.predicted, forecasts.actual, atol=1e-12)


def test_a_rule_cannot_corrupt_later_training_by_mutating_its_arrays():
    class MutatingMean:
        def fit(self, features, target):
            self.mean = float(np.mean(target))
            features[:] = 1e9
            target[:] = 1e9
            return self

        def predict(self, features):
            features[:] = 1e9
            return np.full(len(features), self.mean)

    dates = pd.bdate_range("2024-01-01", periods=21)
    ready = pd.DataFrame(
        {
            "origin_date": dates[:-1],
            "target_date": dates[1:],
            "target": np.arange(20, dtype=float),
            "input": np.arange(20, dtype=float),
        }
    )
    unchanged = ready.copy(deep=True)

    model = ForecastModel("mutating_mean", MutatingMean(), ("input",))
    forecasts = walk_forward(ready, model, min_train_size=3)

    for row in forecasts.itertuples():
        assert row.predicted == pytest.approx(
            ready.loc[ready.target_date <= row.origin_date].target.mean()
        )
    pd.testing.assert_frame_equal(ready, unchanged)


def test_reusing_a_model_object_gives_identical_forecasts():
    data = prepared_variance()
    model = ForecastModel("EWMA", ExponentialSmoothing(decay=0.8))
    first = walk_forward(data, model, min_train_size=5)
    shifted = data.assign(target=data.target + 100)
    walk_forward(shifted, model, min_train_size=5)

    pd.testing.assert_frame_equal(walk_forward(data, model, min_train_size=5), first)


@pytest.mark.parametrize(
    "corruption", ["missing_column", "origin_not_before_target", "duplicate", "missing_date", "inf"]
)
def test_invalid_data_is_rejected(corruption):
    data = prepared_variance(feature_columns=("log_rv_daily",))
    if corruption == "missing_column":
        data = data.drop(columns="target")
    elif corruption == "origin_not_before_target":
        data.loc[0, "origin_date"] = data.loc[0, "target_date"]
    elif corruption == "duplicate":
        data = pd.concat([data, data.iloc[:1]], ignore_index=True)
    elif corruption == "missing_date":
        data.loc[0, "target_date"] = pd.NaT
    else:
        data.loc[0, "log_rv_daily"] = np.inf
    with pytest.raises(ValueError):
        ar1 = ForecastModel("AR1", LinearRegression(), ("log_rv_daily",))
        walk_forward(data, ar1, min_train_size=10)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_min_train_size_must_be_a_positive_integer(size):
    with pytest.raises(ValueError, match="min_train_size"):
        mean = ForecastModel("historical_mean", LinearRegression())
        walk_forward(prepared_variance(), mean, min_train_size=size)


def test_non_finite_predictions_are_rejected():
    class Broken:
        def fit(self, features, target):
            return self

        def predict(self, features):
            return np.array([np.nan])

    with pytest.raises(ValueError, match="broken"):
        walk_forward(prepared_variance(), ForecastModel("broken", Broken()), min_train_size=15)
