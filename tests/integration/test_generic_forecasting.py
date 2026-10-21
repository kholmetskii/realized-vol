import numpy as np
import pandas as pd
import pytest

from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.evaluation import forecast_losses
from rvol.models import ExponentialSmoothing, ForecastModel, LinearRegression
from rvol.walkforward import walk_forward
from tests.forecasting_helpers import benchmark_models, lag_features, prepared_variance
from tests.losses import absolute_error, squared_error


@pytest.mark.parametrize("subject, shift", [("apples", 0), ("price", 100), ("temperature", -20)])
def test_the_same_model_and_engine_work_for_arbitrary_subjects(subject, shift):
    data = pd.DataFrame(
        {"observed_at": pd.date_range("2024-01-01", periods=40), subject: np.arange(40) + shift}
    )
    prepared = prepare_forecast_data(
        lag_features(data, date_col="observed_at", value_col=subject),
        build_next_observation_targets(data, date_col="observed_at", value_col=subject),
    )

    ar1 = ForecastModel("AR1", LinearRegression(), ("lag_0",))
    forecasts = walk_forward(prepared, ar1, min_train_size=5)

    np.testing.assert_allclose(forecasts.predicted, forecasts.actual, atol=1e-10)
    assert forecast_losses(forecasts, squared_error).loss.mean() < 1e-15


def test_negative_zero_and_large_values_are_preserved():
    data = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=45), "value": (np.arange(45) - 20) * 1000.0}
    )
    prepared = prepare_forecast_data(lag_features(data), build_next_observation_targets(data))

    ar1 = ForecastModel("AR1", LinearRegression(), ("lag_0",))
    forecasts = walk_forward(prepared, ar1, min_train_size=5)

    assert forecasts.actual.min() < 0 < forecasts.actual.max()
    assert 0 in forecasts.actual.values
    np.testing.assert_allclose(forecasts.actual, forecasts.predicted, atol=1e-8)


def test_naive_forecasts_and_losses_apply_no_implicit_transformations():
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=12), "value": np.arange(12)})
    prepared = prepare_forecast_data(lag_features(data), build_next_observation_targets(data))

    naive = ForecastModel("naive", ExponentialSmoothing(decay=0.0))
    forecasts = walk_forward(prepared, naive, min_train_size=3)

    np.testing.assert_array_equal(forecasts.predicted, prepared.lag_0.loc[forecasts.index + 3])

    for loss in (squared_error, absolute_error):
        np.testing.assert_allclose(forecast_losses(forecasts, loss).loss, 1)


def test_each_model_waits_only_for_the_inputs_it_uses():
    for model in benchmark_models():
        data = prepared_variance(n=50, feature_columns=model.regressors)

        forecasts = walk_forward(data, model, min_train_size=5)

        first_target = pd.bdate_range("2023-01-02", periods=50)[27 if model.name == "HAR" else 6]
        assert forecasts.target_date.iloc[0] == first_target
