import numpy as np
import pandas as pd

from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.models import ForecastModel, LinearRegression
from rvol.walkforward import walk_forward
from tests.forecasting_helpers import daily_sample, har_features


def test_next_session_labels_become_available_only_at_their_target_date():
    daily = daily_sample()
    prepared = prepare_forecast_data(
        har_features(("log_rv_daily",))(daily),
        build_next_observation_targets(daily, value_col="log_rv"),
    )
    ar1 = ForecastModel("AR1", LinearRegression(), ("log_rv_daily",))
    result = walk_forward(prepared, ar1, min_train_size=10)
    for row in result.itertuples():
        available = prepared.loc[prepared.target_date <= row.origin_date]
        assert row.n_train == len(available)
        index = daily.index[daily.date == row.origin_date][0]
        assert row.target_date == daily.date.iloc[index + 1]
        assert row.actual == daily.log_rv.iloc[index + 1]


def test_next_session_targets_preserve_log_values_without_transformations():
    daily = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3), "log_rv": [-700.0, 700.0, 0.0]}
    )
    with np.errstate(over="raise", invalid="raise"):
        targets = build_next_observation_targets(daily, value_col="log_rv")
    np.testing.assert_array_equal(targets.target.iloc[:2], [700.0, 0.0])
    assert pd.isna(targets.target.iloc[-1])
