import numpy as np
import pandas as pd
import pytest

from rvol.backtesting import WalkForwardPlanner, WalkForwardRunner
from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.data.volatility.features import VarianceFeatureBuilder
from rvol.models import AR1Model
from tests.forecasting_helpers import daily_sample


def test_next_session_labels_become_available_only_at_their_target_date():
    daily = daily_sample()
    prepared = prepare_forecast_data(
        VarianceFeatureBuilder(feature_names=("log_rv_daily",))(daily),
        build_next_observation_targets(daily, value_col="log_rv"),
    )
    result = WalkForwardRunner(WalkForwardPlanner(min_train_size=10)).run(
        prepared, model=AR1Model(), feature_columns=("log_rv_daily",)
    )
    for row in result.itertuples():
        available = prepared.loc[prepared.target_date <= row.origin_date]
        assert row.n_train == len(available)
        assert row.train_end_date == available.target_date.max()
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


@pytest.mark.parametrize("corruption", ["origin", "duplicate", "target", "numeric"])
def test_execution_rejects_invalid_prepared_tables(corruption):
    daily = daily_sample()
    prepared = prepare_forecast_data(
        VarianceFeatureBuilder(feature_names=("log_rv_daily",))(daily),
        build_next_observation_targets(daily, value_col="log_rv"),
    )
    if corruption == "origin":
        prepared.loc[0, "origin_date"] = prepared.loc[0, "target_date"]
    elif corruption == "duplicate":
        prepared = pd.concat([prepared, prepared.iloc[:1]], ignore_index=True)
    elif corruption == "target":
        prepared.loc[0, "target_date"] = pd.NaT
    else:
        prepared.loc[0, "log_rv_daily"] = np.inf
    with pytest.raises(ValueError):
        WalkForwardRunner(WalkForwardPlanner(min_train_size=10)).run(
            prepared, model=AR1Model(), feature_columns=("log_rv_daily",)
        )
