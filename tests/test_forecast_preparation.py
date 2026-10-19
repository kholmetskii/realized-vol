import numpy as np
import pandas as pd
import pytest

from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.data.volatility.features import VarianceFeatureBuilder
from rvol.models import HistoricalMeanModel
from rvol.walkforward import WalkForwardPlanner, WalkForwardRunner


def daily_sample(n: int = 16) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n + 1).delete(min(6, n))
    return pd.DataFrame({"date": dates, "log_rv": np.log(np.arange(1, n + 1))})


def test_targets_label_the_full_timeline_without_feature_warmup():
    daily = daily_sample().sample(frac=1, random_state=42)
    unchanged = daily.copy(deep=True)
    targets = build_next_observation_targets(daily, value_col="log_rv")
    ordered = daily.sort_values("date").reset_index(drop=True)

    assert len(targets) == len(daily)
    assert list(targets.columns) == ["origin_date", "target_date", "target"]
    assert targets["target"].notna().sum() == len(daily) - 1
    for index, row in enumerate(targets.itertuples()):
        assert row.origin_date == ordered["date"].iloc[index]
        if index == len(daily) - 1:
            assert pd.isna(row.target_date) and pd.isna(row.target)
            continue
        assert row.target_date == ordered["date"].iloc[index + 1]
        assert row.target == ordered["log_rv"].iloc[index + 1]
    pd.testing.assert_frame_equal(daily, unchanged)


def test_sparse_reordered_features_keep_targets_on_the_full_session_timeline():
    daily = daily_sample()
    features = VarianceFeatureBuilder(weekly_window=2, monthly_window=4)(daily)
    # Interior omissions must not move labels to the next retained feature row.
    features = features.iloc[[0, 2, 5, 7, 12]].iloc[::-1]
    targets = build_next_observation_targets(daily, value_col="log_rv").iloc[::-1]
    original_features, original_targets = features.copy(deep=True), targets.copy(deep=True)
    prepared = prepare_forecast_data(features, targets)

    origins = [3, 5, 8, 10]
    pd.testing.assert_series_equal(
        prepared["origin_date"],
        daily.loc[origins, "date"].reset_index(drop=True),
        check_names=False,
    )
    for index, row in zip(origins, prepared.itertuples(), strict=True):
        assert row.target_date == daily.loc[index + 1, "date"]
        assert row.target == daily.loc[index + 1, "log_rv"]
    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_frame_equal(targets, original_targets)


def test_independent_plain_builders_need_no_specification_or_shared_base_class():
    daily = daily_sample()
    daily["other_outcome"] = np.arange(len(daily)) / 10 - 5
    features = pd.DataFrame({"origin_date": daily.date, "input": daily.log_rv}).iloc[::2]
    targets = pd.DataFrame(
        {
            "origin_date": daily.date,
            "target_date": daily.date.shift(-1),
            "target": daily.other_outcome.shift(-1),
        }
    )
    prepared = prepare_forecast_data(features, targets)
    forecasts = WalkForwardRunner(WalkForwardPlanner(min_train_size=2)).run(
        prepared, model=HistoricalMeanModel()
    )
    assert not forecasts.empty
    for row in forecasts.itertuples():
        available = prepared.loc[prepared.target_date <= row.origin_date]
        assert row.n_train == len(available)
        assert row.predicted == pytest.approx(available.target.mean())
        index = daily.index[daily.date == row.origin_date][0]
        assert row.target_date == daily.loc[index + 1, "date"]
        assert row.actual == daily.loc[index + 1, "other_outcome"]


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("feature_labels", "must not supply target"),
        ("feature_duplicate", "one valid row per origin"),
        ("target_duplicate", "one valid row per origin"),
        ("unknown_origin", "must exist in the target timeline"),
        ("missing_origin", "missing required columns"),
        ("missing_target", "missing required columns"),
        ("extra_target_column", "only origin and target"),
        ("partial_label", "available together"),
        ("partial_date", "available together"),
        ("invalid_dates", "origin must precede"),
        ("infinite_label", "non-finite"),
        ("later_observation", "next observation"),
        ("duplicate_column", "columns must be unique"),
    ],
)
def test_preparation_rejects_ambiguous_alignment_and_invalid_targets(corruption, message):
    daily = daily_sample()
    features = VarianceFeatureBuilder(weekly_window=2, monthly_window=4)(daily)
    targets = build_next_observation_targets(daily, value_col="log_rv")
    if corruption == "feature_labels":
        features["target"] = 0.0
    elif corruption == "feature_duplicate":
        features = pd.concat([features, features.iloc[:1]])
    elif corruption == "target_duplicate":
        targets = pd.concat([targets, targets.iloc[:1]])
    elif corruption == "unknown_origin":
        features.loc[0, "origin_date"] = pd.Timestamp("2020-01-01")
    elif corruption == "missing_origin":
        features = features.drop(columns="origin_date")
    elif corruption == "missing_target":
        targets = targets.drop(columns="target")
    elif corruption == "extra_target_column":
        targets["feature"] = 0.0
    elif corruption == "partial_label":
        targets.loc[0, "target"] = np.nan
    elif corruption == "partial_date":
        targets.loc[0, "target_date"] = pd.NaT
    elif corruption == "invalid_dates":
        targets.loc[0, "target_date"] = targets.loc[0, "origin_date"]
    elif corruption == "infinite_label":
        targets.loc[0, "target"] = np.inf
    elif corruption == "later_observation":
        targets["target_date"] = daily.date.shift(-2)
        targets["target"] = daily.log_rv.shift(-2)
    elif corruption == "duplicate_column":
        features = pd.concat([features, features[["log_rv_daily"]]], axis=1)
    with pytest.raises(ValueError, match=message):
        prepare_forecast_data(features, targets)


@pytest.mark.parametrize("cause", ["no_history", "no_future", "no_data"])
def test_empty_prepared_data_keeps_the_same_columns(cause):
    daily = daily_sample(1 if cause == "no_future" else 3)
    if cause == "no_data":
        daily = daily.iloc[:0]
    features = VarianceFeatureBuilder(
        feature_names=("log_rv_daily",)
        if cause == "no_future"
        else ("log_rv_daily", "log_rv_weekly", "log_rv_monthly"),
    )(daily)
    targets = build_next_observation_targets(daily, value_col="log_rv")
    result = prepare_forecast_data(features, targets)
    assert result.empty
    assert list(result.columns) == ["origin_date", "target_date", *features.columns[1:], "target"]


def test_preparation_requires_the_target_timeline_before_feature_warmup():
    daily = daily_sample()
    features = VarianceFeatureBuilder(weekly_window=2, monthly_window=4)(daily)
    targets = build_next_observation_targets(daily, value_col="log_rv").drop(index=1)
    with pytest.raises(ValueError, match="next observation"):
        prepare_forecast_data(features, targets)


def test_observation_preparation_rejects_missing_nullable_values():
    observations = pd.DataFrame(
        {
            "date": pd.date_range("2026-10-01", periods=3),
            "value": pd.array([1.0, None, 3.0], dtype="Float64"),
        }
    )

    with pytest.raises(ValueError, match="only finite values"):
        build_next_observation_targets(observations)


@pytest.mark.parametrize(
    "values",
    [
        [np.nan, 2.0, 3.0],
        [np.inf, 2.0, 3.0],
        [-np.inf, 2.0, 3.0],
        pd.array([None, 2.0, 3.0], dtype="Float64"),
        ["invalid", "2", "3"],
    ],
)
def test_preparation_rejects_invalid_features_before_a_dataset_can_be_saved(values):
    observations = pd.DataFrame(
        {"date": pd.date_range("2026-10-01", periods=3), "value": [1.0, 2.0, 3.0]}
    )
    features = pd.DataFrame({"origin_date": observations.date, "input": values})
    targets = build_next_observation_targets(observations)

    with pytest.raises(ValueError, match="finite numerical values"):
        prepare_forecast_data(features, targets)


def test_preparation_accepts_valid_nullable_and_numeric_text_features_without_mutating_inputs():
    observations = pd.DataFrame(
        {"date": pd.date_range("2026-10-01", periods=3), "value": [-5.0, 0.0, 9.0]}
    )
    features = pd.DataFrame(
        {
            "origin_date": observations.date,
            "nullable": pd.array([-2.0, 0.0, 3.0], dtype="Float64"),
            "text": ["-10", "0", "30"],
        }
    )
    targets = build_next_observation_targets(observations)
    original = features.copy(deep=True)

    prepared = prepare_forecast_data(features, targets)

    np.testing.assert_array_equal(prepared.nullable.to_numpy(dtype="float64"), [-2.0, 0.0])
    np.testing.assert_array_equal(prepared.text, [-10, 0])
    np.testing.assert_array_equal(prepared.target, [0.0, 9.0])
    pd.testing.assert_frame_equal(features, original)
