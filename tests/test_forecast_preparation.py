from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rvol.domain import (
    ComponentSpecification,
    ExperimentConfig,
    TargetBuilder,
    TargetHorizon,
    WalkForwardSettings,
)
from rvol.forecasting.composition import standard_experiment_definition
from rvol.forecasting.execution import prepare_forecast_data
from rvol.forecasting.features import HARFeatureBuilder
from rvol.forecasting.models import HistoricalMeanForecaster
from rvol.forecasting.targets import RealizedVarianceTargetBuilder


def daily_sample(n: int = 16) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n + 1).delete(min(6, n))
    return pd.DataFrame({"date": dates, "log_rv": np.log(np.arange(1, n + 1))})


@pytest.mark.parametrize("kind", ["session", "cumulative"])
@pytest.mark.parametrize("sessions", [1, 3, 5, 30])
def test_target_builder_labels_the_full_timeline_without_feature_warmup(kind, sessions):
    daily = daily_sample().sample(frac=1, random_state=42)
    unchanged = daily.copy(deep=True)
    builder = RealizedVarianceTargetBuilder(TargetHorizon(sessions, kind))
    targets = builder(daily)
    ordered = daily.sort_values("date").reset_index(drop=True)

    assert isinstance(builder, TargetBuilder)
    assert len(targets) == len(daily)
    assert list(targets.columns) == [
        "origin_date", "target_date", "target", "target_start_date",
        "horizon_sessions", "target_kind",
    ]
    assert targets.attrs["target_horizon"] == {"sessions": sessions, "kind": kind}
    assert targets["target"].notna().sum() == max(0, len(daily) - sessions)
    for index, row in enumerate(targets.itertuples()):
        assert row.origin_date == ordered["date"].iloc[index]
        if index + sessions >= len(daily):
            assert pd.isna(row.target_date) and pd.isna(row.target)
            continue
        assert row.target_date == ordered["date"].iloc[index + sessions]
        start = index + (1 if kind == "cumulative" else sessions)
        assert row.target_start_date == ordered["date"].iloc[start]
        variance = index + sessions + 1
        if kind == "cumulative":
            variance = sum(range(index + 2, index + sessions + 2))
        assert row.target == pytest.approx(np.log(variance))
    pd.testing.assert_frame_equal(daily, unchanged)


@pytest.mark.parametrize("kind", ["session", "cumulative"])
def test_sparse_reordered_features_keep_targets_on_the_full_session_timeline(kind):
    daily = daily_sample()
    horizon = TargetHorizon(3, kind)
    features = HARFeatureBuilder(weekly_window=2, monthly_window=4)(daily)
    # Interior omissions must not move labels three retained feature rows ahead.
    features = features.iloc[[0, 2, 5, 7, 12]].iloc[::-1]
    targets = RealizedVarianceTargetBuilder(horizon)(daily).iloc[::-1]
    original_features, original_targets = features.copy(deep=True), targets.copy(deep=True)
    prepared = prepare_forecast_data(features, targets)

    origins = [3, 5, 8, 10]
    pd.testing.assert_series_equal(
        prepared["origin_date"], daily.loc[origins, "date"].reset_index(drop=True),
        check_names=False,
    )
    for index, row in zip(origins, prepared.itertuples(), strict=True):
        assert row.target_date == daily.loc[index + 3, "date"]
        expected = index + 4 if kind == "session" else sum(range(index + 2, index + 5))
        assert row.target == pytest.approx(np.log(expected))
    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_frame_equal(targets, original_targets)


def test_definition_can_inject_an_independent_target_builder_and_feature_family():
    horizon = TargetHorizon(2)

    class SparseFeatures:
        specification = ComponentSpecification(
            "sparse", "test.SparseFeatures", feature_names=("input",),
        )

        def __call__(self, daily):
            return pd.DataFrame({"origin_date": daily["date"], "input": daily["log_rv"]}).iloc[::2]

    class OtherTargets:
        horizon = TargetHorizon(2)
        specification = ComponentSpecification("other", "test.OtherTargets")

        def __call__(self, daily):
            # This independent implementation does not use HAR or its feature columns.
            targets = pd.DataFrame({
                "origin_date": daily["date"],
                "target_date": daily["date"].shift(-2),
                "target_start_date": daily["date"].shift(-2),
                "target": daily["other_outcome"].shift(-2),
                "horizon_sessions": 2,
                "target_kind": "session",
            })
            targets.attrs["target_horizon"] = {"sessions": 2, "kind": "session"}
            return targets

    definition = replace(
        standard_experiment_definition(target_horizon=horizon),
        models=(HistoricalMeanForecaster(),), feature_builder=SparseFeatures(),
        target_builder=OtherTargets(), comparison_pairs=(),
        candidate_model="historical_mean", baseline_models=(),
    )
    daily = daily_sample()
    daily["other_outcome"] = np.arange(len(daily)) / 10 - 5
    prepared = definition.prepare(daily)
    result = definition.run(daily, ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=2),
    ))

    assert definition.specification.features.name == "sparse"
    assert definition.specification.target.name == "other"
    assert isinstance(definition.target_builder, TargetBuilder)
    assert result.target_horizon == horizon
    assert result.records
    for record in result.records:
        available = prepared[prepared["target_date"] <= pd.Timestamp(record.origin_date)]
        assert record.n_train == len(available)
        assert record.predicted_log_rv == pytest.approx(available["target"].mean())
        index = daily.index[daily["date"] == pd.Timestamp(record.origin_date)][0]
        assert record.target_date == daily.loc[index + 2, "date"].date()
        assert record.actual_log_rv == daily.loc[index + 2, "other_outcome"]


@pytest.mark.parametrize("corruption, message", [
    ("feature_labels", "must not supply target"),
    ("feature_duplicate", "one valid row per origin"),
    ("target_duplicate", "one valid row per origin"),
    ("unknown_origin", "must exist in the target timeline"),
    ("missing_origin", "missing required columns"),
    ("missing_target", "missing required columns"),
    ("extra_target_column", "only origin and target"),
    ("partial_label", "available together"),
    ("partial_date", "available together"),
    ("invalid_interval", "origin must precede"),
    ("invalid_start", "target start must follow"),
    ("infinite_label", "non-finite"),
    ("mixed_horizon", "one consistent target horizon"),
    ("duplicate_column", "columns must be unique"),
])
def test_preparation_rejects_ambiguous_alignment_and_target_metadata(corruption, message):
    daily = daily_sample()
    features = HARFeatureBuilder(weekly_window=2, monthly_window=4)(daily)
    targets = RealizedVarianceTargetBuilder(TargetHorizon(3))(daily)
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
    elif corruption == "invalid_interval":
        targets.loc[0, "target_date"] = targets.loc[0, "origin_date"]
    elif corruption == "invalid_start":
        targets.loc[0, "target_start_date"] = targets.loc[0, "origin_date"]
    elif corruption == "infinite_label":
        targets.loc[0, "target"] = np.inf
    elif corruption == "mixed_horizon":
        targets.loc[0, "horizon_sessions"] = 5
    elif corruption == "duplicate_column":
        features = pd.concat([features, features[["rv_daily"]]], axis=1)
    with pytest.raises(ValueError, match=message):
        prepare_forecast_data(features, targets)


@pytest.mark.parametrize("cause", ["no_history", "no_future", "no_data"])
def test_empty_prepared_data_retains_its_target_definition(cause):
    daily = daily_sample(3 if cause == "no_history" else 16)
    if cause == "no_data":
        daily = daily.iloc[:0]
    horizon = TargetHorizon(30 if cause == "no_future" else 3, "cumulative")
    features = HARFeatureBuilder(
        weekly_window=2, monthly_window=22 if cause == "no_history" else 4,
    )(daily)
    targets = RealizedVarianceTargetBuilder(horizon)(daily)
    result = prepare_forecast_data(features, targets)
    assert result.empty
    assert result.attrs["target_horizon"] == {"sessions": horizon.sessions, "kind": horizon.kind}


def test_preparation_rejects_a_builder_horizon_that_disagrees_with_its_output():
    builder = RealizedVarianceTargetBuilder(TargetHorizon(3))
    daily = daily_sample()
    with pytest.raises(ValueError, match="must match the configured target builder"):
        prepare_forecast_data(
            HARFeatureBuilder(weekly_window=2, monthly_window=4)(daily), builder(daily),
            expected_horizon=TargetHorizon(5),
        )
