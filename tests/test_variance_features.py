import numpy as np
import pandas as pd
import pytest

from rvol.data.volatility.features import VarianceFeatureBuilder


def daily_sample(n_sessions: int = 35) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=n_sessions),
            "log_rv": np.linspace(-12.0, -9.0, n_sessions),
        }
    )


def test_har_features_include_current_inputs_and_the_latest_observed_origin():
    daily = daily_sample()

    features = VarianceFeatureBuilder()(daily)

    first_origin = 21
    assert features.loc[0, "origin_date"] == daily.loc[first_origin, "date"]
    assert features.loc[0, "log_rv_daily"] == daily.loc[first_origin, "log_rv"]
    assert len(features) == len(daily) - 21
    assert features.iloc[-1]["origin_date"] == daily.iloc[-1]["date"]
    assert list(features.columns) == [
        "origin_date",
        "log_rv_daily",
        "log_rv_weekly",
        "log_rv_monthly",
    ]


def test_har_rolling_features_use_only_information_at_the_origin():
    daily = daily_sample()
    origin = 24

    features = VarianceFeatureBuilder()(daily)
    row = features[features["origin_date"] == daily.loc[origin, "date"]].iloc[0]

    assert np.isclose(row["log_rv_weekly"], daily.loc[origin - 4 : origin, "log_rv"].mean())
    assert np.isclose(row["log_rv_monthly"], daily.loc[origin - 21 : origin, "log_rv"].mean())


def test_changing_a_future_value_cannot_change_earlier_features():
    original = daily_sample()
    changed = original.copy()
    changed_at = 30
    changed.loc[changed_at, "log_rv"] = 100.0

    before = VarianceFeatureBuilder()(original)
    after = VarianceFeatureBuilder()(changed)
    earlier = before["origin_date"] < original.loc[changed_at, "date"]
    columns = ["origin_date", "log_rv_daily", "log_rv_weekly", "log_rv_monthly"]

    pd.testing.assert_frame_equal(
        before.loc[earlier, columns].reset_index(drop=True),
        after.loc[earlier, columns].reset_index(drop=True),
    )


def test_rolling_windows_count_sessions_not_calendar_days():
    daily = daily_sample()
    daily = daily.drop(index=10).reset_index(drop=True)

    features = VarianceFeatureBuilder()(daily)
    first = features.iloc[0]

    assert first["origin_date"] == daily.loc[21, "date"]
    assert np.isclose(first["log_rv_monthly"], daily.loc[:21, "log_rv"].mean())


def test_duplicate_session_dates_are_rejected():
    daily = daily_sample()
    daily.loc[1, "date"] = daily.loc[0, "date"]

    with pytest.raises(ValueError, match="one row per session"):
        VarianceFeatureBuilder()(daily)


def test_large_gap_is_rejected_before_rolling_features_are_built():
    daily = daily_sample()
    daily.loc[20:, "date"] += pd.Timedelta(days=30)

    with pytest.raises(ValueError, match=r"\d+-day gap"):
        VarianceFeatureBuilder()(daily)


def test_configured_builder_uses_the_windows_and_columns_it_describes():
    daily = daily_sample(8).rename(columns={"date": "session", "log_rv": "value"})
    builder = VarianceFeatureBuilder(
        date_col="session",
        rv_col="value",
        weekly_window=2,
        monthly_window=4,
        max_gap_days=None,
    )

    features = builder(daily)
    settings = {
        "date_col": builder.date_col,
        "rv_col": builder.rv_col,
        "weekly_window": builder.weekly_window,
        "monthly_window": builder.monthly_window,
        "max_gap_days": builder.max_gap_days,
    }

    assert settings == {
        "date_col": "session",
        "rv_col": "value",
        "weekly_window": 2,
        "monthly_window": 4,
        "max_gap_days": None,
    }
    assert len(features) == 5
    assert features.loc[0, "origin_date"] == daily.loc[3, "session"]
    assert features.loc[0, "log_rv_weekly"] == pytest.approx(daily.loc[2:3, "value"].mean())
    assert features.loc[0, "log_rv_monthly"] == pytest.approx(daily.loc[:3, "value"].mean())


@pytest.mark.parametrize("names", [(), ("log_rv_daily",)])
def test_models_without_rolling_features_keep_short_histories(names):
    daily = daily_sample(4)

    features = VarianceFeatureBuilder(feature_names=names)(daily)

    assert list(features.columns) == ["origin_date", *names]
    assert features["origin_date"].equals(daily["date"].rename("origin_date"))
    if names:
        np.testing.assert_array_equal(features["log_rv_daily"], daily["log_rv"])


def test_only_selected_rolling_features_determine_the_warmup():
    daily = daily_sample(8)

    features = VarianceFeatureBuilder(feature_names=("log_rv_weekly",))(daily)

    assert list(features.columns) == ["origin_date", "log_rv_weekly"]
    assert features["origin_date"].tolist() == daily.loc[4:, "date"].tolist()
    assert features.loc[0, "log_rv_weekly"] == pytest.approx(daily.loc[:4, "log_rv"].mean())


@pytest.mark.parametrize(
    "names, message",
    [
        (("missing",), "unknown variance features"),
        (("log_rv_daily", "log_rv_daily"), "unique"),
    ],
)
def test_invalid_feature_selections_are_rejected(names, message):
    with pytest.raises(ValueError, match=message):
        VarianceFeatureBuilder(feature_names=names)


def test_edited_builder_settings_are_applied_and_validated_before_preparation():
    daily = daily_sample(8)
    builder = VarianceFeatureBuilder()
    builder.weekly_window = 2
    builder.monthly_window = 4

    expected = VarianceFeatureBuilder(weekly_window=2, monthly_window=4)(daily)
    pd.testing.assert_frame_equal(builder(daily), expected)

    builder.weekly_window = 5
    with pytest.raises(ValueError, match="monthly_window must be at least weekly_window"):
        builder(daily)


@pytest.mark.parametrize("column", ["date", "log_rv"])
def test_daily_feature_preparation_rejects_ambiguous_duplicate_columns(column):
    daily = daily_sample()
    ambiguous = pd.concat([daily, daily[[column]]], axis=1)

    with pytest.raises(ValueError, match="columns must be unique"):
        VarianceFeatureBuilder()(ambiguous)


def test_daily_feature_preparation_rejects_missing_nullable_log_variance():
    daily = daily_sample()
    daily["log_rv"] = daily["log_rv"].astype("Float64")
    daily.loc[0, "log_rv"] = pd.NA

    with pytest.raises(ValueError, match="invalid dates or log_rv values"):
        VarianceFeatureBuilder()(daily)
