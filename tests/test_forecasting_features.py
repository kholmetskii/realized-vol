import numpy as np
import pandas as pd
import pytest

from rvol.features.forecasting import build_har_features


def daily_sample(n_sessions: int = 35) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=n_sessions),
        "log_rv": np.linspace(-12.0, -9.0, n_sessions),
    })


def test_har_features_have_exact_one_session_ahead_alignment():
    daily = daily_sample()

    features = build_har_features(daily)

    first_origin = 21
    assert features.loc[0, "origin_date"] == daily.loc[first_origin, "date"]
    assert features.loc[0, "target_date"] == daily.loc[first_origin + 1, "date"]
    assert features.loc[0, "rv_daily"] == daily.loc[first_origin, "log_rv"]
    assert features.loc[0, "target"] == daily.loc[first_origin + 1, "log_rv"]
    assert len(features) == len(daily) - 22


def test_har_rolling_features_use_only_information_at_the_origin():
    daily = daily_sample()
    origin = 24

    features = build_har_features(daily)
    row = features[features["origin_date"] == daily.loc[origin, "date"]].iloc[0]

    assert np.isclose(row["rv_weekly"], daily.loc[origin - 4 : origin, "log_rv"].mean())
    assert np.isclose(row["rv_monthly"], daily.loc[origin - 21 : origin, "log_rv"].mean())


def test_changing_a_future_value_cannot_change_earlier_features():
    original = daily_sample()
    changed = original.copy()
    changed_at = 30
    changed.loc[changed_at, "log_rv"] = 100.0

    before = build_har_features(original)
    after = build_har_features(changed)
    earlier = before["origin_date"] < original.loc[changed_at, "date"]
    columns = ["origin_date", "target_date", "rv_daily", "rv_weekly", "rv_monthly"]

    pd.testing.assert_frame_equal(
        before.loc[earlier, columns].reset_index(drop=True),
        after.loc[earlier, columns].reset_index(drop=True),
    )


def test_rolling_windows_count_sessions_not_calendar_days():
    daily = daily_sample()
    daily = daily.drop(index=10).reset_index(drop=True)

    features = build_har_features(daily)
    first = features.iloc[0]

    assert first["origin_date"] == daily.loc[21, "date"]
    assert np.isclose(first["rv_monthly"], daily.loc[:21, "log_rv"].mean())


def test_duplicate_session_dates_are_rejected():
    daily = daily_sample()
    daily.loc[1, "date"] = daily.loc[0, "date"]

    with pytest.raises(ValueError, match="one row per session"):
        build_har_features(daily)


def test_large_gap_is_rejected_before_rolling_features_are_built():
    daily = daily_sample()
    daily.loc[20:, "date"] += pd.Timedelta(days=30)

    with pytest.raises(ValueError, match=r"\d+-day gap"):
        build_har_features(daily)
