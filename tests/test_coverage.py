import numpy as np
import pandas as pd

from rvol.features.coverage import audit_daily_dataset


def daily_sample(n_sessions: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=n_sessions),
        "rv_5min": np.linspace(1e-5, 2e-5, n_sessions),
        "observation_count": np.full(n_sessions, 288),
    })


def test_complete_daily_data_passes_and_reports_year_coverage():
    audit = audit_daily_dataset(daily_sample())

    assert audit.passed
    assert audit.n_sessions == 30
    assert audit.start_date == pd.Timestamp("2024-01-02")
    assert audit.end_date == pd.Timestamp("2024-02-12")
    assert audit.median_observation_count == 288
    assert [(item.year, item.sessions) for item in audit.sessions_by_year] == [(2024, 30)]


def test_large_disconnected_period_is_reported():
    first = daily_sample(10)
    second = daily_sample(10)
    second["date"] = second["date"] + pd.DateOffset(years=1)
    daily = pd.concat([first, second], ignore_index=True)

    audit = audit_daily_dataset(daily)

    assert not audit.passed
    assert len(audit.large_gaps) == 1
    assert audit.large_gaps[0].calendar_days > 300
    assert [(item.year, item.sessions) for item in audit.sessions_by_year] == [
        (2024, 10),
        (2025, 10),
    ]


def test_duplicates_ordering_and_invalid_values_fail_the_audit():
    daily = daily_sample()
    daily.loc[5, "date"] = daily.loc[4, "date"]
    daily.loc[[10, 11], "date"] = daily.loc[[11, 10], "date"].to_numpy()
    daily.loc[15, "rv_5min"] = 0.0
    daily.loc[20, "observation_count"] = 100

    audit = audit_daily_dataset(daily)

    assert not audit.passed
    assert audit.duplicate_dates == 1
    assert audit.dates_out_of_order
    assert audit.invalid_rv == 1
    assert audit.unusual_observation_sessions == 1


def test_daylight_saving_length_sessions_are_within_default_tolerance():
    daily = daily_sample(3)
    daily["observation_count"] = [276, 288, 300]

    audit = audit_daily_dataset(daily)

    assert audit.passed
    assert audit.unusual_observation_sessions == 0
