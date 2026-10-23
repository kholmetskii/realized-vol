import pathlib
from typing import cast

import numpy as np
import pandas as pd
import pytest

from rvol.data.volatility import build_daily_dataset

STEP = 1e-6  # log-price increase per minute


def two_sessions() -> pd.DataFrame:
    """One price a minute through two full sessions, 2024-01-16 and 2024-01-17."""
    timestamps = pd.date_range("2024-01-15 22:00", "2024-01-17 21:59", freq="1min", tz="UTC")
    log_prices = np.log(1.10) + np.arange(len(timestamps)) * STEP
    return pd.DataFrame({"ts": timestamps, "mid": np.exp(log_prices)})


def write(ticks: pd.DataFrame, path: pathlib.Path) -> pathlib.Path:
    ticks.to_parquet(path, index=False)
    return path


def test_sessions_split_across_files_give_the_expected_rv(tmp_path):
    ticks = two_sessions()
    split = pd.Timestamp("2024-01-16 00:00", tz="UTC")
    first = write(ticks[ticks["ts"] < split], tmp_path / "ticks_1.parquet")
    second = write(ticks[ticks["ts"] >= split], tmp_path / "ticks_2.parquet")

    result = build_daily_dataset([first, second], frequency="5min")

    # 288 five-minute prices, 5 minutes apart in log price: 287 returns of 5 * STEP.
    assert list(result.columns) == ["date", "rv", "log_rv", "observation_count"]
    assert list(result["date"]) == [pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-17")]
    assert list(result["observation_count"]) == [288, 288]
    np.testing.assert_allclose(result["rv"], 287 * (5 * STEP) ** 2)
    assert np.isclose(cast(float, result.loc[0, "log_rv"]), np.log(287 * (5 * STEP) ** 2))


def test_overlapping_chunks_are_deduplicated(tmp_path):
    ticks = two_sessions()
    first = write(
        ticks[ticks["ts"] < pd.Timestamp("2024-01-17 00:00", tz="UTC")], tmp_path / "a.parquet"
    )
    second = write(
        ticks[ticks["ts"] >= pd.Timestamp("2024-01-16 00:00", tz="UTC")], tmp_path / "b.parquet"
    )
    complete = write(ticks, tmp_path / "complete.parquet")

    pd.testing.assert_frame_equal(
        build_daily_dataset([first, second]), build_daily_dataset([complete])
    )


def test_files_whose_names_are_not_in_time_order_are_rejected(tmp_path):
    ticks = two_sessions()
    split = pd.Timestamp("2024-01-16 12:00", tz="UTC")
    early = write(ticks[ticks["ts"] < split], tmp_path / "b_early.parquet")
    late = write(ticks[ticks["ts"] >= split], tmp_path / "a_late.parquet")

    with pytest.raises(ValueError, match="time order"):
        build_daily_dataset([early, late])


@pytest.mark.parametrize(
    "cut, kept",
    [
        ("data ends at 10:00 New York", pd.Timestamp("2024-01-16")),
        ("data starts at 22:00 New York", pd.Timestamp("2024-01-17")),
    ],
)
def test_sessions_cut_at_the_edges_of_the_data_are_dropped(tmp_path, cut, kept):
    ticks = two_sessions()
    if cut.startswith("data ends"):
        ticks = ticks[ticks["ts"] < pd.Timestamp("2024-01-17 15:00", tz="UTC")]
    else:
        ticks = ticks[ticks["ts"] >= pd.Timestamp("2024-01-16 03:00", tz="UTC")]

    result = build_daily_dataset([write(ticks, tmp_path / "ticks.parquet")])

    assert list(result["date"]) == [kept]
