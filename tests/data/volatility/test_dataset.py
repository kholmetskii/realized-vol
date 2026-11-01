import os
import pathlib
from typing import cast

import numpy as np
import pandas as pd
import pytest

from rvol.data.volatility import COLUMNS, build_daily_dataset, load_daily_dataset

STEP = 1e-6  # log-price increase per minute


def two_sessions(*, zigzag: bool = True) -> pd.DataFrame:
    """One price a minute through two full sessions, 2024-01-16 and 2024-01-17.

    The log price moves STEP a minute, up through one five-minute bin and down
    through the next, so every five-minute return is +-5 * STEP and the signs
    alternate; without the zigzag it only rises.
    """
    timestamps = pd.date_range("2024-01-15 22:00", "2024-01-17 21:59", freq="1min", tz="UTC")
    minutes = np.arange(len(timestamps))
    signs = np.where((minutes // 5) % 2 == 0, 1.0, -1.0) if zigzag else np.ones(len(minutes))
    log_prices = np.log(1.10) + np.cumsum(signs * STEP)
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

    # 288 five-minute prices: 287 returns of +-5 * STEP.
    assert list(result.columns) == list(COLUMNS)
    assert list(result["date"]) == [pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-17")]
    assert list(result["observation_count"]) == [288, 288]
    np.testing.assert_allclose(result["rv"], 287 * (5 * STEP) ** 2)
    assert np.isclose(cast(float, result.loc[0, "log_rv"]), np.log(287 * (5 * STEP) ** 2))


def test_semivariances_split_each_session_by_return_sign(tmp_path):
    result = build_daily_dataset([write(two_sessions(), tmp_path / "ticks.parquet")])

    # Each session's 287 returns come from bins 1 to 287 after its first:
    # 143 even (rising) bins and 144 odd (falling) ones.
    np.testing.assert_allclose(result["rs_pos"], 143 * (5 * STEP) ** 2)
    np.testing.assert_allclose(result["rs_neg"], 144 * (5 * STEP) ** 2)
    np.testing.assert_allclose(result["log_rs_pos"], np.log(result["rs_pos"]))
    np.testing.assert_allclose(result["log_rs_neg"], np.log(result["rs_neg"]))


def test_sessions_with_returns_of_one_sign_are_dropped(tmp_path):
    rising = two_sessions(zigzag=False)

    result = build_daily_dataset([write(rising, tmp_path / "ticks.parquet")])

    assert result.empty and list(result.columns) == list(COLUMNS)


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


def set_age(path: pathlib.Path, seconds: float) -> None:
    os.utime(path, (seconds, seconds))


def test_the_cache_is_built_once_and_then_read(tmp_path):
    ticks = write(two_sessions(), tmp_path / "ticks.parquet")
    cache = tmp_path / "derived" / "daily.parquet"

    built = load_daily_dataset([ticks], cache)
    marked = built.assign(rv=-1.0)  # a value no build produces
    marked.to_parquet(cache, index=False)
    set_age(ticks, 1_000)
    set_age(cache, 2_000)

    pd.testing.assert_frame_equal(built, build_daily_dataset([ticks]))
    pd.testing.assert_frame_equal(load_daily_dataset([ticks], cache), marked)


@pytest.mark.parametrize("stale", ["newer ticks", "other columns"])
def test_a_stale_cache_is_rebuilt(tmp_path, stale):
    ticks = write(two_sessions(), tmp_path / "ticks.parquet")
    cache = tmp_path / "daily.parquet"
    expected = build_daily_dataset([ticks])
    if stale == "newer ticks":
        expected.assign(rv=-1.0).to_parquet(cache, index=False)
        set_age(cache, 1_000)
        set_age(ticks, 2_000)
    else:
        expected.drop(columns="rs_pos").to_parquet(cache, index=False)
        set_age(ticks, 1_000)
        set_age(cache, 2_000)

    pd.testing.assert_frame_equal(load_daily_dataset([ticks], cache), expected)
    pd.testing.assert_frame_equal(pd.read_parquet(cache), expected)


def test_loading_needs_tick_files(tmp_path):
    with pytest.raises(ValueError, match="tick-data file"):
        load_daily_dataset([], tmp_path / "daily.parquet")
