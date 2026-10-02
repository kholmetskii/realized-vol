from typing import cast

import numpy as np
import pandas as pd

from rvol.features.dataset import build_daily_dataset
from rvol.features.realized import sampled_realized_variance


def two_sessions() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-15 22:00",
        "2024-01-17 21:59",
        freq="1min",
        tz="UTC",
    )
    log_prices = np.log(1.10) + np.arange(len(timestamps)) * 1e-6
    return pd.DataFrame({"ts": timestamps, "mid": np.exp(log_prices)})


def test_builder_preserves_sessions_split_across_files(tmp_path):
    ticks = two_sessions()
    split = pd.Timestamp("2024-01-16 00:00", tz="UTC")
    first = tmp_path / "ticks_1.parquet"
    second = tmp_path / "ticks_2.parquet"
    ticks[ticks["ts"] < split].to_parquet(first, index=False)
    ticks[ticks["ts"] >= split].to_parquet(second, index=False)

    result = build_daily_dataset([first, second], frequency="5min")

    assert list(result.columns) == ["date", "rv_5min", "log_rv", "observation_count"]
    assert list(result["date"]) == [pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-17")]
    first_session = ticks[ticks["ts"] < pd.Timestamp("2024-01-16 22:00", tz="UTC")]
    expected = sampled_realized_variance(first_session.set_index("ts")["mid"], "5min")
    assert np.isclose(cast(float, result.loc[0, "rv_5min"]), expected)
    assert np.isclose(cast(float, result.loc[0, "log_rv"]), np.log(expected))


def test_builder_deduplicates_overlapping_chunks_and_is_repeatable(tmp_path):
    ticks = two_sessions()
    boundary = pd.Timestamp("2024-01-16 00:00", tz="UTC")
    first = tmp_path / "ticks_1.parquet"
    second = tmp_path / "ticks_2.parquet"
    complete = tmp_path / "ticks_complete.parquet"
    ticks[ticks["ts"] < pd.Timestamp("2024-01-17 00:00", tz="UTC")].to_parquet(
        first, index=False
    )
    ticks[ticks["ts"] >= boundary].to_parquet(second, index=False)
    ticks.to_parquet(complete, index=False)

    first_run = build_daily_dataset([first, second])
    second_run = build_daily_dataset([second, first])
    expected = build_daily_dataset([complete])

    pd.testing.assert_frame_equal(first_run, second_run)
    pd.testing.assert_frame_equal(first_run, expected)
    assert first_run["date"].is_unique
    assert len(first_run) == 2
