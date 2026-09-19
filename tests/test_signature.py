"""Signature plot on synthetic data: a true price plus microstructure noise.

Expectation: with no noise the curve is flat; with noise it rises toward high
frequencies. This is the reference the real-data figure gets compared against.
"""

import numpy as np
import pandas as pd
import pytest

from rvol.estimators.signature import (
    full_sessions,
    signature,
    trading_day,
)

SIGMA_ANN = 0.10
N_DAYS = 20
PER_DAY = 24 * 60 * 60          # one tick per second


def build_paths(seed: int = 0) -> pd.DataFrame:
    """One continuous path, ticking every second, split into sessions later.

    The path must not restart at each day boundary: a session runs 21:00 to
    21:00 UTC, so a daily reset would sit inside a session and be counted as a
    huge return.
    """
    rng = np.random.default_rng(seed)
    dt_ = 1 / 252 / PER_DAY
    sd = SIGMA_ANN * np.sqrt(dt_)
    n = N_DAYS * PER_DAY
    start = pd.Timestamp("2024-01-15 21:00", tz="UTC")

    log_p = np.log(1.10) + np.cumsum(rng.normal(0, sd, n))
    ts = start + pd.to_timedelta(np.arange(n), unit="s")
    return pd.DataFrame({"ts": ts, "true": np.exp(log_p)})


def tick_to_five_min_ratio(df: pd.DataFrame) -> float:
    sig = signature(df)
    return sig["mean_RV"].iloc[0] / sig.loc[sig["freq"] == "5min", "mean_RV"].iloc[0]


@pytest.mark.slow
def test_flat_without_noise():
    df = build_paths()
    df["mid"] = df["true"]
    ratio = tick_to_five_min_ratio(df)
    assert 0.8 < ratio < 1.25, f"curve should be flat without noise, got {ratio}"


@pytest.mark.slow
def test_rises_with_noise():
    rng = np.random.default_rng(1)
    df = build_paths()
    df["mid"] = df["true"] * np.exp(rng.normal(0, 0.5e-4, len(df)))
    ratio = tick_to_five_min_ratio(df)
    assert ratio > 2, f"tick-scale RV should blow up with noise, got {ratio}"


@pytest.mark.slow
def test_five_minute_estimate_recovers_true_volatility():
    """The point of the plateau: at 5 minutes the estimate is roughly right."""
    rng = np.random.default_rng(2)
    df = build_paths()
    df["mid"] = df["true"] * np.exp(rng.normal(0, 0.5e-4, len(df)))
    sig = signature(df)
    est = sig.loc[sig["freq"] == "5min", "ann_vol_pct"].iloc[0] / 100
    assert abs(est / SIGMA_ANN - 1.0) < 0.15


def test_trading_day_labels_sunday_evening_as_monday():
    """A session runs 21:00 UTC to 21:00 UTC and is named after the day it ends."""
    ts = pd.DatetimeIndex([
        pd.Timestamp("2024-01-14 22:30", tz="UTC"),   # Sunday, after the open
        pd.Timestamp("2024-01-15 14:00", tz="UTC"),   # Monday, London afternoon
        pd.Timestamp("2024-01-15 20:59", tz="UTC"),   # Monday, just before close
        pd.Timestamp("2024-01-15 21:30", tz="UTC"),   # already Tuesday's session
    ])
    labels = trading_day(ts)
    assert list(labels[:3]) == [pd.Timestamp("2024-01-15")] * 3
    assert labels[3] == pd.Timestamp("2024-01-16")


def test_short_sessions_are_dropped():
    """A two-hour Sunday is not a day; keeping it would bias every average."""
    full = pd.date_range("2024-01-15 21:00", periods=24, freq="1h", tz="UTC")
    short = pd.date_range("2024-01-19 21:00", periods=2, freq="1h", tz="UTC")
    s = pd.Series(1.0, index=full.append(short))

    kept = full_sessions(s)
    assert len(kept) == len(full)
    assert set(trading_day(kept.index)) == {pd.Timestamp("2024-01-16")}


@pytest.mark.slow
def test_partial_days_do_not_drag_the_average_down():
    df = build_paths()
    df["mid"] = df["true"]
    full = signature(df, freqs=["5min"])

    stub = df[df["ts"] < df["ts"].iloc[0] + pd.Timedelta(hours=2)].copy()
    stub["ts"] = stub["ts"] + pd.Timedelta(days=N_DAYS + 1)
    with_stub = signature(pd.concat([df, stub], ignore_index=True), freqs=["5min"])

    assert with_stub["n_days"].iloc[0] == full["n_days"].iloc[0]
    assert abs(with_stub["mean_RV"].iloc[0] / full["mean_RV"].iloc[0] - 1) < 1e-9
