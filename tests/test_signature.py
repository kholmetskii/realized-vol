"""Signature plot on synthetic data: a true price plus microstructure noise.

Expectation: with no noise the curve is flat; with noise it rises toward high
frequencies. This is the reference the real-data figure gets compared against.
"""

import numpy as np
import pandas as pd
import pytest

from rvol.estimators.signature import (
    full_sessions,
    noise_test,
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


@pytest.mark.slow
def test_noise_test_finds_nothing_when_there_is_no_noise():
    df = build_paths()
    df["mid"] = df["true"]
    result = noise_test(df, fine="1s", coarse="5min")

    assert result.p_value > 0.05, f"false positive: {result}"
    assert abs(result.mean_ratio - 1.0) < 0.05
    assert result.implied_noise_bps < 0.05


@pytest.mark.slow
def test_noise_test_detects_noise_and_recovers_its_size():
    """With known noise sd, the implied figure should come back close to it."""
    omega = 1e-4                      # 1 bp per observation
    rng = np.random.default_rng(3)
    df = build_paths()
    df["mid"] = df["true"] * np.exp(rng.normal(0, omega, len(df)))
    result = noise_test(df, fine="1s", coarse="5min")

    assert result.p_value < 1e-6, f"noise should be obvious: {result}"
    assert result.mean_ratio > 2
    assert 0.8 < result.implied_noise_bps / (omega * 1e4) < 1.2


def build_paths_varying_vol(seed: int = 4) -> pd.DataFrame:
    """As build_paths, but each session gets its own volatility level.

    Real volatility clusters: a calm week and a turbulent one sit in the same
    sample. That shared, day-level variation is what pairing removes.
    """
    rng = np.random.default_rng(seed)
    dt_ = 1 / 252 / PER_DAY
    n = N_DAYS * PER_DAY
    start = pd.Timestamp("2024-01-15 21:00", tz="UTC")

    per_day = SIGMA_ANN * np.exp(rng.normal(0, 0.5, N_DAYS))   # vol clustering
    sd = np.repeat(per_day, PER_DAY) * np.sqrt(dt_)

    log_p = np.log(1.10) + np.cumsum(rng.normal(0, 1, n) * sd)
    ts = start + pd.to_timedelta(np.arange(n), unit="s")
    return pd.DataFrame({"ts": ts, "true": np.exp(log_p)})


@pytest.mark.slow
def test_pairing_beats_comparing_two_averages():
    """The paired test is the point: it cancels day-to-day volatility swings."""
    rng = np.random.default_rng(5)
    df = build_paths_varying_vol()
    df["mid"] = df["true"] * np.exp(rng.normal(0, 2e-5, len(df)))

    result = noise_test(df, fine="1s", coarse="5min")
    sig = signature(df, freqs=["1s", "5min"])
    unpaired_se = (sig["se"] / sig["mean_RV"]).max()

    # Not a huge factor: the noise term is additive and the volatility level
    # is not, so the ratio itself still varies across days — quiet days show a
    # larger one. Pairing removes the common level, not that.
    assert result.se_log_ratio < unpaired_se / 1.5, (
        f"paired se {result.se_log_ratio:.4f} vs unpaired {unpaired_se:.4f}"
    )
