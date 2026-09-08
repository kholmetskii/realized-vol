"""Signature plot on synthetic data: a true price plus microstructure noise.

Expectation: with no noise the curve is flat; with noise it rises toward high
frequencies. This is the reference the real-data figure gets compared against.
"""

import numpy as np
import pandas as pd
import pytest

from rvol.estimators.signature import signature

SIGMA_ANN = 0.10
N_DAYS = 20
PER_DAY = 24 * 60 * 60          # one tick per second


def build_paths(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt_ = 1 / 252 / PER_DAY
    sd = SIGMA_ANN * np.sqrt(dt_)
    start = pd.Timestamp("2024-01-15", tz="UTC")

    frames = []
    for d in range(N_DAYS):
        log_p = np.log(1.10) + np.cumsum(rng.normal(0, sd, PER_DAY))
        ts = start + pd.Timedelta(days=d) + pd.to_timedelta(np.arange(PER_DAY), unit="s")
        frames.append(pd.DataFrame({"ts": ts, "true": np.exp(log_p)}))
    return pd.concat(frames, ignore_index=True)


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
