"""Time-series plots for daily realized-variance estimates."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_daily_realized_variance(
    daily_close: ArrayLike,
    daily_rv: ArrayLike,
    *,
    rolling_window: int = 20,
    title: str = "Daily realized variance over time",
) -> Figure:
    """Plot daily closing prices and their corresponding daily RV estimates."""
    closes = np.asarray(daily_close, dtype=np.float64)
    rv = np.asarray(daily_rv, dtype=np.float64)
    if closes.ndim != 1 or rv.ndim != 1:
        raise ValueError("daily_close and daily_rv must be one-dimensional")
    if len(closes) < 2 or len(closes) != len(rv):
        raise ValueError("daily_close and daily_rv must have equal length of at least two")
    if not np.all(np.isfinite(closes)) or np.any(closes <= 0):
        raise ValueError("daily_close must be finite and strictly positive")
    if not np.all(np.isfinite(rv)) or np.any(rv < 0):
        raise ValueError("daily_rv must be finite and non-negative")
    if not 2 <= rolling_window <= len(rv):
        raise ValueError("rolling_window must be between two and the number of days")

    days = np.arange(1, len(rv) + 1)
    rolling_mean = np.full(len(rv), np.nan)
    rolling_mean[rolling_window - 1 :] = np.convolve(
        rv,
        np.ones(rolling_window) / rolling_window,
        mode="valid",
    )
    high_threshold = float(np.quantile(rv, 0.9))
    high_days = rv >= high_threshold

    figure, (price_axis, rv_axis) = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.7]},
    )

    price_axis.plot(days, closes, color="tab:blue", linewidth=1.4)
    price_axis.set_ylabel("Closing price")
    price_axis.set_title("1. Simulated daily closing price")
    price_axis.grid(alpha=0.25)

    rv_axis.bar(
        days,
        rv,
        color="tab:blue",
        alpha=0.55,
        width=0.9,
        label="Daily RV",
    )
    rv_axis.bar(
        days[high_days],
        rv[high_days],
        color="tab:red",
        alpha=0.75,
        width=0.9,
        label="Top 10% RV days",
    )
    rv_axis.plot(
        days,
        rolling_mean,
        color="black",
        linewidth=2.0,
        label=f"{rolling_window}-day average",
    )
    rv_axis.axhline(
        np.mean(rv),
        color="tab:orange",
        linestyle="--",
        linewidth=1.3,
        label="Full-period average",
    )
    highest_day = int(np.argmax(rv))
    rv_axis.annotate(
        f"Highest RV\nday {days[highest_day]}",
        xy=(days[highest_day], rv[highest_day]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "0.35"},
    )
    rv_axis.set_xlabel("Trading day")
    rv_axis.set_ylabel("Daily RV")
    rv_axis.set_title("2. Squared intraday returns accumulated within each day")
    rv_axis.grid(axis="y", alpha=0.25)
    rv_axis.legend(frameon=False, loc="upper right", ncols=2)

    figure.suptitle(title)
    figure.tight_layout()
    return figure
