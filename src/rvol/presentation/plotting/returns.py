"""Plots connecting prices, log returns, squared returns, and realized variance."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_returns_and_rv(
    prices: ArrayLike,
    *,
    title: str = "From prices to realized variance",
) -> Figure:
    """Visualise every transformation in ``RV = sum(log-return squared)``."""
    price_values = np.asarray(prices, dtype=np.float64)
    if price_values.ndim != 1 or len(price_values) < 3:
        raise ValueError("prices must be one-dimensional with at least three values")
    if not np.all(np.isfinite(price_values)) or np.any(price_values <= 0):
        raise ValueError("prices must be finite and strictly positive")

    log_returns = np.diff(np.log(price_values))
    squared_returns = log_returns**2
    cumulative_rv = np.r_[0.0, np.cumsum(squared_returns)]
    price_steps = np.arange(len(price_values))
    return_steps = price_steps[1:]

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(10.0, 10.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1, 1, 1]},
    )
    price_axis, return_axis, squared_axis, cumulative_axis = axes

    price_axis.plot(price_steps, price_values, color="tab:blue", linewidth=1.4)
    price_axis.set_ylabel("Price")
    price_axis.set_title("1. Price path")
    price_axis.grid(alpha=0.25)

    return_axis.axhline(0, color="0.35", linewidth=0.9)
    return_axis.plot(return_steps, log_returns, color="tab:orange", linewidth=0.9)
    return_axis.set_ylabel("Log return")
    return_axis.set_title("2. Consecutive log-price differences")
    return_axis.grid(alpha=0.25)

    squared_axis.plot(
        return_steps,
        squared_returns,
        color="tab:purple",
        linewidth=0.9,
    )
    squared_axis.fill_between(
        return_steps,
        0,
        squared_returns,
        color="tab:purple",
        alpha=0.18,
    )
    largest = np.argsort(squared_returns)[-3:]
    squared_axis.scatter(
        return_steps[largest],
        squared_returns[largest],
        color="tab:red",
        s=28,
        zorder=3,
        label="Three largest contributions",
    )
    squared_axis.set_ylabel("Squared return")
    squared_axis.set_title("3. Every move becomes a positive variance contribution")
    squared_axis.grid(alpha=0.25)
    squared_axis.legend(frameon=False, loc="upper left")

    cumulative_axis.plot(
        price_steps,
        cumulative_rv,
        color="tab:green",
        linewidth=1.6,
    )
    cumulative_axis.annotate(
        f"Final RV = {cumulative_rv[-1]:.6f}",
        xy=(price_steps[-1], cumulative_rv[-1]),
        xytext=(-8, -16),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9,
    )
    cumulative_axis.set_xlabel("Observation")
    cumulative_axis.set_ylabel("Cumulative RV")
    cumulative_axis.set_title("4. Cumulative sum of squared log returns")
    cumulative_axis.grid(alpha=0.25)

    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_cumulative_iv_vs_rv(
    prices: ArrayLike,
    variance: ArrayLike,
    dt: float,
    *,
    title: str = "True cumulative integrated variance vs realized variance",
) -> Figure:
    """Compare the known Heston variance integral with RV along one path."""
    price_values = np.asarray(prices, dtype=np.float64)
    variance_values = np.asarray(variance, dtype=np.float64)
    if price_values.ndim != 1 or variance_values.ndim != 1:
        raise ValueError("prices and variance must be one-dimensional")
    if len(price_values) < 3 or len(price_values) != len(variance_values):
        raise ValueError("prices and variance must have the same length of at least three")
    if not np.all(np.isfinite(price_values)) or np.any(price_values <= 0):
        raise ValueError("prices must be finite and strictly positive")
    if not np.all(np.isfinite(variance_values)) or np.any(variance_values < 0):
        raise ValueError("variance must be finite and non-negative")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    squared_returns = np.diff(np.log(price_values)) ** 2
    cumulative_rv = np.r_[0.0, np.cumsum(squared_returns)]
    cumulative_iv = np.r_[0.0, np.cumsum(variance_values[:-1] * dt)]
    difference = cumulative_rv - cumulative_iv
    steps = np.arange(len(price_values))

    figure, (comparison_axis, difference_axis) = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    comparison_axis.plot(
        steps,
        cumulative_iv,
        color="black",
        linewidth=2.0,
        label="True cumulative IV",
    )
    comparison_axis.plot(
        steps,
        cumulative_rv,
        color="tab:green",
        linewidth=1.5,
        label="Cumulative RV",
    )
    comparison_axis.fill_between(
        steps,
        cumulative_iv,
        cumulative_rv,
        color="tab:orange",
        alpha=0.18,
        label="Estimation gap",
    )
    comparison_axis.scatter(
        [steps[-1], steps[-1]],
        [cumulative_iv[-1], cumulative_rv[-1]],
        color=["black", "tab:green"],
        s=35,
        zorder=3,
    )
    comparison_axis.set_ylabel("Cumulative variance")
    comparison_axis.set_title("Known variance integral and price-based estimate")
    comparison_axis.grid(alpha=0.25)
    comparison_axis.legend(frameon=False, loc="upper left")
    comparison_axis.text(
        0.98,
        0.05,
        f"Final IV = {cumulative_iv[-1]:.6f}\nFinal RV = {cumulative_rv[-1]:.6f}",
        transform=comparison_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )

    difference_axis.axhline(0, color="0.35", linewidth=1.0)
    difference_axis.plot(steps, difference, color="tab:orange", linewidth=1.4)
    difference_axis.fill_between(
        steps,
        0,
        difference,
        where=difference >= 0,
        color="tab:red",
        alpha=0.18,
        label="RV above IV",
    )
    difference_axis.fill_between(
        steps,
        0,
        difference,
        where=difference < 0,
        color="tab:blue",
        alpha=0.18,
        label="RV below IV",
    )
    difference_axis.set_xlabel("Observation")
    difference_axis.set_ylabel("RV − IV")
    difference_axis.set_title("Running estimation error")
    difference_axis.grid(alpha=0.25)
    difference_axis.legend(frameon=False, loc="best", ncols=2)

    figure.suptitle(title)
    figure.tight_layout()
    return figure
