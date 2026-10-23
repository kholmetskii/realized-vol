"""Figures shared by the asset notebooks: one style and one colour per model."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

#: Realized values are drawn in a neutral grey; models get the palette in order.
ACTUAL = "#8a8984"
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7c5cd6", "#3fa7b5")
BACKGROUND = "#fcfcfb"
INK = "#52514e"

STYLE = {
    "figure.dpi": 110,
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "axes.edgecolor": ACTUAL,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.size": 10,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
}


def use_style() -> None:
    plt.rcParams.update(STYLE)


def model_colors(models: Sequence[str]) -> dict[str, str]:
    """One palette colour per model, in the order given."""
    if len(models) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} models have a colour")
    return dict(zip(models, PALETTE, strict=False))


def _month_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def volatility_history(
    dates: pd.Series, vol: np.ndarray, log_rv: pd.Series, *, pair: str, color: str, window: int
) -> Figure:
    """Annualized volatility with its trailing mean above, log RV below."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(dates, vol, color=ACTUAL, linewidth=0.8, label="daily")
    axes[0].plot(
        dates,
        pd.Series(vol).rolling(window).mean(),
        color=color,
        linewidth=2,
        label=f"{window}-session mean",
    )
    axes[0].set_title(f"{pair} realized volatility, annualized (%)")
    axes[0].legend(loc="upper left")
    axes[1].plot(dates, log_rv, color=color, linewidth=0.8)
    axes[1].set_title("Log realized variance, the series the models forecast")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    _month_axis(axes[1])
    fig.tight_layout()
    return fig


def distributions(columns: dict[str, pd.Series], *, color: str) -> Figure:
    """One histogram per named series, side by side."""
    fig, axes = plt.subplots(1, len(columns), figsize=(10, 3.2))
    for ax, (title, values) in zip(np.atleast_1d(axes), columns.items(), strict=True):
        ax.hist(values, bins=60, color=color, edgecolor=BACKGROUND, linewidth=0.5)
        ax.set_title(title)
        ax.set_ylabel("sessions")
    fig.tight_layout()
    return fig


def autocorrelation(acf: pd.Series, *, color: str) -> Figure:
    """Autocorrelation by lag, as bars."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(acf.index, acf.to_numpy(), width=0.7, color=color)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_title("Autocorrelation of log RV")
    ax.set_xlabel("lag (sessions)")
    fig.tight_layout()
    return fig


def forecasts_vs_realized(
    dates: pd.Series,
    realized: np.ndarray,
    paths: dict[str, np.ndarray],
    colors: dict[str, str],
    *,
    highlight: str,
) -> Figure:
    """Realized volatility and each model's forecast path, on the same axis."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(dates, realized, color=ACTUAL, linewidth=0.9, label="realized")
    for model, path in paths.items():
        width = 2 if model == highlight else 1.2
        ax.plot(dates, path, color=colors[model], linewidth=width, label=f"{model} forecast")
    ax.set_title("Next-session volatility: forecasts vs realized (annualized %)")
    ax.legend(loc="upper left", ncols=len(paths) + 1)
    _month_axis(ax)
    fig.tight_layout()
    return fig


def cumulative_advantage(
    advantage: pd.DataFrame, colors: dict[str, str], *, candidate: str
) -> Figure:
    """Running sum of benchmark loss minus candidate loss, one line per benchmark."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for model in advantage.columns:
        ax.plot(advantage.index, advantage[model], color=colors[model], label=model)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_title(f"Cumulative QLIKE advantage of {candidate} over each benchmark")
    ax.set_ylabel("sum of loss differences")
    ax.legend(loc="upper left")
    _month_axis(ax)
    fig.tight_layout()
    return fig
