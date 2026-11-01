"""Figures shared by the notebooks: one style, one colour per model, one panel per asset.

Every figure takes a mapping from an asset's display name to its data and draws
the same panel for each asset, stacked top to bottom.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
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


def _grid(rows: int, cols: int, row_height: float, *, sharex: bool = False) -> np.ndarray:
    _, axes = plt.subplots(
        rows, cols, figsize=(10, row_height * rows), squeeze=False, sharex=sharex
    )
    return axes


def _month_axis(ax: Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def _finish(axes: np.ndarray) -> Figure:
    figure = axes.flat[0].figure
    figure.tight_layout()
    return figure


def volatility_history(
    histories: Mapping[str, tuple[pd.Series, np.ndarray, pd.Series]], *, color: str, window: int
) -> Figure:
    """Per asset: annualized volatility with its trailing mean, then log RV below it."""
    axes = _grid(2 * len(histories), 1, 2.75)
    for i, (pair, (dates, vol, log_rv)) in enumerate(histories.items()):
        top, bottom = axes[2 * i, 0], axes[2 * i + 1, 0]
        top.plot(dates, vol, color=ACTUAL, linewidth=0.8, label="daily")
        top.plot(
            dates,
            pd.Series(vol).rolling(window).mean(),
            color=color,
            linewidth=2,
            label=f"{window}-session mean",
        )
        top.set_title(f"{pair} realized volatility, annualized (%)")
        top.legend(loc="upper left")
        top.sharex(bottom)
        top.tick_params(labelbottom=False)
        bottom.plot(dates, log_rv, color=color, linewidth=0.8)
        bottom.set_title(f"{pair} log realized variance, the series the models forecast")
        bottom.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        _month_axis(bottom)
    return _finish(axes)


def distributions(columns: Mapping[str, Mapping[str, pd.Series]], *, color: str) -> Figure:
    """Per asset: one histogram per named series, side by side."""
    width = max(len(series) for series in columns.values())
    axes = _grid(len(columns), width, 3.2)
    for row, (pair, series) in zip(axes, columns.items(), strict=True):
        for ax, (title, values) in zip(row, series.items(), strict=False):
            ax.hist(values, bins=60, color=color, edgecolor=BACKGROUND, linewidth=0.5)
            ax.set_title(f"{pair} {title}")
            ax.set_ylabel("sessions")
    return _finish(axes)


def autocorrelation(acfs: Mapping[str, pd.Series], *, color: str) -> Figure:
    """Per asset: autocorrelation of log RV by lag, as bars."""
    axes = _grid(len(acfs), 1, 3)
    for ax, (pair, acf) in zip(axes[:, 0], acfs.items(), strict=True):
        ax.bar(acf.index, acf.to_numpy(), width=0.7, color=color)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_title(f"{pair} autocorrelation of log RV")
        ax.set_xlabel("lag (sessions)")
    return _finish(axes)


def forecasts_vs_realized(
    panels: Mapping[str, tuple[pd.Series, np.ndarray, Mapping[str, np.ndarray]]],
    colors: Mapping[str, str],
    *,
    highlight: str,
) -> Figure:
    """Per asset: realized volatility and each model's forecast path."""
    axes = _grid(len(panels), 1, 3.8)
    for ax, (pair, (dates, realized, paths)) in zip(axes[:, 0], panels.items(), strict=True):
        ax.plot(dates, realized, color=ACTUAL, linewidth=0.9, label="realized")
        for model, path in paths.items():
            width = 2 if model == highlight else 1.2
            ax.plot(dates, path, color=colors[model], linewidth=width, label=f"{model} forecast")
        ax.set_title(f"{pair} next-session volatility: forecasts vs realized (annualized %)")
        ax.legend(loc="upper left", ncols=len(paths) + 1)
        _month_axis(ax)
    return _finish(axes)


def cumulative_advantage(
    advantages: Mapping[str, pd.DataFrame], colors: Mapping[str, str], *, title: str
) -> Figure:
    """Per asset: the running sum of each column of loss differences."""
    axes = _grid(len(advantages), 1, 3.8)
    for ax, (pair, advantage) in zip(axes[:, 0], advantages.items(), strict=True):
        for model in advantage.columns:
            ax.plot(advantage.index, advantage[model], color=colors[model], label=model)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_title(f"{pair} {title}")
        ax.set_ylabel("sum of loss differences")
        ax.legend(loc="upper left")
        _month_axis(ax)
    return _finish(axes)
