"""Presentation shared by the notebooks: figures, tables, names and colours.

Every figure takes a mapping from an asset's display name to its data and draws
the same panel for each asset, stacked top to bottom.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rvol.evaluation import StudyDesign
from rvol.models import ForecastModel

#: Light figures, so they read on GitHub's white page; a dark editor can invert
#: them (PyCharm: "Invert image outputs for dark themes").
BACKGROUND = "#fcfcfb"
INK = "#52514e"
EDGE = "#8a8984"
GRID = "#e6e5e0"
#: Realized values are drawn in a neutral grey; models get the palette in order.
ACTUAL = "#8a8984"
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7c5cd6", "#3fa7b5")
#: CSS for the highlighted cell in a styled table. The text colour is set too, so
#: the cell stays readable when the editor's theme draws table text light.
HIGHLIGHT = "background-color: #cde2fb; color: #1e1f22"

STYLE = {
    "figure.dpi": 110,
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.size": 10,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "text.color": INK,
    "legend.labelcolor": INK,
}


def use_style() -> None:
    plt.rcParams.update(STYLE)


#: Every model name in the notebooks, so a model keeps its colour everywhere.
COLORS = dict(
    zip(
        ["HAR", "historical_mean", "naive", "EWMA", "AR1", "SHAR", "HAR-ridge"],
        PALETTE,
        strict=True,
    )
)
MAIN_COLOR = COLORS["HAR"]


def pair(asset: str) -> str:
    """Display name of a currency pair: EURUSD -> EUR/USD."""
    return f"{asset[:3]}/{asset[3:]}" if len(asset) == 6 else asset


def by_asset(frames: Mapping[str, pd.DataFrame], name: str = "asset") -> pd.DataFrame:
    """Stack per-asset tables under an asset level; with one asset, the table itself."""
    if len(frames) == 1:
        return next(iter(frames.values()))
    return pd.concat({pair(asset): frame for asset, frame in frames.items()}, names=[name])


def design_table(design: StudyDesign, assets: Sequence[str]) -> pd.DataFrame:
    """The study design as a one-column table, for the top of each notebook."""
    return pd.Series(
        {
            "assets": ", ".join(pair(asset) for asset in assets),
            "minimum training sessions": design.min_train_size,
            "first forecast target": design.forecast_start,
            "evaluation start": design.evaluation_start,
            "last forecast target": design.forecast_end,
            "candidate model": design.candidate,
            "loss": design.loss_name,
        },
        name="study design",
    ).to_frame()


def model_table(models: Sequence[ForecastModel]) -> pd.DataFrame:
    """One row per model: its rule, the rule's settings and its regressors."""
    rows = []
    for model in models:
        names = inspect.signature(type(model.rule)).parameters
        settings = ", ".join(f"{name} = {getattr(model.rule, name)}" for name in names)
        rows.append(
            {
                "rule": type(model.rule).__name__,
                "settings": settings,
                "regressors": ", ".join(model.regressors) or "none",
            }
        )
    return pd.DataFrame(rows, index=pd.Index([model.name for model in models], name="model"))


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
