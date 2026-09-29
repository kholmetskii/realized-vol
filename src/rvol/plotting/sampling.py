"""Plots explaining regular grids and subsampled realized variance."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from rvol.estimators.realized import realized_variance


def plot_sampling_grids(
    prices: ArrayLike,
    *,
    step: int,
    title: str = "Sampling grids and subsampled realized variance",
) -> Figure:
    """Show selected observations and RV dispersion across all grid origins."""
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 * step:
        raise ValueError("prices must be one-dimensional and span at least two steps")
    if step < 2:
        raise ValueError("step must be at least two to compare grid origins")

    offsets = np.arange(step)
    grid_rv = np.array([
        realized_variance(values, step=step, offset=int(offset))
        for offset in offsets
    ])

    figure, (path_axis, rv_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    observations = np.arange(len(values))
    path_axis.plot(observations, values, color="0.55", linewidth=1, label="All prices")
    shown_offsets = [0, 1] if step > 1 else [0]
    for color, marker, offset in zip(
        ("tab:blue", "tab:orange"),
        ("o", "s"),
        shown_offsets,
        strict=True,
    ):
        selected = observations[offset::step]
        path_axis.plot(
            selected,
            values[offset::step],
            color=color,
            marker=marker,
            linewidth=1.4,
            markersize=5,
            label=f"Grid offset {offset}",
        )

    path_axis.set_ylabel("Price")
    path_axis.set_title("Same path, different grid origins")
    path_axis.grid(alpha=0.25)
    path_axis.legend(frameon=False)

    bars = rv_axis.bar(offsets, grid_rv, color="tab:blue", alpha=0.8)
    average = float(np.mean(grid_rv))
    rv_axis.axhline(
        average,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"Subsampled mean = {average:.6f}",
    )
    rv_axis.set_ylim(0, float(np.max(grid_rv)) * 1.22)
    rv_axis.set_xlabel("Grid offset")
    rv_axis.set_ylabel("Realized variance")
    rv_axis.set_title("RV changes with the grid origin")
    rv_axis.set_xticks(offsets)
    rv_axis.grid(axis="y", alpha=0.25)
    rv_axis.legend(frameon=False)
    for bar, estimate in zip(bars, grid_rv, strict=True):
        rv_axis.annotate(
            f"{estimate:.5f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.suptitle(title)
    figure.tight_layout()
    return figure
