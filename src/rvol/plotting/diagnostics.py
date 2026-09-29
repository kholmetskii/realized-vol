"""Plots for volatility signatures and market-microstructure diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from rvol.diagnostics.microstructure import NoiseTest

SERIES_STYLE = {
    "mid": {"color": "tab:blue", "marker": "o"},
    "bid": {"color": "tab:orange", "marker": "s"},
    "ask": {"color": "tab:green", "marker": "^"},
}


def plot_volatility_signature(
    signatures: Mapping[str, pd.DataFrame],
    *,
    title: str = "Volatility signature",
) -> Figure:
    """Plot one or more annualised-volatility signatures with uncertainty."""
    if not signatures:
        raise ValueError("at least one signature is required")

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for index, (name, values) in enumerate(signatures.items()):
        style = SERIES_STYLE.get(
            name,
            {"color": f"C{index}", "marker": "o"},
        )
        axis.errorbar(
            values["seconds"],
            values["ann_vol_pct"],
            yerr=values["ann_vol_se"],
            color=style["color"],
            marker=style["marker"],
            label=name,
            linewidth=1.5,
            markersize=4.5,
            capsize=2.5,
            elinewidth=0.9,
        )

    axis.axvline(300, linestyle="--", linewidth=1, color="0.45")
    axis.annotate("5 min", xy=(300, axis.get_ylim()[0]), xytext=(5, 5),
                  textcoords="offset points", color="0.35", fontsize=9)
    axis.set_xscale("log")
    axis.set_xlabel("Sampling interval, seconds (log scale)")
    axis.set_ylabel("Annualised volatility from RV, %")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    if len(signatures) > 1:
        axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def plot_diagnostics_overview(
    signatures: Mapping[str, pd.DataFrame],
    noise_tests: Mapping[str, NoiseTest],
    *,
    title: str = "EUR/USD realized-volatility diagnostics",
) -> Figure:
    """Combine volatility signatures and fine-versus-coarse noise estimates."""
    if not signatures or not noise_tests:
        raise ValueError("signatures and noise_tests must not be empty")

    figure, (signature_axis, noise_axis) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        gridspec_kw={"width_ratios": [1.7, 1]},
    )

    for index, (name, values) in enumerate(signatures.items()):
        style = SERIES_STYLE.get(name, {"color": f"C{index}", "marker": "o"})
        signature_axis.errorbar(
            values["seconds"],
            values["ann_vol_pct"],
            yerr=values["ann_vol_se"],
            color=style["color"],
            marker=style["marker"],
            label=name,
            linewidth=1.5,
            markersize=4,
            capsize=2,
            elinewidth=0.8,
        )

    signature_axis.axvline(300, linestyle="--", linewidth=1, color="0.45")
    signature_axis.set_xscale("log")
    signature_axis.set_xlabel("Sampling interval, seconds (log scale)")
    signature_axis.set_ylabel("Annualised volatility, %")
    signature_axis.set_title("Volatility signature")
    signature_axis.grid(alpha=0.25)
    signature_axis.legend(frameon=False)

    names = list(noise_tests)
    results = [noise_tests[name] for name in names]
    inflation = np.array([result.mean_ratio - 1 for result in results]) * 100
    colors = [SERIES_STYLE.get(name, {"color": f"C{i}"})["color"]
              for i, name in enumerate(names)]
    bars = noise_axis.bar(names, inflation, color=colors, alpha=0.85)
    noise_axis.axhline(0, linewidth=1, color="0.35")
    noise_axis.set_ylim(0, float(np.max(inflation)) * 1.28)
    noise_axis.set_ylabel("RV inflation: fine / coarse − 1, %")
    noise_axis.set_title("Microstructure-noise diagnostic", pad=12)
    noise_axis.grid(axis="y", alpha=0.25)
    for bar, result in zip(bars, results, strict=True):
        noise_axis.annotate(
            f"{result.implied_noise_bps:.3f} bps\np={result.p_value:.2g}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.suptitle(title)
    figure.tight_layout()
    return figure
