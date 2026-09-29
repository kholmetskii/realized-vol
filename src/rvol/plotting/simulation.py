"""Plots for latent-price simulations and Monte Carlo validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from rvol.simulation.monte_carlo import MonteCarloSummary


def plot_heston_path(
    prices: ArrayLike,
    variance_path: ArrayLike,
    *,
    dt: float,
    theta: float | None = None,
    title: str = "Heston stochastic-volatility simulation",
) -> Figure:
    """Plot price, latent variance, and cumulative integrated variance."""
    price_values = np.asarray(prices, dtype=np.float64)
    variance_values = np.asarray(variance_path, dtype=np.float64)
    if price_values.ndim != 1 or variance_values.ndim != 1:
        raise ValueError("price and variance paths must be one-dimensional")
    if price_values.shape != variance_values.shape or len(price_values) < 2:
        raise ValueError("price and variance paths must have equal nontrivial length")
    if dt <= 0:
        raise ValueError("dt must be positive")

    steps = np.arange(len(price_values))
    cumulative_iv = np.r_[0.0, np.cumsum(variance_values[:-1]) * dt]
    figure, (price_axis, variance_axis, iv_axis) = plt.subplots(
        3,
        1,
        figsize=(9.0, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1, 1]},
    )

    price_axis.plot(steps, price_values, color="tab:blue", linewidth=1.3)
    price_axis.set_ylabel("Efficient price")
    price_axis.set_title("Simulated efficient-price path")
    price_axis.grid(alpha=0.25)

    variance_axis.plot(
        steps,
        variance_values,
        color="tab:purple",
        linewidth=1.2,
        label="Latent variance",
    )
    if theta is not None:
        variance_axis.axhline(
            theta,
            color="tab:orange",
            linestyle="--",
            linewidth=1.2,
            label=f"Long-run level θ = {theta:.3f}",
        )
        variance_axis.legend(frameon=False)
    variance_axis.set_ylabel("Instantaneous variance")
    variance_axis.set_title("Stochastic variance with mean reversion")
    variance_axis.grid(alpha=0.25)

    iv_axis.plot(steps, cumulative_iv, color="tab:green", linewidth=1.5)
    iv_axis.annotate(
        f"Final IV = {cumulative_iv[-1]:.6f}",
        xy=(steps[-1], cumulative_iv[-1]),
        xytext=(-8, -16),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9,
    )
    iv_axis.set_xlabel("Simulation step")
    iv_axis.set_ylabel("Integrated variance")
    iv_axis.set_title("Accumulated variance over the simulated period")
    iv_axis.grid(alpha=0.25)

    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_simulation_overview(
    efficient_prices: ArrayLike,
    observed_prices: ArrayLike,
    variance_path: ArrayLike,
    scenarios: Mapping[str, Sequence[MonteCarloSummary]],
    *,
    zoom_observations: int = 300,
    title: str = "Simulation and Monte Carlo validation",
) -> Figure:
    """Plot latent state, observed noise, estimator bias, and estimator RMSE."""
    efficient = np.asarray(efficient_prices, dtype=np.float64)
    observed = np.asarray(observed_prices, dtype=np.float64)
    variance = np.asarray(variance_path, dtype=np.float64)
    if efficient.shape != observed.shape or efficient.shape != variance.shape:
        raise ValueError("price and variance paths must have the same shape")
    if not scenarios:
        raise ValueError("at least one Monte Carlo scenario is required")

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    price_axis, variance_axis, bias_axis, rmse_axis = axes.flat

    zoom = min(zoom_observations, len(efficient))
    time = np.arange(zoom)
    price_axis.plot(time, efficient[:zoom], label="Efficient price", linewidth=1.7)
    price_axis.plot(time, observed[:zoom], label="Observed price", linewidth=1, alpha=0.75)
    price_axis.set_xlabel("Simulation step")
    price_axis.set_ylabel("Price")
    price_axis.set_title("Observation noise around the efficient price")
    price_axis.grid(alpha=0.25)
    price_axis.legend(frameon=False)

    variance_axis.plot(np.arange(len(variance)), variance, color="tab:purple", linewidth=1.3)
    variance_axis.set_xlabel("Simulation step")
    variance_axis.set_ylabel("Instantaneous variance")
    variance_axis.set_title("Latent Heston variance")
    variance_axis.grid(alpha=0.25)

    scenario_names = list(scenarios)
    estimator_names = [summary.estimator for summary in scenarios[scenario_names[0]]]
    positions = np.arange(len(scenario_names))
    width = 0.8 / len(estimator_names)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")

    for estimator_index, estimator in enumerate(estimator_names):
        bias_values = []
        rmse_values = []
        for scenario in scenario_names:
            summary = next(
                item for item in scenarios[scenario] if item.estimator == estimator
            )
            bias_values.append(summary.relative_bias * 100)
            rmse_values.append(summary.rmse / summary.mean_target * 100)
        shift = (estimator_index - (len(estimator_names) - 1) / 2) * width
        bias_axis.bar(
            positions + shift,
            bias_values,
            width,
            label=estimator,
            color=colors[estimator_index],
        )
        rmse_axis.bar(
            positions + shift,
            rmse_values,
            width,
            label=estimator,
            color=colors[estimator_index],
        )

    for axis, ylabel, panel_title in (
        (bias_axis, "Bias / true IV, %", "Relative bias"),
        (rmse_axis, "RMSE / true IV, %", "Relative RMSE"),
    ):
        axis.axhline(0, color="0.35", linewidth=1)
        axis.set_xticks(positions, scenario_names)
        axis.set_ylabel(ylabel)
        axis.set_title(panel_title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)

    figure.suptitle(title)
    figure.tight_layout()
    return figure
