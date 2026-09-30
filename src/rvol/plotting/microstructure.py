"""Plots explaining quotes, transaction prices, and bid-ask bounce."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def plot_bid_ask_bounce(
    efficient: ArrayLike,
    bid: ArrayLike,
    ask: ArrayLike,
    mid: ArrayLike,
    trades: ArrayLike,
    *,
    title: str = "Bid, ask, mid, and bid-ask bounce",
) -> Figure:
    """Show how trades at opposite quote sides create artificial returns."""
    series = {
        "efficient": np.asarray(efficient, dtype=np.float64),
        "bid": np.asarray(bid, dtype=np.float64),
        "ask": np.asarray(ask, dtype=np.float64),
        "mid": np.asarray(mid, dtype=np.float64),
        "trades": np.asarray(trades, dtype=np.float64),
    }
    lengths = {len(values) for values in series.values() if values.ndim == 1}
    if any(values.ndim != 1 for values in series.values()) or lengths == set():
        raise ValueError("all price series must be one-dimensional")
    if len(lengths) != 1 or next(iter(lengths)) < 3:
        raise ValueError("all price series must have the same length of at least three")
    if any(not np.all(np.isfinite(values)) for values in series.values()):
        raise ValueError("all price series must be finite")
    if any(np.any(values <= 0) for values in series.values()):
        raise ValueError("all price series must be strictly positive")
    if np.any(series["ask"] < series["bid"]):
        raise ValueError("ask must not be below bid")
    if np.any(series["mid"] < series["bid"]) or np.any(
        series["mid"] > series["ask"]
    ):
        raise ValueError("mid must lie between bid and ask")

    steps = np.arange(next(iter(lengths)))
    trade_at_ask = np.abs(series["trades"] - series["ask"]) <= np.abs(
        series["trades"] - series["bid"]
    )
    mid_returns = np.diff(np.log(series["mid"]))
    trade_returns = np.diff(np.log(series["trades"]))
    mid_rv = float(np.sum(mid_returns**2))
    trade_rv = float(np.sum(trade_returns**2))

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.3, 1]},
    )
    quote_axis, trade_axis, return_axis = axes

    quote_axis.fill_between(
        steps,
        series["bid"],
        series["ask"],
        color="tab:gray",
        alpha=0.18,
        label="Quoted spread",
    )
    quote_axis.plot(steps, series["ask"], color="tab:green", label="Ask")
    quote_axis.plot(steps, series["bid"], color="tab:red", label="Bid")
    quote_axis.plot(steps, series["mid"], color="tab:blue", linewidth=1.7, label="Mid")
    quote_axis.plot(
        steps,
        series["efficient"],
        color="black",
        linewidth=1.2,
        linestyle="--",
        label="Efficient price",
    )
    quote_axis.set_ylabel("Price")
    quote_axis.set_title("1. Bid and ask surround the mid price")
    quote_axis.grid(alpha=0.25)
    quote_axis.legend(frameon=False, loc="best", ncols=3)

    trade_axis.fill_between(
        steps,
        series["bid"],
        series["ask"],
        color="tab:gray",
        alpha=0.14,
    )
    trade_axis.plot(
        steps,
        series["mid"],
        color="tab:blue",
        linewidth=1.5,
        label="Mid",
    )
    trade_axis.plot(
        steps,
        series["trades"],
        color="0.45",
        linewidth=0.8,
        alpha=0.8,
        label="Trade path",
    )
    trade_axis.scatter(
        steps[~trade_at_ask],
        series["trades"][~trade_at_ask],
        color="tab:red",
        s=20,
        zorder=3,
        label="Trade at bid",
    )
    trade_axis.scatter(
        steps[trade_at_ask],
        series["trades"][trade_at_ask],
        color="tab:green",
        s=20,
        zorder=3,
        label="Trade at ask",
    )
    trade_axis.set_ylabel("Price")
    trade_axis.set_title("2. Transactions bounce between the two quote sides")
    trade_axis.grid(alpha=0.25)
    trade_axis.legend(frameon=False, loc="best", ncols=2)

    return_steps = steps[1:]
    return_axis.axhline(0, color="0.35", linewidth=0.9)
    return_axis.plot(
        return_steps,
        trade_returns,
        color="tab:orange",
        linewidth=1.0,
        label="Trade returns",
    )
    return_axis.plot(
        return_steps,
        mid_returns,
        color="tab:blue",
        linewidth=1.0,
        alpha=0.9,
        label="Mid returns",
    )
    return_axis.text(
        0.98,
        0.05,
        f"Mid RV = {mid_rv:.7f}\nTrade RV = {trade_rv:.7f}\n"
        f"Inflation = {trade_rv / mid_rv:.2f}×",
        transform=return_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    return_axis.set_xlabel("Observation")
    return_axis.set_ylabel("Log return")
    return_axis.set_title("3. Quote-side changes add artificial high-frequency returns")
    return_axis.grid(alpha=0.25)
    return_axis.legend(frameon=False, loc="upper left")

    figure.suptitle(title)
    figure.tight_layout()
    return figure
