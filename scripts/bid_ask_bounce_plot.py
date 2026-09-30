"""Plot bid, ask, mid, and transaction-price bid-ask bounce.

Usage:
    python scripts/bid_ask_bounce_plot.py
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rvol.plotting.microstructure import plot_bid_ask_bounce  # noqa: E402
from rvol.simulation.heston import simulate_heston  # noqa: E402
from rvol.simulation.noise import add_bid_ask_bounce  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/bid_ask_bounce.png")
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--spread-bps", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_steps < 2:
        parser.error("--n-steps must be at least two")
    if args.hours <= 0:
        parser.error("--hours must be positive")
    if args.spread_bps < 0:
        parser.error("--spread-bps must be non-negative")

    dt = args.hours / 24 / 252 / args.n_steps
    efficient, _, _ = simulate_heston(
        n_steps=args.n_steps,
        dt=dt,
        s0=100.0,
        rng=np.random.default_rng(args.seed),
    )
    half_spread = args.spread_bps / 2 / 10_000
    bid = efficient * np.exp(-half_spread)
    ask = efficient * np.exp(half_spread)
    mid = 0.5 * (bid + ask)
    trades = add_bid_ask_bounce(
        efficient,
        half_spread,
        rng=np.random.default_rng(args.seed + 1),
    )

    figure = plot_bid_ask_bounce(
        efficient,
        bid,
        ask,
        mid,
        trades,
        title=f"Bid, ask, mid, and bid-ask bounce (spread={args.spread_bps:g} bps)",
    )
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    mid_rv = float(np.sum(np.diff(np.log(mid)) ** 2))
    trade_rv = float(np.sum(np.diff(np.log(trades)) ** 2))
    print(f"quoted spread: {args.spread_bps:g} bps")
    print(f"mid RV:        {mid_rv:.8f}")
    print(f"trade RV:      {trade_rv:.8f}")
    print(f"RV inflation:  {trade_rv / mid_rv:.2f}x")
    print(f"figure:        {output}")


if __name__ == "__main__":
    main()
