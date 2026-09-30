"""Plot daily realized variance through a multi-day Heston simulation.

Usage:
    python scripts/daily_rv_plot.py
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rvol.plotting.daily import plot_daily_realized_variance  # noqa: E402
from rvol.simulation.heston import simulate_heston  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/daily_rv.png")
    parser.add_argument("--days", type=int, default=252)
    parser.add_argument("--observations-per-day", type=int, default=96)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.days < 2:
        parser.error("--days must be at least two")
    if args.observations_per_day < 2:
        parser.error("--observations-per-day must be at least two")
    if not 2 <= args.rolling_window <= args.days:
        parser.error("--rolling-window must be between two and --days")

    total_steps = args.days * args.observations_per_day
    dt = 1 / 252 / args.observations_per_day
    prices, _, _ = simulate_heston(
        n_steps=total_steps,
        dt=dt,
        s0=100.0,
        rng=np.random.default_rng(args.seed),
    )
    intraday_returns = np.diff(np.log(prices)).reshape(
        args.days,
        args.observations_per_day,
    )
    daily_rv = np.sum(intraday_returns**2, axis=1)
    daily_close = prices[args.observations_per_day :: args.observations_per_day]

    figure = plot_daily_realized_variance(
        daily_close,
        daily_rv,
        rolling_window=args.rolling_window,
        title=f"Daily realized variance over time (seed={args.seed})",
    )
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    highest = int(np.argmax(daily_rv))
    print(f"days:             {args.days}")
    print(f"mean daily RV:    {np.mean(daily_rv):.8f}")
    print(f"highest-RV day:   {highest + 1}")
    print(f"highest daily RV: {daily_rv[highest]:.8f}")
    print(f"figure:           {output}")


if __name__ == "__main__":
    main()
