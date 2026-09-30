"""Plot the full transformation from prices to realized variance.

Usage:
    python scripts/returns_and_rv_plot.py --out figures/returns_and_rv.png
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rvol.estimators.realized import realized_variance  # noqa: E402
from rvol.plotting.returns import plot_returns_and_rv  # noqa: E402
from rvol.simulation.heston import simulate_heston  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/returns_and_rv.png")
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--days", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_steps < 2:
        parser.error("--n-steps must be at least two")
    if args.days <= 0:
        parser.error("--days must be positive")

    dt = args.days / 252 / args.n_steps
    prices, _, _ = simulate_heston(
        n_steps=args.n_steps,
        dt=dt,
        s0=100.0,
        rng=np.random.default_rng(args.seed),
    )
    rv = realized_variance(prices)
    figure = plot_returns_and_rv(
        prices,
        title=f"Price → log returns → squared returns → RV (seed={args.seed})",
    )

    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    print(f"prices:      {len(prices)}")
    print(f"log returns: {len(prices) - 1}")
    print(f"final RV:    {rv:.8f}")
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
