"""Generate a standalone plot of one Heston simulation.

Usage:
    python scripts/heston_plot.py --out figures/heston_simulation.png
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rvol.estimators.realized import realized_variance  # noqa: E402
from rvol.plotting.simulation import plot_heston_path  # noqa: E402
from rvol.simulation.heston import simulate_heston  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/heston_simulation.png")
    parser.add_argument("--n-steps", type=int, default=4_000)
    parser.add_argument("--days", type=float, default=1.0)
    parser.add_argument("--s0", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=0.04)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--theta", type=float, default=0.04)
    parser.add_argument("--xi", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=-0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_steps < 2:
        parser.error("--n-steps must be at least two")
    if args.days <= 0:
        parser.error("--days must be positive")
    if not -1 <= args.rho <= 1:
        parser.error("--rho must be between -1 and 1")

    dt = args.days / 252 / args.n_steps
    prices, variance, integrated_variance = simulate_heston(
        n_steps=args.n_steps,
        dt=dt,
        s0=args.s0,
        v0=args.v0,
        kappa=args.kappa,
        theta=args.theta,
        xi=args.xi,
        rho=args.rho,
        rng=np.random.default_rng(args.seed),
    )
    realized = realized_variance(prices)
    figure = plot_heston_path(
        prices,
        variance,
        dt=dt,
        theta=args.theta,
        title=(
            f"Heston simulation: {args.days:g} trading day(s), "
            f"ρ={args.rho:g}, ξ={args.xi:g}"
        ),
    )

    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    print(f"integrated variance: {integrated_variance:.8f}")
    print(f"realized variance:   {realized:.8f}")
    print(f"RV / IV:             {realized / integrated_variance:.4f}")
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
