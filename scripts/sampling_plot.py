"""Illustrate sampling-grid dependence and subsampled realized variance.

Usage:
    python scripts/sampling_plot.py --step 5 --out figures/sampling_grids.png
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rvol.plotting.sampling import plot_sampling_grids  # noqa: E402


def illustrative_path(n_observations: int, seed: int) -> np.ndarray:
    """Create a reproducible path with ordinary moves and two visible shocks."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 8e-4, n_observations - 1)
    returns[n_observations // 3] += 0.012
    returns[n_observations // 3 + 1] -= 0.009
    returns[2 * n_observations // 3] -= 0.010
    return 100 * np.exp(np.r_[0.0, np.cumsum(returns)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/sampling_grids.png")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--observations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.step < 2:
        parser.error("--step must be at least two")
    if args.observations < 2 * args.step:
        parser.error("--observations must be at least twice --step")

    prices = illustrative_path(args.observations, args.seed)
    figure = plot_sampling_grids(
        prices,
        step=args.step,
        title=f"Sampling-grid sensitivity (step={args.step})",
    )

    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
