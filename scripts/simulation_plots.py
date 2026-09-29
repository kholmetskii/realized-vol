"""Plot a Heston path, observation noise, and Monte Carlo estimator validation.

Usage:
    python scripts/simulation_plots.py --out figures/simulation_overview.png
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rvol.estimators.realized import (  # noqa: E402
    realized_variance,
    subsampled_variance,
)
from rvol.plotting.simulation import plot_simulation_overview  # noqa: E402
from rvol.simulation.heston import simulate_heston  # noqa: E402
from rvol.simulation.monte_carlo import evaluate_estimators  # noqa: E402
from rvol.simulation.noise import add_iid_log_noise  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/simulation_overview.png")
    parser.add_argument("--n-steps", type=int, default=4_000)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--noise-sd", type=float, default=1e-4)
    parser.add_argument("--subsample-step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    if args.n_steps < 2 * args.subsample_step:
        parser.error("--n-steps must be at least twice --subsample-step")

    dt = 1 / 252 / args.n_steps
    estimators = {
        "Naive RV": realized_variance,
        "Subsampled RV": lambda prices: subsampled_variance(
            prices,
            step=args.subsample_step,
        ),
    }

    def clean_path(rng: np.random.Generator) -> tuple[np.ndarray, float]:
        prices, _, target = simulate_heston(
            args.n_steps,
            dt,
            xi=0.0,
            rng=rng,
        )
        return prices, target

    def noisy_path(rng: np.random.Generator) -> tuple[np.ndarray, float]:
        prices, _, target = simulate_heston(
            args.n_steps,
            dt,
            xi=0.0,
            rng=rng,
        )
        observed = add_iid_log_noise(prices, args.noise_sd, rng)
        return observed, target

    scenarios = {
        "Clean": evaluate_estimators(
            clean_path,
            estimators,
            n_replications=args.replications,
            seed=args.seed,
        ),
        "Noisy": evaluate_estimators(
            noisy_path,
            estimators,
            n_replications=args.replications,
            seed=args.seed + 1,
        ),
    }

    example_rng = np.random.default_rng(args.seed)
    efficient, variance, _ = simulate_heston(
        args.n_steps,
        dt,
        rng=example_rng,
    )
    observed = add_iid_log_noise(efficient, args.noise_sd, example_rng)
    figure = plot_simulation_overview(
        efficient,
        observed,
        variance,
        scenarios,
        title=(
            f"Heston simulation and estimator validation "
            f"({args.replications} replications)"
        ),
    )

    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    for scenario, summaries in scenarios.items():
        print(f"--- {scenario}")
        for summary in summaries:
            print(
                f"{summary.estimator}: relative bias={summary.relative_bias:.2%}, "
                f"relative RMSE={summary.rmse / summary.mean_target:.2%}"
            )
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
