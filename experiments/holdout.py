"""EUR/USD forecasts fitted once using outcomes through 31 December 2024.

Run from the repository root with ``python -m experiments.holdout DAILY``.
"""

from dataclasses import replace
from datetime import date

from experiments.eurusd import build_experiment as build_baseline
from rvol.cli.run_experiment import main
from rvol.domain import HoldoutSettings, RunIdentity
from rvol.forecasting.composition import Experiment


def build_experiment() -> Experiment:
    """Keep baseline components and reporting dates with a fixed training cutoff."""
    baseline = build_baseline()
    config = replace(
        baseline.config,
        execution=HoldoutSettings(
            train_end=date(2024, 12, 31),
            min_train_size=252,
            update_observations=False,
        ),
        identity=RunIdentity("eurusd_holdout"),
    )
    return replace(baseline, config=config)


if __name__ == "__main__":
    main(build_experiment())
