"""Expanding-window EUR/USD forecasts with daily refitting.

Run from the repository root with ``python -m experiments.eurusd DAILY``.
Edit the typed settings here for a different research specification.
"""

from datetime import date

from rvol.cli.run_experiment import main
from rvol.domain import ExperimentConfig, RunIdentity, WalkForwardSettings
from rvol.forecasting.composition import Experiment, standard_experiment


def build_experiment() -> Experiment:
    """Return fresh components for the baseline EUR/USD experiment."""
    return standard_experiment(ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=252),
        instrument="EUR/USD",
        identity=RunIdentity("eurusd_baseline"),
        forecast_start=date(2025, 1, 1),
        forecast_end=date(2026, 9, 30),
    ))


if __name__ == "__main__":
    main(build_experiment())
