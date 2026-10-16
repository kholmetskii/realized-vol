"""Standard application assembly for reproducible forecast workflows."""

from rvol.forecasting.composition.configuration import experiment_from_mapping
from rvol.forecasting.composition.forecasting import (
    ROBUSTNESS_CHECK_NAMES,
    Experiment,
    standard_diagnostic_checks,
    standard_experiment,
    standard_forecasters,
)

__all__ = [
    "ROBUSTNESS_CHECK_NAMES",
    "Experiment",
    "experiment_from_mapping",
    "standard_diagnostic_checks",
    "standard_experiment",
    "standard_forecasters",
]
