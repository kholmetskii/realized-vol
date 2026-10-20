"""Evaluate forecasts and compare models: losses, metrics and pairwise tests.

The package scores whatever forecasts it receives and knows nothing about what
they forecast.
"""

from rvol.evaluation.comparison_outcome import ComparisonOutcome
from rvol.evaluation.diebold_mariano_test import DieboldMarianoTest, diebold_mariano
from rvol.evaluation.evaluate import evaluate
from rvol.evaluation.evaluation_result import EvaluationResult
from rvol.evaluation.forecast_comparison_test import ForecastComparisonTest
from rvol.evaluation.forecast_metric import ForecastMetric
from rvol.evaluation.qlike_metric import QLikeMetric
from rvol.evaluation.robustness import hac_sensitivity, win_rates

__all__ = [
    "ComparisonOutcome",
    "DieboldMarianoTest",
    "EvaluationResult",
    "ForecastComparisonTest",
    "ForecastMetric",
    "QLikeMetric",
    "diebold_mariano",
    "evaluate",
    "hac_sensitivity",
    "win_rates",
]
