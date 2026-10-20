"""Evaluate forecasts and compare models: losses, metrics and pairwise tests.

Robustness diagnostics live in rvol.evaluation.robustness and volatility-specific
metrics in rvol.evaluation.volatility.
"""

from rvol.evaluation.comparison_outcome import ComparisonOutcome
from rvol.evaluation.diebold_mariano_test import DieboldMarianoTest, diebold_mariano
from rvol.evaluation.evaluation_result import EvaluationResult
from rvol.evaluation.evaluator import evaluate, validate_evaluation
from rvol.evaluation.forecast_comparison_test import ForecastComparisonTest
from rvol.evaluation.forecast_metric import ForecastMetric
from rvol.evaluation.mae_metric import MAEMetric, absolute_error
from rvol.evaluation.mse_metric import MSEMetric, squared_error

__all__ = [
    "ComparisonOutcome",
    "DieboldMarianoTest",
    "EvaluationResult",
    "ForecastComparisonTest",
    "ForecastMetric",
    "MAEMetric",
    "MSEMetric",
    "absolute_error",
    "diebold_mariano",
    "evaluate",
    "squared_error",
    "validate_evaluation",
]
