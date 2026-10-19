"""Evaluate forecasts, compare models and run robustness diagnostics."""

from rvol.evaluation.comparison import (
    DieboldMarianoTest,
    diebold_mariano,
)
from rvol.evaluation.contracts import ComparisonOutcome
from rvol.evaluation.evaluator import EvaluationResult, evaluate
from rvol.evaluation.metrics import MAEMetric, MSEMetric
from rvol.evaluation.robustness import analyze_robustness, cumulative_loss_advantage

__all__ = [
    "ComparisonOutcome",
    "DieboldMarianoTest",
    "EvaluationResult",
    "MAEMetric",
    "MSEMetric",
    "analyze_robustness",
    "cumulative_loss_advantage",
    "diebold_mariano",
    "evaluate",
]
