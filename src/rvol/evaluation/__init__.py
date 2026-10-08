"""Forecast evaluation and model comparison."""

from rvol.evaluation.comparison import DieboldMarianoTest
from rvol.evaluation.diagnostics import DiagnosticCheck, DiagnosticContext, DiagnosticResult
from rvol.evaluation.evaluator import (
    EvaluationResult,
    ForecastEvaluator,
    LossRecord,
    ModelLossSummary,
    PairwiseComparison,
)
from rvol.evaluation.metrics import LogMSEMetric, QLikeMetric
from rvol.evaluation.robustness import (
    ForecastRobustnessAnalyzer,
    LagSensitivityCheck,
    LargestErrorsCheck,
    MonthlyLossCheck,
    RobustnessResult,
    WinRateCheck,
)
from rvol.evaluation.validation import AggregateLossSummary, ValidationResult

__all__ = [
    "AggregateLossSummary",
    "DieboldMarianoTest",
    "DiagnosticCheck",
    "DiagnosticContext",
    "DiagnosticResult",
    "EvaluationResult",
    "ForecastEvaluator",
    "ForecastRobustnessAnalyzer",
    "LogMSEMetric",
    "LossRecord",
    "ModelLossSummary",
    "PairwiseComparison",
    "QLikeMetric",
    "LagSensitivityCheck",
    "LargestErrorsCheck",
    "MonthlyLossCheck",
    "RobustnessResult",
    "WinRateCheck",
    "ValidationResult",
]
