"""Forecast evaluation and model comparison."""

from rvol.forecasting.evaluation.comparison import DieboldMarianoTest
from rvol.forecasting.evaluation.diagnostics import (
    DiagnosticCheck,
    DiagnosticContext,
    DiagnosticResult,
)
from rvol.forecasting.evaluation.evaluator import (
    EvaluationResult,
    ForecastEvaluator,
    LossRecord,
    ModelLossSummary,
    PairwiseComparison,
)
from rvol.forecasting.evaluation.metrics import LogMSEMetric, QLikeMetric
from rvol.forecasting.evaluation.robustness import (
    ForecastRobustnessAnalyzer,
    LagSensitivityCheck,
    LargestErrorsCheck,
    MonthlyLossCheck,
    RobustnessResult,
    WinRateCheck,
)
from rvol.forecasting.evaluation.validation import AggregateLossSummary, ValidationResult

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
