"""Robustness diagnostics computed from already evaluated forecasts.

Each diagnostic is a class implementing Diagnostic; analyze_robustness runs a
sequence of them. A new diagnostic is a new class, not a change to this package.
"""

from rvol.evaluation.robustness.analysis import analyze_robustness, standard_diagnostics
from rvol.evaluation.robustness.diagnostic import Diagnostic
from rvol.evaluation.robustness.hac_sensitivity import HacSensitivity
from rvol.evaluation.robustness.largest_errors import LargestErrors
from rvol.evaluation.robustness.monthly_losses import MonthlyLosses
from rvol.evaluation.robustness.win_rates import WinRates

__all__ = [
    "Diagnostic",
    "HacSensitivity",
    "LargestErrors",
    "MonthlyLosses",
    "WinRates",
    "analyze_robustness",
    "standard_diagnostics",
]
