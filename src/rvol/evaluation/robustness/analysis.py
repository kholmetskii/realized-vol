"""Run robustness diagnostics on an existing evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from rvol.evaluation import EvaluationResult, validate_evaluation
from rvol.evaluation.robustness.diagnostic import Diagnostic
from rvol.evaluation.robustness.hac_sensitivity import HacSensitivity
from rvol.evaluation.robustness.largest_errors import LargestErrors
from rvol.evaluation.robustness.monthly_losses import MonthlyLosses
from rvol.evaluation.robustness.win_rates import WinRates


def standard_diagnostics(candidate: str | None = None) -> tuple[Diagnostic, ...]:
    """The usual set: descriptive tables, plus pairwise checks when a candidate is given."""
    descriptive: tuple[Diagnostic, ...] = (MonthlyLosses(), LargestErrors())
    if candidate is None:
        return descriptive
    return (HacSensitivity(candidate), WinRates(candidate), *descriptive)


def analyze_robustness(
    forecasts: pd.DataFrame,
    evaluation: EvaluationResult,
    diagnostics: Sequence[Diagnostic] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return one table per diagnostic, keyed by its name, with the asset attached.

    The evaluation must have been computed from exactly these forecasts. Without
    ``diagnostics``, only the descriptive standard diagnostics run.
    """
    frame = validate_evaluation(forecasts, evaluation)
    selected = standard_diagnostics() if diagnostics is None else tuple(diagnostics)
    names = [diagnostic.name for diagnostic in selected]
    if not names or len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise ValueError("diagnostics must have non-empty unique names")
    metadata = frame[["asset"]].iloc[0].to_dict()
    result = {}
    for diagnostic in selected:
        table = diagnostic.run(frame.copy(), evaluation)
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"diagnostic {diagnostic.name!r} must return a DataFrame")
        table = table.copy()
        for key, value in metadata.items():
            table[key] = value
        result[diagnostic.name] = table
    return result
