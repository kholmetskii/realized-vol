"""The interface every robustness diagnostic implements."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from rvol.evaluation import EvaluationResult


class Diagnostic(Protocol):
    """Turn validated forecasts and their existing losses into one plain table.

    Diagnostics never fit models or recompute losses. ``forecasts`` has already
    been checked against ``evaluation`` by analyze_robustness().
    """

    @property
    def name(self) -> str: ...

    def run(self, forecasts: pd.DataFrame, evaluation: EvaluationResult) -> pd.DataFrame: ...
