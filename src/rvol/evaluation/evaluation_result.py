"""The tables returned by evaluate()."""

from __future__ import annotations

import pandas as pd


class EvaluationResult:
    """Three ordinary tables: individual losses, means and requested comparisons."""

    def __init__(
        self,
        losses: pd.DataFrame,
        summaries: pd.DataFrame,
        comparisons: pd.DataFrame,
    ) -> None:
        self.losses = losses
        self.summaries = summaries
        self.comparisons = comparisons
