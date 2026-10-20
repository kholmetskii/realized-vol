"""A calendar gap found by the coverage audit."""

from __future__ import annotations

import pandas as pd


class SessionGap:
    """A calendar gap between consecutive observed trading sessions."""

    def __init__(
        self,
        previous_date: pd.Timestamp,
        next_date: pd.Timestamp,
        calendar_days: int,
    ) -> None:
        self.previous_date = previous_date
        self.next_date = next_date
        self.calendar_days = calendar_days
