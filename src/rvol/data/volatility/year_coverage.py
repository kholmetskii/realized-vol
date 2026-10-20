"""Per-year session counts reported by the coverage audit."""

from __future__ import annotations


class YearCoverage:
    """Number of unique sessions observed in one calendar year."""

    def __init__(self, year: int, sessions: int) -> None:
        self.year = year
        self.sessions = sessions
