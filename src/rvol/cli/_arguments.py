"""Argument parsers shared by command-line entry points."""

from __future__ import annotations

import argparse
import datetime as dt


def iso_date(value: str) -> dt.date:
    """Parse an ISO calendar date for argparse."""
    try:
        return dt.date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected date in YYYY-MM-DD format") from error
