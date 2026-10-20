"""Next-observation targets for one-step-ahead forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ordered_values(data: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    if (
        any(not isinstance(name, str) or not name.strip() for name in (date_col, value_col))
        or date_col == value_col
    ):
        raise ValueError("date and value columns must have distinct non-empty names")
    if not data.columns.is_unique:
        raise ValueError("input columns must be unique")
    missing = {date_col, value_col}.difference(data.columns)
    if missing:
        raise ValueError(f"missing input columns: {', '.join(sorted(missing))}")
    result = (
        pd.DataFrame(
            {
                "date": pd.to_datetime(data[date_col], errors="raise"),
                "value": pd.to_numeric(data[value_col], errors="raise"),
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    if result["date"].isna().any() or result["date"].duplicated().any():
        raise ValueError("input must contain one valid observation per date")
    if not np.isfinite(result["value"].to_numpy(dtype="float64", na_value=np.nan)).all():
        raise ValueError("observations must contain only finite values")
    return result


def build_next_observation_targets(
    data: pd.DataFrame,
    *,
    date_col: str = "date",
    value_col: str = "value",
) -> pd.DataFrame:
    """Pair each origin with the next observed date and unchanged value.

    Keep the full timeline before feature warm-up or filtering. The final origin
    retains a missing target date and value. Targets become available for training
    only at their target dates.
    """
    ordered = _ordered_values(data, date_col, value_col)
    return pd.DataFrame(
        {
            "origin_date": ordered["date"],
            "target_date": ordered["date"].shift(-1),
            "target": ordered["value"].shift(-1),
        }
    )
