"""Convert log-RV forecasts to the variance scale with Duan smearing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rvol.walkforward import validated_forecasts


def variance_forecasts(forecasts: pd.DataFrame, *, min_errors: int = 20) -> pd.DataFrame:
    """Convert log-RV forecasts to variance forecasts with Duan smearing.

    exp(predicted) estimates the conditional median of RV, not its mean, so it
    is biased low. Each forecast is multiplied by the mean of exp(actual -
    predicted) over that model's earlier forecasts whose target_date is on or
    before the origin_date, so only errors observable at the origin are used.

    Rows with fewer than ``min_errors`` observable errors are dropped. The
    returned table keeps the forecast columns, now on the variance scale, and
    adds the applied ``smearing_factor``.
    """
    if isinstance(min_errors, bool) or not isinstance(min_errors, int) or min_errors < 1:
        raise ValueError("min_errors must be a positive integer")
    frame = validated_forecasts(forecasts)
    converted = []
    for _, group in frame.groupby(["asset", "model"], sort=False):
        ordered = group.sort_values("target_date")
        exp_errors = np.exp(ordered["actual"] - ordered["predicted"]).to_numpy(dtype="float64")
        observable = np.searchsorted(
            ordered["target_date"].to_numpy(),
            ordered["origin_date"].to_numpy(),
            side="right",
        )
        totals = np.concatenate([[0.0], np.cumsum(exp_errors)])
        factor = totals[observable] / np.maximum(observable, 1)
        keep = observable >= min_errors
        result = ordered.loc[keep].copy()
        result["smearing_factor"] = factor[keep]
        result["actual"] = np.exp(result["actual"])
        result["predicted"] = np.exp(result["predicted"]) * result["smearing_factor"]
        converted.append(result)
    combined = pd.concat(converted, ignore_index=True)
    if combined.empty:
        raise ValueError("no forecast has enough observable errors for smearing")
    return combined.sort_values(["model", "target_date"]).reset_index(drop=True)
