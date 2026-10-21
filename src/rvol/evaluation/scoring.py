"""Score forecasts with a loss function and compare models against a candidate."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from rvol.evaluation.diebold_mariano import diebold_mariano
from rvol.models import FloatArray
from rvol.walkforward import CASE_COLUMNS, validated_forecasts

Loss = Callable[[FloatArray, FloatArray], FloatArray]


def forecast_losses(forecasts: pd.DataFrame, loss: Loss) -> pd.DataFrame:
    """One row per forecast with its ``loss``.

    All models must cover one series and the same dates with the same actual
    values, so that their losses can be compared date by date.
    """
    frame = validated_forecasts(forecasts)
    if frame["series"].nunique() != 1:
        raise ValueError("evaluate each series separately")
    aligned = ["origin_date", "target_date", "actual"]
    groups = [group[aligned].reset_index(drop=True) for _, group in frame.groupby("model")]
    if any(not group.equals(groups[0]) for group in groups[1:]):
        raise ValueError("all models must have identical dates and actual values")
    values = np.asarray(
        loss(
            frame["actual"].to_numpy(dtype="float64"), frame["predicted"].to_numpy(dtype="float64")
        ),
        dtype="float64",
    )
    if values.shape != (len(frame),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("the loss must return one finite non-negative value per forecast")
    result = frame[[*CASE_COLUMNS, "actual", "predicted"]].copy()
    result["loss"] = values
    return result


def compare_models(
    losses: pd.DataFrame, candidate: str, *, hac_lags: int | None = None
) -> pd.DataFrame:
    """Mean loss of every model and the one-sided DM test of ``candidate`` against it.

    A small p-value means the candidate's expected loss is lower than that
    model's. The candidate's own row has no test.
    """
    table = losses.pivot(index="target_date", columns="model", values="loss").sort_index()
    if candidate not in table.columns:
        raise ValueError(f"unknown candidate model {candidate!r}")
    proposed = table[candidate].to_numpy(dtype="float64")
    rows = []
    for model in sorted(table.columns):
        baseline = table[model].to_numpy(dtype="float64")
        statistic, p_value = (
            (np.nan, np.nan)
            if model == candidate
            else diebold_mariano(baseline, proposed, hac_lags=hac_lags)
        )
        rows.append(
            {
                "model": model,
                "n_obs": len(baseline),
                "mean_loss": float(np.mean(baseline)),
                "dm_statistic": statistic,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)
