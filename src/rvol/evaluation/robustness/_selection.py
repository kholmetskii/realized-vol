"""Candidate/baseline selection and loss tables shared by pairwise diagnostics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pandas as pd


def checked_selection(
    candidate: str, baselines: Sequence[str] | None
) -> tuple[str, ...] | None:
    """Validate the settings a diagnostic receives before any data is seen."""
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("candidate must be a non-empty model name")
    if baselines is None:
        return None
    selected = tuple(baselines)
    if (
        not selected
        or isinstance(baselines, str)
        or len(set(selected)) != len(selected)
        or candidate in selected
    ):
        raise ValueError("baselines must be unique models other than the candidate")
    return selected


def resolved_baselines(
    forecasts: pd.DataFrame, candidate: str, baselines: tuple[str, ...] | None
) -> tuple[str, ...]:
    """Default to every other model; require all named models to be present."""
    models = set(forecasts["model"])
    if candidate not in models:
        raise ValueError(f"unknown candidate model {candidate!r}")
    selected = tuple(sorted(models - {candidate})) if baselines is None else baselines
    if not selected:
        raise ValueError("comparison diagnostics require at least one baseline model")
    missing = set(selected).difference(models)
    if missing:
        raise ValueError(f"unknown baseline models: {', '.join(sorted(missing))}")
    return selected


def losses_by_metric(losses: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """One target_date x model table of losses per metric, in metric order."""
    for metric, group in losses.groupby("metric", sort=True):
        yield str(metric), group.pivot(
            index="target_date", columns="model", values="loss"
        ).sort_index()
