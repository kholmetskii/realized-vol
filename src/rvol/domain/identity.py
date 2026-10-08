"""Stable identities for independent experiment runs and validation folds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunIdentity:
    """The pair (run_id, fold_id) identifies one independently scored run.

    The deterministic default names a standalone run. Collections must supply
    distinct run or fold IDs; duplicate identities are rejected during scoring.
    """

    run_id: str = "default"
    fold_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if self.fold_id is not None and (
            not isinstance(self.fold_id, str) or not self.fold_id.strip()
        ):
            raise ValueError("fold_id must be a non-empty string or None")
