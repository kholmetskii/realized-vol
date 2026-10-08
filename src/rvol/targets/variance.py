"""Future session and cumulative realized-variance labels."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rvol.domain.specifications import ComponentSpecification
from rvol.domain.targets import TargetHorizon
from rvol.market.daily import ordered_log_variance, validate_daily_columns


@dataclass(frozen=True)
class RealizedVarianceTargetBuilder:
    """Label the full observed timeline before any feature rows are filtered.

    Each origin has its target interval and log variance. Trailing origins
    whose outcomes are not yet complete retain missing end dates and labels.
    The final target date is also the earliest date the label is available.
    """

    horizon: TargetHorizon = field(default_factory=TargetHorizon)
    date_col: str = "date"
    rv_col: str = "log_rv"

    def __post_init__(self) -> None:
        validate_daily_columns(self.date_col, self.rv_col)
        if not isinstance(self.horizon, TargetHorizon):
            raise ValueError("target horizon must be a TargetHorizon")

    @property
    def specification(self) -> ComponentSpecification:
        return ComponentSpecification(
            name="realized_variance",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            parameters=(
                ("date_col", self.date_col), ("rv_col", self.rv_col),
                ("horizon_sessions", self.horizon.sessions), ("target_kind", self.horizon.kind),
            ),
        )

    def __call__(self, daily: pd.DataFrame) -> pd.DataFrame:
        ordered = ordered_log_variance(daily, date_col=self.date_col, rv_col=self.rv_col)
        log_rv = ordered["log_rv"]
        horizon = self.horizon
        target = log_rv.shift(-horizon.sessions)
        if horizon.kind == "cumulative" and 1 < horizon.sessions < len(log_rv):
            # Sum variances stably; any missing future session makes the label unavailable.
            cumulative = log_rv.shift(-1).to_numpy(copy=True)
            with np.errstate(invalid="ignore"):
                for offset in range(2, horizon.sessions + 1):
                    np.logaddexp(cumulative, log_rv.shift(-offset).to_numpy(), out=cumulative)
            target = pd.Series(cumulative, index=log_rv.index)
        result = pd.DataFrame({
            "origin_date": ordered["date"],
            "target_date": ordered["date"].shift(-horizon.sessions),
            "target": target,
            "target_start_date": ordered["date"].shift(
                -1 if horizon.kind == "cumulative" else -horizon.sessions,
            ),
            "horizon_sessions": horizon.sessions,
            "target_kind": horizon.kind,
        })
        result.attrs["target_horizon"] = {"sessions": horizon.sessions, "kind": horizon.kind}
        return result
