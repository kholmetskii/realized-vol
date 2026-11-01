"""The forecast target and the feature tables built from the daily data."""

from __future__ import annotations

import pandas as pd

from rvol.data.preparation import (
    TrailingMeanFeatures,
    build_next_observation_targets,
    join_features,
    prepare_forecast_data,
)
from rvol.study.design import MAX_GAP_DAYS

#: The daily column the models forecast, one session ahead.
LOG_RV = "log_rv"

#: HAR (Corsi, 2009) averages log RV over roughly a day, a week and a month of sessions.
HAR_WINDOWS = (1, 5, 22)
LOG_RV_MEANS = TrailingMeanFeatures(LOG_RV, HAR_WINDOWS, max_gap_days=MAX_GAP_DAYS)
HAR_FEATURES = LOG_RV_MEANS.columns

#: SHAR (Patton & Sheppard, 2015) splits HAR's daily term by the sign of each
#: return: today's log semivariances replace today's log RV, and the weekly and
#: monthly means stay. Patton and Sheppard use levels; logs here match HAR.
LOG_RS_POS = TrailingMeanFeatures("log_rs_pos", (1,), max_gap_days=MAX_GAP_DAYS)
LOG_RS_NEG = TrailingMeanFeatures("log_rs_neg", (1,), max_gap_days=MAX_GAP_DAYS)
SHAR_FEATURES = (
    *LOG_RS_POS.columns,
    *LOG_RS_NEG.columns,
    *(LOG_RV_MEANS.column(window) for window in HAR_WINDOWS[1:]),
)

#: Every feature table the study builds; ``prepare`` joins them by origin.
FEATURE_TABLES = (LOG_RV_MEANS, LOG_RS_POS, LOG_RS_NEG)


def prepare(daily: pd.DataFrame) -> pd.DataFrame:
    """Build every feature table and the next-session target from the daily table.

    One table holds all features, so every model forecasts the same dates.
    """
    return prepare_forecast_data(
        join_features([build(daily) for build in FEATURE_TABLES]),
        build_next_observation_targets(daily, value_col=LOG_RV),
    )
