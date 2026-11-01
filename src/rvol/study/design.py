"""The study design: every choice fixed before the evaluation period is scored.

Every function in ``rvol.study`` takes these constants as its defaults, so
each asset runs the same study.
"""

from __future__ import annotations

from datetime import date

from rvol.evaluation import qlike

#: The assets the study covers, by the name of their tick-data folder.
ASSETS = ("EURUSD",)

#: Consecutive sessions may be at most this many calendar days apart; a longer
#: gap is a data outage, not a weekend or holiday, and stops feature building.
MAX_GAP_DAYS = 7

#: Every fit uses at least this many observed outcomes, about one year of sessions.
MIN_TRAIN_SESSIONS = 252

#: The first and last target dates forecast. Forecasts before EVALUATION_START
#: form the calibration window: they supply errors for the smearing factor and
#: choose the ridge penalty, and the evaluation never scores them.
FORECAST_START = date(2024, 10, 1)
EVALUATION_START = date(2025, 1, 1)
FORECAST_END = date(2026, 9, 30)

#: The model every other model is tested against.
CANDIDATE = "HAR"

#: The loss every variance forecast is scored with. QLIKE ranks forecasts
#: correctly even though RV only measures the true variance with noise
#: (Patton, 2011).
LOSS = qlike
LOSS_LABEL = LOSS.__name__.upper()

#: Ridge penalties tried for HAR. The penalty multiplies the squared slopes
#: against the summed squared errors, so the same penalty shrinks less as the
#: training sample grows.
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)
