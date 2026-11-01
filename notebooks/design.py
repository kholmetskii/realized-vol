"""The study's fixed choices, shared by every notebook so their numbers are comparable.

Each notebook defines the models its own question is about. What all of them
must share lives here: the assets, the dates and loss of the comparison, and
HAR, the model every other is tested against.
"""

from datetime import date
from pathlib import Path

from rvol.data.layout import DataLayout
from rvol.data.preparation import TrailingMeanFeatures
from rvol.evaluation import StudyDesign, qlike
from rvol.models import ForecastModel, LinearRegression

#: Where the tick files and the cached daily tables live.
LAYOUT = DataLayout(Path(__file__).resolve().parent.parent / "data")

#: The assets the study covers, by the name of their tick-data folder.
ASSETS = ("EURUSD",)

#: Forecasts run from October 2024 to September 2026, every fit on at least a
#: year of sessions. October to December 2024 is the calibration window: it
#: supplies errors for the smearing factor and may tune a model, and is never
#: scored. QLIKE ranks forecasts correctly even though RV only measures the true
#: variance with noise (Patton, 2011).
DESIGN = StudyDesign(
    forecast_start=date(2024, 10, 1),
    evaluation_start=date(2025, 1, 1),
    forecast_end=date(2026, 9, 30),
    min_train_size=252,
    candidate="HAR",
    loss=qlike,
    loss_name="QLIKE",
)

#: Consecutive sessions may be at most this many calendar days apart; a longer
#: gap is a data outage, not a weekend or holiday, and stops feature building.
MAX_GAP_DAYS = 7

#: The daily column every model forecasts, one session ahead.
LOG_RV = "log_rv"

#: HAR (Corsi, 2009) averages log RV over roughly a day, a week and a month of sessions.
HAR_WINDOWS = (1, 5, 22)
LOG_RV_MEANS = TrailingMeanFeatures(LOG_RV, HAR_WINDOWS, max_gap_days=MAX_GAP_DAYS)
HAR_FEATURES = LOG_RV_MEANS.columns
HAR = ForecastModel(DESIGN.candidate, LinearRegression(), HAR_FEATURES)
