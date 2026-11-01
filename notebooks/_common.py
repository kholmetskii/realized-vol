"""What every notebook shares: the study design, the data, HAR, and the forecasting steps.

Each notebook defines the models its own question is about. What must be
identical across notebooks for their numbers to be comparable lives here once:
the dates and sample sizes, the loss, the assets, HAR (the model every other is
tested against) and the steps that turn daily data into scored forecasts.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import _plots
import numpy as np
import pandas as pd

from rvol.data.layout import DataLayout
from rvol.data.preparation import (
    TrailingMeanFeatures,
    build_next_observation_targets,
    join_features,
    prepare_forecast_data,
)
from rvol.data.volatility import SAMPLING_FREQUENCY, load_daily_dataset
from rvol.evaluation import (
    compare_models,
    default_hac_lags,
    forecast_losses,
    qlike,
    win_rates,
)
from rvol.models import ForecastModel, LinearRegression
from rvol.walkforward import smeared_forecasts, walk_forward

# --- Study design: fixed before the evaluation period is scored ---------------

#: The assets the study covers, by the name of their tick-data folder.
ASSETS = ("EURUSD",)

#: Consecutive sessions may be at most this many calendar days apart; a longer
#: gap is a data outage, not a weekend or holiday, and stops feature building.
MAX_GAP_DAYS = 7

#: Every fit uses at least this many observed outcomes, about one year of sessions.
MIN_TRAIN_SESSIONS = 252

#: The first and last target dates forecast. Forecasts before EVALUATION_START
#: form the calibration window: they supply errors for the smearing factor and
#: may tune a model, and the evaluation never scores them.
FORECAST_START = date(2024, 10, 1)
EVALUATION_START = date(2025, 1, 1)
FORECAST_END = date(2026, 9, 30)

#: The loss every variance forecast is scored with. QLIKE ranks forecasts
#: correctly even though RV only measures the true variance with noise
#: (Patton, 2011).
LOSS = qlike
LOSS_LABEL = "QLIKE"

#: The model every other model is tested against.
CANDIDATE = "HAR"

# --- The target and HAR, shared by every model notebook -----------------------

#: The daily column every model forecasts, one session ahead.
LOG_RV = "log_rv"

#: HAR (Corsi, 2009) averages log RV over roughly a day, a week and a month of sessions.
HAR_WINDOWS = (1, 5, 22)
LOG_RV_MEANS = TrailingMeanFeatures(LOG_RV, HAR_WINDOWS, max_gap_days=MAX_GAP_DAYS)
HAR_FEATURES = LOG_RV_MEANS.columns
HAR = ForecastModel(CANDIDATE, LinearRegression(), HAR_FEATURES)

# --- Display --------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
LAYOUT = DataLayout(ROOT / "data")

#: Every model name in the notebooks, so a model keeps its colour everywhere.
COLORS = _plots.model_colors(
    [CANDIDATE, "historical_mean", "naive", "EWMA", "AR1", "SHAR", "HAR-ridge"]
)
MAIN_COLOR = COLORS[CANDIDATE]


def pair(asset: str) -> str:
    """Display name of a currency pair: EURUSD -> EUR/USD."""
    return f"{asset[:3]}/{asset[3:]}" if len(asset) == 6 else asset


def design() -> pd.DataFrame:
    """The study design as a one-column table, for the top of each notebook."""
    return pd.Series(
        {
            "assets": ", ".join(pair(asset) for asset in ASSETS),
            "minimum training sessions": MIN_TRAIN_SESSIONS,
            "first forecast target": FORECAST_START,
            "evaluation start": EVALUATION_START,
            "last forecast target": FORECAST_END,
            "candidate model": CANDIDATE,
            "loss": LOSS_LABEL,
        },
        name="study design",
    ).to_frame()


def by_asset(frames: dict[str, pd.DataFrame], name: str = "asset") -> pd.DataFrame:
    """Stack per-asset tables under an asset level; with one asset, the table itself."""
    if len(frames) == 1:
        return next(iter(frames.values()))
    return pd.concat({pair(asset): frame for asset, frame in frames.items()}, names=[name])


def sessions_per_year(table: pd.DataFrame) -> float:
    """Sessions per calendar year, counted from the data rather than assumed."""
    years = (table["date"].iloc[-1] - table["date"].iloc[0]).days / 365.25
    return (len(table) - 1) / years


def annualized_vol(variance, per_year: float) -> np.ndarray:
    """Daily variance -> annualized volatility in percent, the unit traders quote."""
    return 100 * np.sqrt(per_year * np.asarray(variance, dtype="float64"))


def specification(model: ForecastModel) -> dict:
    """What defines a model before fitting: its rule, settings and regressors."""
    rule = model.rule
    settings = {k: getattr(rule, k) for k in inspect.signature(type(rule)).parameters}
    return {
        "rule": type(rule).__name__,
        "settings": ", ".join(f"{k} = {v}" for k, v in settings.items()),
        "regressors": ", ".join(model.regressors) or "none",
    }


def model_table(models: Sequence[ForecastModel]) -> pd.DataFrame:
    """One row per model: its rule, settings and regressors."""
    return pd.DataFrame(
        [specification(model) for model in models],
        index=pd.Index([model.name for model in models], name="model"),
    )


# --- Data -------------------------------------------------------------------------


def tick_files(asset: str) -> list[Path]:
    return sorted(LAYOUT.ticks(asset).glob("*.parquet"))


@functools.cache
def daily(asset: str) -> pd.DataFrame:
    """The asset's daily table, built from its ticks once and cached on disk."""
    files = tick_files(asset)
    if not files:
        raise FileNotFoundError(f"no tick files in {LAYOUT.ticks(asset)}")
    cache = LAYOUT.daily_rv(asset, SAMPLING_FREQUENCY)
    return load_daily_dataset(files, cache, frequency=SAMPLING_FREQUENCY)


# --- Forecasting steps ------------------------------------------------------------


def prepare(table: pd.DataFrame, features: Sequence[TrailingMeanFeatures] = ()) -> pd.DataFrame:
    """HAR's features, any extra feature tables, and the next-session target.

    One table holds all features, so every model forecasts the same dates.
    """
    tables = [LOG_RV_MEANS, *features]
    return prepare_forecast_data(
        join_features([build(table) for build in tables]),
        build_next_observation_targets(table, value_col=LOG_RV),
    )


def forecast(
    prepared: pd.DataFrame,
    models: Sequence[ForecastModel],
    *,
    asset: str,
    start: date = FORECAST_START,
    end: date = FORECAST_END,
) -> pd.DataFrame:
    """Walk-forward forecasts of next-session log RV from every model, stacked.

    Each model is refitted every session on every outcome observed so far.
    """
    runs = [
        walk_forward(
            prepared,
            model,
            min_train_size=MIN_TRAIN_SESSIONS,
            start=start,
            end=end,
            series=asset,
        )
        for model in models
    ]
    return pd.concat(runs, ignore_index=True)


@dataclass(frozen=True)
class Comparison:
    """Variance-scale forecasts on the evaluation period, their losses, and the
    one-sided DM test of the candidate against every other model."""

    forecasts: pd.DataFrame
    losses: pd.DataFrame
    summary: pd.DataFrame


def compare(log_forecasts: pd.DataFrame) -> Comparison:
    """Smear log forecasts back to variance and score the evaluation period."""
    smeared = smeared_forecasts(log_forecasts)
    window = (
        smeared.loc[smeared["target_date"] >= pd.Timestamp(EVALUATION_START)]
        .sort_values(["model", "target_date"])
        .reset_index(drop=True)
    )
    losses = forecast_losses(window, LOSS)
    return Comparison(window, losses, compare_models(losses, CANDIDATE))


def results_table(comparison: Comparison) -> pd.DataFrame:
    """One row per model, best first: mean loss, the candidate's advantage,
    the DM p-value and the candidate's win rate."""
    summary = comparison.summary.set_index("model")
    wins = win_rates(comparison.losses, CANDIDATE).set_index("baseline_model")
    table = pd.DataFrame(
        {
            LOSS_LABEL: summary["mean_loss"],
            f"{CANDIDATE} lower by": 1 - summary.loc[CANDIDATE, "mean_loss"] / summary["mean_loss"],
            "DM p-value": summary["p_value"],
            f"{CANDIDATE} wins": wins["candidate_win_rate"],
        }
    )
    table.index.name = "model"
    return table.sort_values(LOSS_LABEL)


@dataclass(frozen=True)
class Run:
    """One asset's daily data, prepared table, log forecasts and comparison."""

    asset: str
    daily: pd.DataFrame
    prepared: pd.DataFrame
    forecasts: pd.DataFrame
    comparison: Comparison

    def losses(self) -> pd.DataFrame:
        """Daily losses, one column per model."""
        return self.comparison.losses.pivot(index="target_date", columns="model", values="loss")


def run_asset(
    asset: str,
    models: Sequence[ForecastModel],
    features: Sequence[TrailingMeanFeatures] = (),
) -> Run:
    """Forecast and score ``models`` on one asset; the candidate must be among them."""
    if CANDIDATE not in {model.name for model in models}:
        raise ValueError(f"the candidate {CANDIDATE!r} must be among the models")
    table = daily(asset)
    prepared = prepare(table, features)
    forecasts = forecast(prepared, models, asset=asset)
    return Run(asset, table, prepared, forecasts, compare(forecasts))


def run_models(
    models: Sequence[ForecastModel], features: Sequence[TrailingMeanFeatures] = ()
) -> dict[str, Run]:
    """The same models on every asset."""
    return {asset: run_asset(asset, models, features) for asset in ASSETS}


# --- Helpers several notebooks use ------------------------------------------------


def hac_lag_grid(n_obs: int) -> list[int]:
    """Lags from none up to six times the usual rule for ``n_obs`` sessions."""
    usual = default_hac_lags(n_obs)
    return sorted({*range(usual + 1), 2 * usual, 4 * usual, 6 * usual})


def coefficients(run: Run, model: ForecastModel, before: date | pd.Timestamp) -> pd.Series:
    """The model's intercept and slopes fitted on every outcome observed before ``before``."""
    known = run.prepared.loc[run.prepared["target_date"] < pd.Timestamp(before)]
    fitted = model.rule.fit(known[list(model.regressors)].to_numpy(), known["target"].to_numpy())
    return pd.Series(
        [fitted.intercept, *fitted.coefficients], index=["intercept", *model.regressors]
    )
