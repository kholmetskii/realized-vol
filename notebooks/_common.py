"""What every notebook needs: the assets, their data and study runs, and display helpers."""

from __future__ import annotations

import functools
from pathlib import Path

import _plots
import numpy as np
import pandas as pd

from rvol import study
from rvol.data.layout import DataLayout
from rvol.data.volatility import SAMPLING_FREQUENCY, load_daily_dataset
from rvol.study import StudyRun

ROOT = Path(__file__).resolve().parent.parent
LAYOUT = DataLayout(ROOT / "data")
ASSETS = study.ASSETS
CANDIDATE = study.CANDIDATE
LOSS = study.LOSS_LABEL

#: Every model of the study, candidate first, in the order the study lists them,
#: so a model keeps its colour in every notebook.
MODELS = [
    CANDIDATE,
    *(
        model.name
        for model in (*study.benchmark_models(), study.ridge_har(study.RIDGE_GRID[0]))
        if model.name != CANDIDATE
    ),
]
COLORS = _plots.model_colors(MODELS)
MAIN_COLOR = COLORS[CANDIDATE]


def pair(asset: str) -> str:
    """Display name of a currency pair: EURUSD -> EUR/USD."""
    return f"{asset[:3]}/{asset[3:]}" if len(asset) == 6 else asset


def tick_files(asset: str) -> list[Path]:
    return sorted(LAYOUT.ticks(asset).glob("*.parquet"))


@functools.cache
def daily(asset: str) -> pd.DataFrame:
    """The asset's daily table, built from its ticks once and cached on disk."""
    return load_daily_dataset(
        tick_files(asset), LAYOUT.daily_rv(asset, SAMPLING_FREQUENCY), frequency=SAMPLING_FREQUENCY
    )


@functools.cache
def runs() -> dict[str, StudyRun]:
    """Every asset's study run, keyed by asset."""
    return {asset: study.study_run(daily(asset), asset=asset) for asset in ASSETS}


def sessions_per_year(table: pd.DataFrame) -> float:
    """Sessions per calendar year, counted from the data rather than assumed."""
    years = (table["date"].iloc[-1] - table["date"].iloc[0]).days / 365.25
    return (len(table) - 1) / years


def annualized_vol(variance, per_year: float) -> np.ndarray:
    """Daily variance -> annualized volatility in percent, the unit traders quote."""
    return 100 * np.sqrt(per_year * np.asarray(variance, dtype="float64"))


def by_asset(frames: dict[str, pd.DataFrame | pd.Series], name: str = "asset") -> pd.DataFrame:
    """Stack per-asset tables under an asset level, labelled by display name."""
    return pd.concat({pair(asset): frame for asset, frame in frames.items()}, names=[name])
