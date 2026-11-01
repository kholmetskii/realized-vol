"""One asset's whole study, from its tick files to the scored comparison."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rvol.data.layout import DataLayout
from rvol.data.volatility import SAMPLING_FREQUENCY, load_daily_dataset
from rvol.models import ForecastModel
from rvol.study.features import prepare
from rvol.study.models import calibrated_models, ridge_calibration
from rvol.study.pipeline import Comparison, compare, forecast


@dataclass(frozen=True)
class StudyRun:
    """One asset's study: its data, the models with the chosen ridge penalty,
    their log forecasts and the scored comparison."""

    asset: str
    daily: pd.DataFrame
    prepared: pd.DataFrame
    calibration: pd.Series
    models: tuple[ForecastModel, ...]
    forecasts: pd.DataFrame
    comparison: Comparison


def study_run(daily: pd.DataFrame, *, asset: str) -> StudyRun:
    """Run the study on one asset's daily data.

    The study is deterministic and takes well under a second per asset, so it is
    recomputed rather than cached: a run always reflects the current code.
    """
    prepared = prepare(daily)
    calibration = ridge_calibration(prepared)
    models = calibrated_models(calibration)
    forecasts = forecast(prepared, models, asset=asset)
    return StudyRun(asset, daily, prepared, calibration, models, forecasts, compare(forecasts))


def run_study(asset: str, layout: DataLayout | None = None) -> StudyRun:
    """Load an asset's daily data from its tick files, then run the study on it."""
    paths = layout or DataLayout()
    ticks = sorted(paths.ticks(asset).glob("*.parquet"))
    if not ticks:
        raise FileNotFoundError(f"no tick files in {paths.ticks(asset)}")
    daily = load_daily_dataset(ticks, paths.daily_rv(asset, SAMPLING_FREQUENCY))
    return study_run(daily, asset=asset)
