"""Reproducibility metadata accompanying a saved forecast figure."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping
from dataclasses import asdict

from rvol.domain import ExperimentResult
from rvol.reporting._files import atomic_text
from rvol.reporting._metadata import configuration_payload
from rvol.reporting.artifacts import DatasetSnapshot


def write_forecast_plot_metadata(
    figure_path: str | pathlib.Path, experiment: ExperimentResult, *,
    dataset: DatasetSnapshot, configuration: Mapping[str, object],
) -> pathlib.Path:
    """Write a JSON sidecar next to the image with resolved settings and input identity."""
    target_dates = sorted({record.target_date for record in experiment.records})
    if not target_dates:
        raise ValueError("plot metadata requires forecast records")
    path = pathlib.Path(str(figure_path) + ".json")
    payload = {
        "schema_version": 1,
        "resolved_configuration": configuration_payload(configuration),
        "dataset": asdict(dataset) | {
            "start_date": dataset.start_date.isoformat(), "end_date": dataset.end_date.isoformat(),
            "instrument": None if experiment.config is None else experiment.config.instrument,
        },
        "identity": asdict(experiment.identity),
        "target": asdict(experiment.target_horizon),
        "forecast_start": target_dates[0].isoformat(),
        "forecast_end": target_dates[-1].isoformat(),
        "forecast_count": experiment.n_forecasts,
    }
    atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)
    return path
