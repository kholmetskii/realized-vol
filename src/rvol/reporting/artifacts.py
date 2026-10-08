"""Deterministic CSV and JSON artifacts for forecast experiments."""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd

from rvol.domain import (
    ComponentSpecification,
    DescribedExecutionSettings,
    ExperimentConfig,
    ExperimentResult,
)
from rvol.domain.specifications import ExperimentSpecification
from rvol.evaluation import EvaluationResult
from rvol.reporting._files import atomic_csv, atomic_text
from rvol.reporting._metadata import configuration_payload


@dataclass(frozen=True)
class DatasetSnapshot:
    """Identity and coverage of the daily dataset used by one experiment."""

    source: str
    sha256: str
    n_rows: int
    start_date: date
    end_date: date

    @classmethod
    def from_frame(
        cls,
        source: str | pathlib.Path,
        dataset: pd.DataFrame,
        *,
        date_column: str = "date",
    ) -> DatasetSnapshot:
        """Describe a loaded dataset and fingerprint its source file."""
        path = pathlib.Path(source)
        if date_column not in dataset:
            raise ValueError(f"dataset is missing date column: {date_column}")
        dates = pd.to_datetime(dataset[date_column], errors="coerce")
        if dataset.empty or dates.isna().any():
            raise ValueError("dataset must contain valid dates")

        digest = hashlib.sha256()
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(
            source=str(source),
            sha256=digest.hexdigest(),
            n_rows=len(dataset),
            start_date=dates.min().date(),
            end_date=dates.max().date(),
        )


@dataclass(frozen=True)
class ForecastArtifactPaths:
    """Paths written for one forecast evaluation."""

    forecasts: pathlib.Path
    loss_summary: pathlib.Path
    comparisons: pathlib.Path
    experiment: pathlib.Path

    def all(self) -> tuple[pathlib.Path, ...]:
        return (
            self.forecasts,
            self.loss_summary,
            self.comparisons,
            self.experiment,
        )


class ForecastArtifactWriter:
    """Write normalized, deterministic artifacts for one completed run."""

    def __init__(self, output_dir: str | pathlib.Path) -> None:
        self.output_dir = pathlib.Path(output_dir)

    def write(
        self,
        experiment: ExperimentResult,
        evaluation: EvaluationResult,
        *,
        config: ExperimentConfig[DescribedExecutionSettings],
        dataset: DatasetSnapshot,
        specification: ExperimentSpecification,
        configuration: Mapping[str, object] | None = None,
    ) -> ForecastArtifactPaths:
        """Write forecasts, aggregate losses, comparisons, and run metadata."""
        if not experiment.records:
            raise ValueError("experiment must contain forecast records")
        if sorted(model.name for model in specification.models) != list(experiment.models):
            raise ValueError("model specifications must match the forecast models")
        if experiment.config != config:
            raise ValueError("artifact configuration must match the completed experiment")
        if experiment.identity != evaluation.identity:
            raise ValueError("evaluation identity must match the completed experiment")
        if experiment.target_horizon != evaluation.target_horizon:
            raise ValueError("evaluation target must match the completed experiment")
        execution = experiment.execution
        if execution is None:
            raise ValueError("experiment must include execution metadata")
        if execution.settings != config.execution.specification:
            raise ValueError("execution specification must match the completed experiment")
        if any(record.fit_date is None for record in experiment.records):
            raise ValueError("forecast records must include fit dates and training boundaries")
        if evaluation.comparisons and evaluation.comparison_test is None:
            raise ValueError("comparisons must include their test specification")
        resolved = configuration_payload(configuration)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = ForecastArtifactPaths(
            forecasts=self.output_dir / "forecasts.csv",
            loss_summary=self.output_dir / "loss_summary.csv",
            comparisons=self.output_dir / "comparisons.csv",
            experiment=self.output_dir / "experiment.json",
        )

        forecast_rows = sorted(
            (
                {
                    "model": record.model,
                    "origin_date": record.origin_date,
                    "target_date": record.target_date,
                    "n_train": record.n_train,
                    "actual_log_rv": record.actual_log_rv,
                    "predicted_log_rv": record.predicted_log_rv,
                    "actual_rv": record.actual_rv,
                    "predicted_rv": record.predicted_rv,
                    "fit_date": record.fit_date,
                    "train_start_date": record.train_start_date,
                    "train_end_date": record.train_end_date,
                    "run_id": experiment.identity.run_id,
                    "fold_id": experiment.identity.fold_id,
                    "target_start_date": record.target_start,
                    "horizon_sessions": experiment.target_horizon.sessions,
                    "target_kind": experiment.target_horizon.kind,
                }
                for record in experiment.records
            ),
            key=lambda row: (row["target_date"], row["model"]),
        )
        summary_rows = sorted(
            (asdict(summary) | asdict(evaluation.identity) for summary in evaluation.summaries),
            key=lambda row: (row["metric"], row["model"]),
        )
        comparison_rows = sorted(
            (
                asdict(comparison) | asdict(evaluation.identity) | {"test_parameters": json.dumps(
                    dict(comparison.test_parameters), sort_keys=True,
                )}
                for comparison in evaluation.comparisons
            ),
            key=lambda row: (row["metric"], row["baseline_model"], row["candidate_model"]),
        )
        for row in (*summary_rows, *comparison_rows):
            row["horizon_sessions"] = experiment.target_horizon.sessions
            row["target_kind"] = experiment.target_horizon.kind

        forecast_columns = [
            "model",
            "origin_date",
            "target_date",
            "n_train",
            "actual_log_rv",
            "predicted_log_rv",
            "actual_rv",
            "predicted_rv",
            "fit_date",
            "train_start_date",
            "train_end_date",
            "run_id",
            "fold_id",
            "target_start_date",
            "horizon_sessions",
            "target_kind",
        ]
        summary_columns = [
            "model", "metric", "n_obs", "mean_loss", "run_id", "fold_id",
            "horizon_sessions", "target_kind",
        ]
        comparison_columns = [
            "metric",
            "baseline_model",
            "candidate_model",
            "n_obs",
            "baseline_mean_loss",
            "candidate_mean_loss",
            "mean_loss_difference",
            "test_name",
            "statistic",
            "p_value",
            "test_parameters",
            "run_id",
            "fold_id",
            "horizon_sessions",
            "target_kind",
        ]
        atomic_csv(pd.DataFrame(forecast_rows, columns=forecast_columns), paths.forecasts)
        atomic_csv(pd.DataFrame(summary_rows, columns=summary_columns), paths.loss_summary)
        atomic_csv(
            pd.DataFrame(comparison_rows, columns=comparison_columns),
            paths.comparisons,
        )

        target_dates = sorted({record.target_date for record in experiment.records})
        metric_names = list(dict.fromkeys(summary.metric for summary in evaluation.summaries))
        comparison_pairs = list(dict.fromkeys(
            (item.baseline_model, item.candidate_model)
            for item in evaluation.comparisons
        ))
        payload = {
            "schema_version": 9,
            "resolved_configuration": resolved,
            "identity": asdict(experiment.identity),
            "target": _component_payload(experiment.target_horizon.specification),
            "execution": {
                "executor": execution.executor,
                "settings": _component_payload(execution.settings.component),
                "policies": {
                    name: _component_payload(policy)
                    for name, policy in execution.settings.policies
                },
                "facts": dict(execution.facts),
            },
            "specification": {
                "features": _component_payload(specification.features),
                "target": _component_payload(specification.target),
                "models": [
                    _component_payload(model)
                    for model in sorted(specification.models, key=lambda item: item.name)
                ],
            },
            "dataset": {
                "instrument": config.instrument,
                "source": dataset.source,
                "sha256": dataset.sha256,
                "rows": dataset.n_rows,
                "start_date": dataset.start_date.isoformat(),
                "end_date": dataset.end_date.isoformat(),
            },
            "experiment": {
                "configured_forecast_start": (
                    None if config.forecast_start is None else config.forecast_start.isoformat()
                ),
                "configured_forecast_end": (
                    None if config.forecast_end is None else config.forecast_end.isoformat()
                ),
                "actual_forecast_start": target_dates[0].isoformat(),
                "actual_forecast_end": target_dates[-1].isoformat(),
                "target_count": len(target_dates),
                "models": list(experiment.models),
            },
            "evaluation": {
                "metrics": metric_names,
                "comparison_pairs": [
                    {"baseline": baseline, "candidate": candidate}
                    for baseline, candidate in comparison_pairs
                ],
                "comparison_test": (
                    None if evaluation.comparison_test is None
                    else _component_payload(evaluation.comparison_test)
                ),
            },
        }
        atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", paths.experiment)
        return paths


def _component_payload(specification: ComponentSpecification) -> dict[str, object]:
    return {
        "name": specification.name,
        "implementation": specification.implementation,
        "feature_names": list(specification.feature_names),
        "parameters": dict(specification.parameters),
    }
