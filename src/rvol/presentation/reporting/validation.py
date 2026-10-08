"""Artifacts preserving separate run/fold scores and their aggregation rule."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass

import pandas as pd

from rvol.forecasting.evaluation import ValidationResult
from rvol.presentation.reporting._files import atomic_csv, atomic_text


@dataclass(frozen=True)
class ValidationArtifactPaths:
    losses: pathlib.Path
    run_summaries: pathlib.Path
    run_comparisons: pathlib.Path
    aggregate_summaries: pathlib.Path
    metadata: pathlib.Path

    def all(self) -> tuple[pathlib.Path, ...]:
        return (self.losses, self.run_summaries, self.run_comparisons,
                self.aggregate_summaries, self.metadata)


class ValidationArtifactWriter:
    def __init__(self, output_dir: str | pathlib.Path) -> None:
        self.output_dir = pathlib.Path(output_dir)

    def write(self, validation: ValidationResult) -> ValidationArtifactPaths:
        identities = [run.identity for run in validation.runs]
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("validation results must have unique, non-empty run/fold identities")
        if len({run.target_horizon for run in validation.runs}) != 1:
            raise ValueError("validation results must use identical target horizons")
        horizon = validation.runs[0].target_horizon
        target_fields = {"horizon_sessions": horizon.sessions, "target_kind": horizon.kind}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = ValidationArtifactPaths(
            self.output_dir / "losses.csv", self.output_dir / "run_loss_summary.csv",
            self.output_dir / "run_comparisons.csv", self.output_dir / "aggregate_loss_summary.csv",
            self.output_dir / "validation.json",
        )
        losses = [
            asdict(item) | asdict(run.identity) | target_fields
            for run in validation.runs for item in run.losses
        ]
        summaries = [
            asdict(item) | asdict(run.identity) | target_fields
            for run in validation.runs for item in run.summaries
        ]
        comparisons = [
            asdict(item) | asdict(run.identity) | target_fields | {
                "test_parameters": json.dumps(dict(item.test_parameters), sort_keys=True),
            }
            for run in validation.runs for item in run.comparisons
        ]
        atomic_csv(pd.DataFrame(losses), paths.losses)
        atomic_csv(pd.DataFrame(summaries), paths.run_summaries)
        atomic_csv(pd.DataFrame(comparisons, columns=[
            "metric", "baseline_model", "candidate_model", "n_obs", "baseline_mean_loss",
            "candidate_mean_loss", "mean_loss_difference", "test_name", "statistic", "p_value",
            "test_parameters", "run_id", "fold_id",
            "horizon_sessions", "target_kind",
        ]), paths.run_comparisons)
        atomic_csv(
            pd.DataFrame([asdict(item) | target_fields for item in validation.summaries]),
            paths.aggregate_summaries,
        )
        payload = {
            "schema_version": 3,
            "target": asdict(horizon),
            "aggregation": validation.aggregation,
            "overlap_policy": "keep_separate_run_fold_forecasts",
            "statistical_tests": "within_each_run_fold_only",
            "runs": [
                {
                    "identity": asdict(run.identity),
                    "instrument": run.instrument,
                    "forecast_cases": len({(item.model, item.target_date) for item in run.losses}),
                    "comparison_test": None if run.comparison_test is None else (
                        asdict(run.comparison_test)
                        | {"parameters": dict(run.comparison_test.parameters)}
                    ),
                }
                for run in validation.runs
            ],
        }
        atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", paths.metadata)
        return paths
