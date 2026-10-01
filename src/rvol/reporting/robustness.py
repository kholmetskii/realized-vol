"""Deterministic artifacts for forecast robustness diagnostics."""

from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass

import pandas as pd

from rvol.evaluation import RobustnessResult
from rvol.reporting._files import atomic_csv, atomic_text


@dataclass(frozen=True)
class RobustnessArtifactPaths:
    """Paths written for one robustness report."""

    hac_sensitivity: pathlib.Path
    monthly_losses: pathlib.Path
    win_rates: pathlib.Path
    largest_errors: pathlib.Path
    report: pathlib.Path

    def all(self) -> tuple[pathlib.Path, ...]:
        return (
            self.hac_sensitivity,
            self.monthly_losses,
            self.win_rates,
            self.largest_errors,
            self.report,
        )


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


class RobustnessArtifactWriter:
    """Write CSV detail and a human-readable Markdown robustness report."""

    def __init__(self, output_dir: str | pathlib.Path) -> None:
        self.output_dir = pathlib.Path(output_dir)

    def write(self, robustness: RobustnessResult) -> RobustnessArtifactPaths:
        """Persist every robustness diagnostic in deterministic order."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = RobustnessArtifactPaths(
            hac_sensitivity=self.output_dir / "hac_sensitivity.csv",
            monthly_losses=self.output_dir / "monthly_losses.csv",
            win_rates=self.output_dir / "win_rates.csv",
            largest_errors=self.output_dir / "largest_errors.csv",
            report=self.output_dir / "robustness.md",
        )

        hac_rows = sorted(
            (asdict(item) for item in robustness.hac_comparisons),
            key=lambda row: (
                row["metric"],
                row["baseline_model"],
                row["candidate_model"],
                row["hac_lags"],
            ),
        )
        monthly_rows = sorted(
            (asdict(item) for item in robustness.monthly_losses),
            key=lambda row: (row["month"], row["metric"], row["model"]),
        )
        win_rows = sorted(
            (asdict(item) for item in robustness.win_rates),
            key=lambda row: (row["metric"], row["baseline_model"]),
        )
        error_rows = sorted(
            (asdict(item) for item in robustness.largest_errors),
            key=lambda row: (row["model"], -row["absolute_log_error"]),
        )
        atomic_csv(pd.DataFrame(hac_rows), paths.hac_sensitivity)
        atomic_csv(pd.DataFrame(monthly_rows), paths.monthly_losses)
        atomic_csv(pd.DataFrame(win_rows), paths.win_rates)
        atomic_csv(pd.DataFrame(error_rows), paths.largest_errors)

        hac_table = _table(
            ("Metric", "Baseline", "Candidate", "NW lags", "DM", "p-value"),
            [
                (
                    row["metric"],
                    row["baseline_model"],
                    row["candidate_model"],
                    str(row["hac_lags"]),
                    f"{row['dm_statistic']:.3f}",
                    f"{row['p_value']:.4g}",
                )
                for row in hac_rows
            ],
        )
        win_table = _table(
            ("Metric", "Baseline", "Candidate", "Wins", "Ties", "Win rate"),
            [
                (
                    row["metric"],
                    row["baseline_model"],
                    row["candidate_model"],
                    f"{row['candidate_wins']}/{row['n_obs']}",
                    str(row["ties"]),
                    f"{row['candidate_win_rate']:.1%}",
                )
                for row in win_rows
            ],
        )
        monthly_table = _table(
            ("Month", "Metric", "Model", "n", "Mean loss"),
            [
                (
                    row["month"],
                    row["metric"],
                    row["model"],
                    str(row["n_obs"]),
                    f"{row['mean_loss']:.6g}",
                )
                for row in monthly_rows
            ],
        )
        error_table = _table(
            ("Model", "Target", "Actual log RV", "Predicted", "Absolute error"),
            [
                (
                    row["model"],
                    row["target_date"].isoformat(),
                    f"{row['actual_log_rv']:.6f}",
                    f"{row['predicted_log_rv']:.6f}",
                    f"{row['absolute_log_error']:.6f}",
                )
                for row in error_rows
            ],
        )
        report = (
            "# Forecast robustness report\n\n"
            "All checks use the fixed model specifications and out-of-sample forecasts. "
            "No model is refitted or selected from these diagnostics.\n\n"
            "## Newey–West lag sensitivity\n\n"
            f"{hac_table}\n\n"
            "## Candidate session-level win rates\n\n"
            f"{win_table}\n\n"
            "## Monthly forecast losses\n\n"
            f"{monthly_table}\n\n"
            "## Largest absolute log-RV errors\n\n"
            f"{error_table}\n"
        )
        atomic_text(report, paths.report)
        return paths
