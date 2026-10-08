"""Deterministic artifacts rendered from selected diagnostic result tables."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import pandas as pd

from rvol.evaluation import RobustnessResult
from rvol.reporting._files import atomic_csv, atomic_text
from rvol.reporting._metadata import configuration_payload


@dataclass(frozen=True)
class RobustnessArtifactPaths:
    diagnostics: tuple[tuple[str, pathlib.Path], ...]
    report: pathlib.Path
    metadata: pathlib.Path

    def csv(self, name: str) -> pathlib.Path:
        return dict(self.diagnostics)[name]

    def all(self) -> tuple[pathlib.Path, ...]:
        return tuple(path for _, path in self.diagnostics) + (self.report, self.metadata)


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


class RobustnessArtifactWriter:
    """Render check-owned columns, labels and formatting without diagnostic branches."""

    def __init__(self, output_dir: str | pathlib.Path) -> None:
        self.output_dir = pathlib.Path(output_dir)

    def write(
        self, robustness: RobustnessResult, *, configuration: Mapping[str, object] | None = None,
    ) -> RobustnessArtifactPaths:
        names = [result.name for result in robustness.checks]
        if not names or len(set(names)) != len(names):
            raise ValueError("robustness results must have non-empty, unique checks")
        resolved = configuration_payload(configuration)
        sections = []
        for result in robustness.checks:
            table = _table(
                tuple(label for label, _ in result.report_columns),
                [tuple(template.format_map(row) for _, template in result.report_columns)
                 for row in result.records()],
            )
            sections.append(f"## {result.title}\n\n{table}\n")
        # Validate every template before creating any output files.
        paths = RobustnessArtifactPaths(
            diagnostics=tuple((name, self.output_dir / f"{name}.csv") for name in names),
            report=self.output_dir / "robustness.md",
            metadata=self.output_dir / "robustness.json",
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for result in robustness.checks:
            frame = pd.DataFrame(result.rows, columns=result.columns)
            for name, value in (asdict(robustness.identity) | {
                "horizon_sessions": robustness.target_horizon.sessions,
                "target_kind": robustness.target_horizon.kind,
            }).items():
                frame[name] = value
            atomic_csv(frame, paths.csv(result.name))
        report = (
            "# Forecast robustness report\n\n"
            + (f"Instrument: {robustness.instrument}\n\n" if robustness.instrument else "")
            + f"Target: {robustness.target_horizon.kind}, "
            f"horizon {robustness.target_horizon.sessions} observed sessions.\n\n"
            "All checks use the fixed model specifications and out-of-sample forecasts. "
            "No model is refitted or selected from these diagnostics.\n\n"
            + "\n".join(sections)
        )
        atomic_text(report, paths.report)
        payload = {
            "schema_version": 4,
            "instrument": robustness.instrument,
            "resolved_configuration": resolved,
            "identity": asdict(robustness.identity),
            "target": asdict(robustness.target_horizon),
            "checks": [
                {
                    "specification": asdict(result.specification) | {
                        "parameters": dict(result.specification.parameters),
                    },
                    "components": {
                        name: asdict(component) | {"parameters": dict(component.parameters)}
                        for name, component in result.components
                    },
                    "rows": len(result.rows),
                    "columns": list(result.columns),
                }
                for result in robustness.checks
            ],
        }
        atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", paths.metadata)
        return paths
