import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_forecast_cli_composes_all_models_and_honours_date_bounds(tmp_path):
    n_sessions = 75
    step = np.arange(n_sessions, dtype="float64")
    log_rv = -10.0 + 0.25 * np.sin(step / 4) + 0.002 * step
    daily = pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=n_sessions),
        "rv_5min": np.exp(log_rv),
        "log_rv": log_rv,
        "observation_count": np.full(n_sessions, 288),
    })
    source = tmp_path / "daily.parquet"
    output_dir = tmp_path / "artifacts"
    daily.to_parquet(source, index=False)
    start = daily.loc[50, "date"].date().isoformat()
    end = daily.loc[55, "date"].date().isoformat()

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_forecasts.py",
            str(source),
            "--min-train-size",
            "20",
            "--forecast-start",
            start,
            "--forecast-end",
            end,
            "--hac-lags",
            "2",
            "--output-dir",
            str(output_dir),
            "--compare",
            "naive",
            "HAR",
            "--compare",
            "AR1",
            "HAR",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "targets: 6" in completed.stdout
    assert f"period:  {start} .. {end}" in completed.stdout
    for model in ("historical_mean", "naive", "AR1", "HAR"):
        assert model in completed.stdout
    assert "QLIKE" in completed.stdout
    assert "log-RV MSE" in completed.stdout
    assert "naive            HAR" in completed.stdout
    assert "AR1              HAR" in completed.stdout
    assert "Newey-West lags: 2" in completed.stdout

    artifact_names = (
        "forecasts.csv",
        "loss_summary.csv",
        "comparisons.csv",
        "experiment.json",
    )
    first_run = {
        name: (output_dir / name).read_bytes()
        for name in artifact_names
    }
    forecasts = pd.read_csv(output_dir / "forecasts.csv")
    summaries = pd.read_csv(output_dir / "loss_summary.csv")
    comparisons = pd.read_csv(output_dir / "comparisons.csv")
    metadata = json.loads((output_dir / "experiment.json").read_text())

    assert list(forecasts.columns) == [
        "model",
        "origin_date",
        "target_date",
        "n_train",
        "actual_log_rv",
        "predicted_log_rv",
        "actual_rv",
        "predicted_rv",
    ]
    assert len(forecasts) == 24
    assert len(summaries) == 8
    assert len(comparisons) == 4
    assert metadata["schema_version"] == 1
    assert metadata["dataset"]["rows"] == n_sessions
    assert len(metadata["dataset"]["sha256"]) == 64
    assert metadata["experiment"] == {
        "actual_forecast_end": end,
        "actual_forecast_start": start,
        "configured_forecast_end": end,
        "configured_forecast_start": start,
        "min_train_size": 20,
        "models": ["AR1", "HAR", "historical_mean", "naive"],
        "target_count": 6,
    }
    assert metadata["evaluation"]["hac_lags"] == 2
    assert metadata["evaluation"]["comparison_pairs"] == [
        {"baseline": "naive", "candidate": "HAR"},
        {"baseline": "AR1", "candidate": "HAR"},
    ]

    repeated = subprocess.run(
        completed.args,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert repeated.returncode == 0, repeated.stderr
    assert {
        name: (output_dir / name).read_bytes()
        for name in artifact_names
    } == first_run
