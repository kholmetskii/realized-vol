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
