import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def daily_dataset(tmp_path):
    n_sessions = 75
    step = np.arange(n_sessions, dtype="float64")
    log_rv = -10.0 + 0.25 * np.sin(step / 4) + 0.002 * step
    log_rv += np.random.default_rng(41).normal(0.0, 0.03, n_sessions)
    daily = pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=n_sessions),
        "rv_5min": np.exp(log_rv),
        "log_rv": log_rv,
        "observation_count": np.full(n_sessions, 288),
    })
    source = tmp_path / "daily.parquet"
    daily.to_parquet(source, index=False)
    return source, daily


@pytest.mark.parametrize(
    "comparison_pairs",
    [None, (("naive", "HAR"), ("AR1", "HAR"))],
    ids=["defaults", "overrides"],
)
def test_forecast_cli_composes_all_models_and_honours_date_bounds(
    tmp_path, daily_dataset, comparison_pairs,
):
    source, daily = daily_dataset
    n_sessions = len(daily)
    output_dir = tmp_path / "artifacts"
    start = cast(pd.Timestamp, daily.loc[50, "date"]).date().isoformat()
    end = cast(pd.Timestamp, daily.loc[55, "date"]).date().isoformat()
    comparison_args = [
        argument
        for pair in (comparison_pairs or ())
        for argument in ("--compare", *pair)
    ]
    expected_pairs = (("naive", "HAR"),) if comparison_pairs is None else comparison_pairs

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
            *comparison_args,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "targets: 6" in completed.stdout
    assert f"period:  {start} .. {end}" in completed.stdout
    for model in ("historical_mean", "naive", "EWMA", "AR1", "HAR"):
        assert model in completed.stdout
    assert "QLIKE" in completed.stdout
    assert "log-RV MSE" in completed.stdout
    assert "naive            HAR" in completed.stdout
    if comparison_pairs is not None:
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
        "fit_date",
        "train_start_date",
        "train_end_date",
    ]
    assert len(forecasts) == 30
    assert len(summaries) == 10
    assert len(comparisons) == 2 * len(expected_pairs)
    assert metadata["schema_version"] == 3
    assert metadata["execution"] == {
        "implementation": "rvol.domain.policies.WalkForwardStrategy",
        "training_window": {
            "name": "expanding",
            "implementation": "rvol.domain.policies.ExpandingWindow",
            "feature_names": [],
            "parameters": {},
        },
        "retrain": {
            "name": "every_n_sessions",
            "implementation": "rvol.domain.policies.EveryNSessions",
            "feature_names": [],
            "parameters": {"interval": 1},
        },
        "schedule_anchor": cast(pd.Timestamp, daily.loc[41, "date"]).date().isoformat(),
        "schedule_anchor_policy": "first_eligible_origin",
        "observation_update_policy": "newly_available_targets_between_refits",
    }
    assert forecasts["fit_date"].equals(forecasts["origin_date"])
    assert forecasts["train_end_date"].equals(forecasts["origin_date"])
    assert set(forecasts["train_start_date"]) == {
        cast(pd.Timestamp, daily.loc[22, "date"]).date().isoformat(),
    }
    assert metadata["dataset"]["rows"] == n_sessions
    assert len(metadata["dataset"]["sha256"]) == 64
    assert metadata["experiment"] == {
        "actual_forecast_end": end,
        "actual_forecast_start": start,
        "configured_forecast_end": end,
        "configured_forecast_start": start,
        "min_train_size": 20,
        "models": ["AR1", "EWMA", "HAR", "historical_mean", "naive"],
        "target_count": 6,
    }
    specification = metadata["specification"]
    assert specification["features"] == {
        "name": "HAR_features",
        "implementation": "rvol.features.forecasting.HARFeatureBuilder",
        "feature_names": ["rv_daily", "rv_weekly", "rv_monthly"],
        "parameters": {
            "date_col": "date", "rv_col": "log_rv", "weekly_window": 5,
            "monthly_window": 22, "max_gap_days": 7,
        },
    }
    assert [model["name"] for model in specification["models"]] == metadata["experiment"]["models"]
    models = {model["name"]: model for model in specification["models"]}
    assert models["EWMA"]["parameters"] == {"decay": 0.94}
    assert models["HAR"]["parameters"] == {"variance_correction": "duan_smearing"}
    assert models["HAR"]["feature_names"] == ["rv_daily", "rv_weekly", "rv_monthly"]
    assert models["AR1"]["feature_names"] == models["naive"]["feature_names"] == ["rv_daily"]
    assert models["historical_mean"]["feature_names"] == models["EWMA"]["feature_names"] == []
    assert metadata["evaluation"]["hac_lags"] == 2
    assert metadata["evaluation"]["comparison_pairs"] == [
        {"baseline": baseline, "candidate": candidate}
        for baseline, candidate in expected_pairs
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


@pytest.mark.parametrize("metric", [None, "log-RV MSE"], ids=["default_metric", "log_mse"])
def test_plot_cli_uses_the_shared_experiment_and_metric_choices(tmp_path, daily_dataset, metric):
    source, _ = daily_dataset
    output = tmp_path / "forecast.png"
    metric_args = [] if metric is None else ["--metric", metric]

    completed = subprocess.run(
        [
            sys.executable, "scripts/forecast_plots.py", str(source),
            "--min-train-size", "20", "--forecast-start", "2022-03-14",
            "--forecast-end", "2022-03-21", "--out", str(output), *metric_args,
        ],
        cwd=PROJECT_ROOT, check=False, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "targets: 6" in completed.stdout
    assert f"metric:  {metric or 'QLIKE'}" in completed.stdout
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_robustness_cli_uses_shared_baselines_and_preserves_option_overrides(
    tmp_path, daily_dataset,
):
    source, _ = daily_dataset
    output = tmp_path / "robustness"

    completed = subprocess.run(
        [
            sys.executable, "scripts/robustness_report.py", str(source),
            "--min-train-size", "20", "--forecast-start", "2022-03-14",
            "--forecast-end", "2022-03-21", "--max-hac-lag", "1",
            "--top-errors", "2", "--output-dir", str(output),
        ],
        cwd=PROJECT_ROOT, check=False, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "targets:          6" in completed.stdout
    comparisons = pd.read_csv(output / "hac_sensitivity.csv")
    assert len(comparisons) == 16
    assert set(comparisons["hac_lags"]) == {0, 1}
    assert set(comparisons["candidate_model"]) == {"HAR"}
    assert set(comparisons["baseline_model"]) == {"naive", "EWMA", "AR1", "historical_mean"}
    assert len(pd.read_csv(output / "largest_errors.csv")) == 10
    assert len(pd.read_csv(output / "win_rates.csv")) == 8
