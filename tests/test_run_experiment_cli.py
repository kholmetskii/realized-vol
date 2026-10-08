import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from rvol.application import HoldoutExperiment
from rvol.cli import run_experiment
from rvol.composition import (
    ExperimentRun,
    standard_diagnostic_checks,
    standard_experiment_definition,
)
from rvol.domain import (
    EveryNSessions,
    ExpandingWindow,
    ExperimentConfig,
    ExperimentResult,
    HoldoutSettings,
    RollingWindow,
    RunIdentity,
    TargetHorizon,
    WalkForwardSettings,
)
from rvol.reporting import (
    DatasetSnapshot,
    ForecastArtifactWriter,
    RobustnessArtifactWriter,
)

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
            "scripts/run_experiment.py",
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
    assert "comparison test: DM; parameters: {'hac_lags': 2}" in completed.stdout

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
        "run_id",
        "fold_id",
        "target_start_date",
        "horizon_sessions",
        "target_kind",
    ]
    assert len(forecasts) == 30
    assert len(summaries) == 10
    assert len(comparisons) == 2 * len(expected_pairs)
    assert metadata["schema_version"] == 9
    assert metadata["execution"] == {
        "executor": "rvol.application.forecast_experiment.WalkForwardExperiment",
        "settings": {
            "name": "walk_forward",
            "implementation": "rvol.domain.settings.WalkForwardSettings",
            "feature_names": [],
            "parameters": {"min_train_size": 20},
        },
        "policies": {
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
        },
        "facts": {
            "schedule_anchor": cast(pd.Timestamp, daily.loc[41, "date"]).date().isoformat(),
            "schedule_anchor_policy": "first_eligible_origin",
            "observation_update_policy": "newly_available_targets_between_refits",
        },
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
    assert metadata["evaluation"]["comparison_test"]["parameters"]["hac_lags"] == 2
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


@pytest.mark.parametrize("training_window", ["expanding", "rolling"])
@pytest.mark.parametrize("sessions, kind", [(1, "session"), (3, "session"), (3, "cumulative")])
def test_forecast_cli_strategy_matches_the_python_run(
    tmp_path, daily_dataset, training_window, sessions, kind,
):
    source, daily = daily_dataset
    output = tmp_path / "artifacts"
    start = cast(pd.Timestamp, daily.loc[50, "date"]).date()
    end = cast(pd.Timestamp, daily.loc[55, "date"]).date()
    window_args = ["--window-size", "25"] if training_window == "rolling" else []
    completed = subprocess.run(
        [
            sys.executable, "scripts/run_experiment.py", str(source),
            "--min-train-size", "20", "--training-window", training_window, *window_args,
            "--retrain-every", "5", "--forecast-start", start.isoformat(),
            "--horizon", str(sessions), "--target-kind", kind,
            "--forecast-end", end.isoformat(), "--output-dir", str(output),
        ],
        cwd=PROJECT_ROOT, check=False, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    definition = standard_experiment_definition(target_horizon=TargetHorizon(sessions, kind))
    config = ExperimentConfig(execution=WalkForwardSettings(
            RollingWindow(25) if training_window == "rolling" else ExpandingWindow(),
            EveryNSessions(5), min_train_size=20), forecast_start=start, forecast_end=end)
    experiment = definition.run(daily, config)
    expected = ForecastArtifactWriter(tmp_path / "expected").write(
        experiment, definition.evaluator().evaluate(experiment), config=config,
        dataset=DatasetSnapshot.from_frame(source, daily), specification=definition.specification,
        configuration=ExperimentRun(definition, config).resolved_configuration,
    )
    assert {path.name: path.read_bytes() for path in expected.all()} == {
        path.name: (output / path.name).read_bytes() for path in expected.all()
    }
    forecasts = pd.read_csv(output / "forecasts.csv")
    assert forecasts["fit_date"].nunique() == 2
    assert (forecasts["fit_date"] < forecasts["origin_date"]).any()
    if sessions == 1:
        assert set(forecasts["n_train"]) == ({25} if training_window == "rolling" else {25, 30})
    else:
        assert set(forecasts["n_train"]) == {20, 25}


@pytest.mark.parametrize("metric", [None, "log-RV MSE"], ids=["default_metric", "log_mse"])
def test_plot_option_uses_the_shared_experiment_and_metric_choices(tmp_path, daily_dataset, metric):
    source, _ = daily_dataset
    output = tmp_path / "artifacts"
    metric_args = [] if metric is None else ["--metric", metric]
    completed = subprocess.run(
        [
            sys.executable, "scripts/run_experiment.py", str(source),
            "--min-train-size", "20", "--forecast-start", "2022-03-14",
            "--forecast-end", "2022-03-21", "--plot", "--output-dir", str(output), *metric_args,
        ],
        cwd=PROJECT_ROOT, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "targets: 6" in completed.stdout
    assert f"plot metric: {metric or 'QLIKE'}" in completed.stdout
    assert (output / "forecasts.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "forecasts.csv").exists()
    assert not (output / "robustness").exists()


@pytest.mark.parametrize("rolling", [False, True], ids=["default", "rolling_refit"])
def test_robustness_option_preserves_overrides_and_matches_the_python_api(
    tmp_path, daily_dataset, rolling,
):
    source, daily = daily_dataset
    output = tmp_path / "artifacts"
    strategy_args = (
        ["--training-window", "rolling", "--window-size", "25", "--retrain-every", "5"]
        if rolling else []
    )
    completed = subprocess.run(
        [
            sys.executable, "scripts/run_experiment.py", str(source),
            "--min-train-size", "20", "--forecast-start", "2022-03-14",
            "--forecast-end", "2022-03-21", "--robustness", "--max-hac-lag", "1",
            "--top-errors", "2", "--output-dir", str(output), *strategy_args,
        ],
        cwd=PROJECT_ROOT, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "targets: 6" in completed.stdout
    diagnostics = output / "robustness"
    comparisons = pd.read_csv(diagnostics / "hac_sensitivity.csv")
    assert len(comparisons) == 16
    assert set(comparisons["hac_lags"]) == {0, 1}
    assert set(comparisons["candidate_model"]) == {"HAR"}
    assert set(comparisons["baseline_model"]) == {"naive", "EWMA", "AR1", "historical_mean"}
    assert len(pd.read_csv(diagnostics / "largest_errors.csv")) == 10
    assert len(pd.read_csv(diagnostics / "win_rates.csv")) == 8
    assert (output / "forecasts.csv").exists()
    assert not (output / "forecasts.png").exists()
    if rolling:
        definition = standard_experiment_definition()
        config = ExperimentConfig(
            execution=WalkForwardSettings(RollingWindow(25), EveryNSessions(5), min_train_size=20),
            forecast_start=date(2022, 3, 14), forecast_end=date(2022, 3, 21),
        )
        experiment = definition.run(daily, config)
        expected = RobustnessArtifactWriter(tmp_path / "expected").write(
            definition.robustness_analyzer(hac_lags=(0, 1), top_errors=2).analyze(experiment),
            configuration=ExperimentRun(
                definition, config,
                diagnostic_checks=standard_diagnostic_checks(hac_lags=(0, 1), top_errors=2),
            ).resolved_configuration,
        )
        assert {p.name: p.read_bytes() for p in expected.all()} == {
            p.name: (diagnostics / p.name).read_bytes() for p in expected.all()
        }


@pytest.mark.parametrize("arguments, message", [
    (["--horizon", "0"], "expected a positive integer"),
    (["--target-kind", "unknown"], "invalid choice"),
    (["--training-window", "rolling"], "execution.window_size is required"),
    (["--executor", "holdout"], "execution.train_end is required"),
    (["--train-end", "2024-12-31"], "execution.train_end requires execution.name = 'holdout'"),
    (["--update-observations"],
     "execution.update_observations requires execution.name = 'holdout'"),
    (["--executor", "holdout", "--train-end", "2024-12-31", "--training-window", "expanding"],
     "execution.training_window requires execution.name = 'walk-forward'"),
    (["--executor", "holdout", "--train-end", "2024-12-31", "--window-size", "504"],
     "execution.window_size requires execution.name = 'walk-forward'"),
    (["--executor", "holdout", "--train-end", "2024-12-31", "--retrain-every", "1"],
     "execution.retrain_every requires execution.name = 'walk-forward'"),
])
def test_command_validates_execution_options_before_loading_data(
    monkeypatch, capsys, arguments, message,
):
    def unexpected_load(self):
        raise AssertionError("invalid options must fail before data I/O")

    monkeypatch.setattr(run_experiment.ParquetDatasetRepository, "load", unexpected_load)
    monkeypatch.setattr(sys, "argv", ["run_experiment", "missing.parquet", *arguments])
    with pytest.raises(SystemExit) as error:
        run_experiment.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize(
    "sessions, target_kind", [(1, "session"), (3, "session"), (3, "cumulative")],
)
def test_all_outputs_use_holdout_results_from_the_python_api(
    tmp_path, daily_dataset, monkeypatch, update, sessions, target_kind,
):
    from rvol.plotting import forecasting

    source, daily = daily_dataset
    definition = standard_experiment_definition(
        executor_factory=HoldoutExperiment, target_horizon=TargetHorizon(sessions, target_kind),
    )
    config = ExperimentConfig(
        execution=HoldoutSettings(date(2022, 3, 4), 20, update),
        identity=RunIdentity("holdout_check", "fold_1"),
        forecast_start=date(2022, 3, 14), forecast_end=date(2022, 3, 21),
    )
    expected = definition.run(daily, config)
    checks = standard_diagnostic_checks(hac_lags=(0, 1), top_errors=2)
    configuration = ExperimentRun(definition, config, checks).resolved_configuration
    output = tmp_path / "holdout"
    captured: list[ExperimentResult] = []
    plot = forecasting.plot_forecast_evaluation

    def capture(experiment, *args, **kwargs):
        captured.append(experiment)
        return plot(experiment, *args, **kwargs)

    monkeypatch.setattr(forecasting, "plot_forecast_evaluation", capture)
    monkeypatch.setattr(sys, "argv", [
        "run_experiment", str(source), "--executor", "holdout", "--train-end", "2022-03-04",
        "--run-id", "holdout_check", "--fold-id", "fold_1",
        "--horizon", str(sessions), "--target-kind", target_kind,
        "--min-train-size", "20", "--forecast-start", "2022-03-14",
        "--forecast-end", "2022-03-21", *(["--update-observations"] if update else []),
        "--plot", "--robustness", "--max-hac-lag", "1", "--top-errors", "2",
        "--output-dir", str(output),
    ])
    run_experiment.main()
    assert captured == [expected]
    assert (output / "forecasts.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    evaluation = definition.evaluator().evaluate(expected)
    numeric_paths = ForecastArtifactWriter(tmp_path / "expected").write(
        expected, evaluation, config=config, dataset=DatasetSnapshot.from_frame(source, daily),
        specification=definition.specification, configuration=configuration,
    ).all()
    robust_paths = RobustnessArtifactWriter(tmp_path / "expected" / "robustness").write(
        definition.robustness_analyzer(checks=checks).analyze(expected, evaluation=evaluation),
        configuration=configuration,
    ).all()
    for directory, paths in ((output, numeric_paths), (output / "robustness", robust_paths)):
        assert {p.name: p.read_bytes() for p in paths} == {
            p.name: (directory / p.name).read_bytes() for p in paths
        }


def test_selected_descriptive_checks_work_on_a_short_report(tmp_path, daily_dataset, monkeypatch):
    source, _ = daily_dataset
    output = tmp_path / "monthly"
    monkeypatch.setattr(sys, "argv", [
        "run_experiment", str(source), "--min-train-size", "20",
        "--forecast-start", "2022-03-14", "--forecast-end", "2022-03-14",
        "--baselines", "--robustness", "--check", "monthly_losses", "--output-dir", str(output),
    ])
    run_experiment.main()
    assert {path.name for path in (output / "robustness").iterdir()} == {
        "monthly_losses.csv", "robustness.md", "robustness.json",
    }
    metadata = json.loads((output / "robustness" / "robustness.json").read_text())
    assert [item["specification"]["name"] for item in metadata["checks"]] == ["monthly_losses"]


def test_duplicate_checks_fail_before_loading_data(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "run_experiment", "missing.parquet", "--robustness",
        "--check", "win_rates", "--check", "win_rates",
    ])
    with pytest.raises(SystemExit) as error:
        run_experiment.main()
    assert error.value.code == 2
    assert "unique" in capsys.readouterr().err


def test_default_output_directory_contains_only_numerical_results(
    tmp_path, daily_dataset, monkeypatch,
):
    source, _ = daily_dataset
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_experiment", str(source), "--min-train-size", "20"])
    run_experiment.main()
    assert {path.name for path in (tmp_path / "outputs" / "experiment").iterdir()} == {
        "forecasts.csv", "loss_summary.csv", "comparisons.csv", "experiment.json",
    }


@pytest.mark.parametrize("reports", [[], ["--plot", "--robustness"]], ids=["numbers", "reports"])
def test_single_model_run_needs_no_comparison_configuration(
    tmp_path, daily_dataset, monkeypatch, reports,
):
    source, _ = daily_dataset
    output = tmp_path / "single"
    monkeypatch.setattr(sys, "argv", [
        "run_experiment", str(source), "--models", "HAR", "--min-train-size", "20",
        "--output-dir", str(output), *reports,
    ])
    run_experiment.main()
    forecasts = pd.read_csv(output / "forecasts.csv")
    assert set(forecasts.model) == {"HAR"}
    assert pd.read_csv(output / "comparisons.csv").empty
    assert (output / "forecasts.png").exists() == ("--plot" in reports)
    if "--robustness" in reports:
        assert {p.name for p in (output / "robustness").iterdir()} == {
            "monthly_losses.csv", "largest_errors.csv", "robustness.md", "robustness.json",
        }


def test_empty_experiment_does_not_write_outputs(tmp_path, daily_dataset, monkeypatch):
    source, _ = daily_dataset
    output = tmp_path / "empty"
    monkeypatch.setattr(sys, "argv", [
        "run_experiment", str(source), "--forecast-start", "2030-01-01",
        "--output-dir", str(output), "--plot", "--robustness",
    ])
    with pytest.raises(SystemExit, match="no forecasts"):
        run_experiment.main()
    assert not output.exists()
