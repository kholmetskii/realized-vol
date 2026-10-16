import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from rvol.cli._arguments import add_forecast_arguments, experiment_from_args
from rvol.domain import (
    ComponentSpecification,
    EveryNSessions,
    ExperimentConfig,
    HoldoutSettings,
    RollingWindow,
    RunIdentity,
    TargetHorizon,
    WalkForwardSettings,
)
from rvol.domain.contracts import FloatArray, ForecastScale
from rvol.forecasting.composition import Experiment, standard_experiment
from rvol.forecasting.evaluation import (
    DieboldMarianoTest,
    LargestErrorsCheck,
    LogMSEMetric,
    MonthlyLossCheck,
)
from rvol.forecasting.evaluation.diagnostics import DiagnosticContext, DiagnosticResult
from rvol.forecasting.features import HARFeatureBuilder
from rvol.forecasting.models import EWMAModel, HARModel, HistoricalMeanModel
from rvol.forecasting.targets import RealizedVarianceTargetBuilder
from rvol.presentation.reporting import DatasetSnapshot, ForecastArtifactWriter


def daily_sample() -> pd.DataFrame:
    values = -10 + np.sin(np.arange(75) / 4) / 5
    return pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=75), "log_rv": values})


def parse(arguments: list[str], experiment: Experiment | None = None) -> Experiment:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_forecast_arguments(parser)
    return experiment_from_args(parser, parser.parse_args(arguments), experiment)


def test_python_setup_records_selected_components_and_their_actual_parameters():
    horizon = TargetHorizon(3, "cumulative")
    base = standard_experiment(ExperimentConfig(
        execution=WalkForwardSettings(RollingWindow(25), EveryNSessions(3), min_train_size=20),
        instrument="GBP/USD",
    ), target_horizon=horizon)
    naive = next(model for model in base.models if model.name == "naive")
    run = replace(
        base,
        feature_builder=HARFeatureBuilder(weekly_window=2, monthly_window=4, max_gap_days=None),
        models=(naive, EWMAModel(decay=0.8), HARModel()),
        metrics=(LogMSEMetric(),), plot_metric="log-RV MSE",
        candidate_model="EWMA", baseline_models=("naive", "HAR"),
        comparison_pairs=(("HAR", "EWMA"),),
        comparison_test=DieboldMarianoTest(hac_lags=0, min_hac_lags=2),
        diagnostic_checks=(MonthlyLossCheck(), LargestErrorsCheck(2)),
    )
    resolved = json.loads(json.dumps(run.resolved_configuration))
    assert resolved["dataset"]["instrument"] == "GBP/USD"
    assert resolved["features"]["parameters"]["max_gap_days"] is None
    assert resolved["target"]["name"] == "realized_variance"
    assert resolved["target"]["parameters"] == {
        "date_col": "date", "rv_col": "log_rv", "horizon_sessions": 3,
        "target_kind": "cumulative",
    }
    assert "horizon_sessions" not in resolved["features"]["parameters"]
    models = {model["name"]: model for model in resolved["models"]}
    assert models["EWMA"]["parameters"] == {"decay": 0.8}
    assert models["naive"]["parameters"]["variance_multiplier"] == 3.0
    assert run.plot_metric == "log-RV MSE"
    result = run.run(daily_sample())
    assert result.config is not None and result.config.instrument == "GBP/USD"
    evaluation = run.evaluator().evaluate(result)
    assert {summary.metric for summary in evaluation.summaries} == {"log-RV MSE"}
    assert {comparison.candidate_model for comparison in evaluation.comparisons} == {"EWMA"}
    assert evaluation.instrument == "GBP/USD"


@pytest.mark.parametrize("changes", [
    {"models": ()},
    {"models": (HARModel(), HARModel())},
    {"metrics": ()},
    {"metrics": (LogMSEMetric(), LogMSEMetric()), "plot_metric": None},
    {"candidate_model": "missing"},
    {"baseline_models": ("missing",)},
    {"baseline_models": ("HAR",)},
    {"baseline_models": ("EWMA", "EWMA")},
    {"comparison_pairs": (("naive", "naive"),)},
    {"comparison_pairs": (("naive", "HAR"), ("naive", "HAR"))},
    {"comparison_pairs": (("missing", "HAR"),)},
    {"plot_metric": "missing"},
    {"diagnostic_checks": ()},
    {"diagnostic_checks": (MonthlyLossCheck(), MonthlyLossCheck())},
])
def test_invalid_python_selections_are_rejected_when_the_experiment_is_constructed(changes):
    with pytest.raises(ValueError):
        replace(standard_experiment(), **changes)


def test_editable_python_presets_build_independent_walk_forward_and_holdout_experiments():
    from experiments.eurusd import build_experiment as build_eurusd
    from experiments.holdout import build_experiment as build_holdout

    first, second = build_eurusd(), build_eurusd()
    holdout = build_holdout()

    assert first.config == ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=252),
        instrument="EUR/USD", identity=RunIdentity("eurusd_baseline"),
        forecast_start=date(2025, 1, 1), forecast_end=date(2026, 9, 30),
    )
    assert holdout.config == replace(
        first.config, execution=HoldoutSettings(date(2024, 12, 31), min_train_size=252),
        identity=RunIdentity("eurusd_holdout"),
    )
    assert first.resolved_configuration == second.resolved_configuration
    assert all(a is not b for a, b in zip(first.models, second.models, strict=True))
    assert all(a is not b for a, b in zip(first.models, holdout.models, strict=True))
    assert first.specification == holdout.specification


@pytest.mark.parametrize("module", ["eurusd", "holdout"])
def test_python_preset_module_matches_direct_execution(tmp_path, module):
    from experiments.eurusd import build_experiment as build_eurusd
    from experiments.holdout import build_experiment as build_holdout

    step = np.arange(600)
    daily = pd.DataFrame({
        "date": pd.bdate_range("2023-01-03", periods=600),
        "log_rv": -10 + np.sin(step / 4) / 5,
    })
    source = tmp_path / "daily.parquet"
    daily.to_parquet(source, index=False)
    output = tmp_path / "output"
    completed = subprocess.run(
        [sys.executable, "-m", f"experiments.{module}", str(source),
         "--forecast-start", "2025-01-06", "--forecast-end", "2025-01-13",
         "--output-dir", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "targets: 6" in completed.stdout
    assert "instrument: EUR/USD" in completed.stdout

    base = build_holdout() if module == "holdout" else build_eurusd()
    expected = replace(base, config=replace(
        base.config, forecast_start=date(2025, 1, 6), forecast_end=date(2025, 1, 13),
    ))
    result = expected.run(daily)
    paths = ForecastArtifactWriter(tmp_path / "expected").write(
        result, expected.evaluator().evaluate(result), config=expected.config,
        dataset=DatasetSnapshot.from_frame(source, daily), specification=expected.specification,
        configuration=expected.resolved_configuration,
    ).all()
    assert {p.name: p.read_bytes() for p in paths} == {
        p.name: (output / p.name).read_bytes() for p in paths
    }


def test_feature_and_target_source_columns_and_settings_are_independent():
    daily = daily_sample().rename(columns={"date": "session", "log_rv": "input_variance"})
    daily["outcome_variance"] = daily["input_variance"] + 1.5
    horizon = TargetHorizon(3, "cumulative")
    run = replace(
        standard_experiment(ExperimentConfig(execution=WalkForwardSettings(min_train_size=20)),
                            target_horizon=horizon),
        feature_builder=HARFeatureBuilder(date_col="session", rv_col="input_variance",
                                          weekly_window=2, monthly_window=4),
        target_builder=RealizedVarianceTargetBuilder(horizon, "session", "outcome_variance"),
    )
    prepared = run.prepare(daily)
    assert run.target_builder.horizon == TargetHorizon(3, "cumulative")
    for index, row in enumerate(prepared.itertuples(), start=3):
        assert row.rv_daily == daily.loc[index, "input_variance"]
        expected = np.exp(daily.loc[index + 1:index + 3, "outcome_variance"]).sum()
        assert row.target == pytest.approx(np.log(expected))
        assert row.target_date == daily.loc[index + 3, "session"]

    changed = replace(run, feature_builder=HARFeatureBuilder(
        date_col="session", rv_col="input_variance", weekly_window=3, monthly_window=10,
    ))
    assert changed.specification.target == run.specification.target
    pd.testing.assert_frame_equal(changed.target_builder(daily), run.target_builder(daily))


@pytest.mark.parametrize("arguments, models, candidate, baselines, pairs", [
    (["--models", "HAR", "naive"], ["HAR", "naive"], "HAR", ("naive",), (("naive", "HAR"),)),
    (["--models", "HAR", "EWMA"], ["HAR", "EWMA"], "HAR", ("EWMA",), (("EWMA", "HAR"),)),
    (["--candidate", "naive"], ["historical_mean", "naive", "EWMA", "AR1", "HAR"], "naive",
     ("EWMA", "AR1", "historical_mean", "HAR"), (("EWMA", "naive"),)),
    (["--models", "EWMA", "naive", "--candidate", "naive"], ["EWMA", "naive"], "naive",
     ("EWMA",), (("EWMA", "naive"),)),
    (["--baselines", "EWMA"], ["historical_mean", "naive", "EWMA", "AR1", "HAR"], "HAR",
     ("EWMA",), (("EWMA", "HAR"),)),
])
def test_model_overrides_reconcile_the_supplied_setup_references(
    arguments, models, candidate, baselines, pairs,
):
    original = standard_experiment()
    unchanged = original.resolved_configuration
    run = parse(arguments, original)

    assert [model.name for model in run.models] == models
    assert run.candidate_model == candidate
    assert run.baseline_models == baselines
    assert run.comparison_pairs == pairs
    assert original.resolved_configuration == unchanged


def test_model_selection_preserves_surviving_comparisons_and_their_order():
    original = replace(standard_experiment(), comparison_pairs=(
        ("EWMA", "HAR"), ("AR1", "HAR"), ("naive", "HAR"),
    ))
    run = parse(["--models", "naive", "HAR", "EWMA"], original)

    assert run.comparison_pairs == (("EWMA", "HAR"), ("naive", "HAR"))


def test_an_explicit_empty_comparison_list_remains_disabled_after_selection():
    original = replace(standard_experiment(), comparison_pairs=())

    assert parse(["--candidate", "naive"], original).comparison_pairs == ()
    assert parse(["--models", "HAR", "EWMA"], original).comparison_pairs == ()


class SampleCountCheck:
    specification = ComponentSpecification("sample_count", "test.SampleCountCheck")

    def run(self, context: DiagnosticContext) -> DiagnosticResult:
        return DiagnosticResult(
            self.specification, "Forecast count", ("count",),
            ((context.experiment.n_forecasts,),), (("Count", "{count}"),),
        )


@pytest.mark.parametrize("arguments", [["--models", "EWMA"], ["--baselines"]])
def test_comparison_free_selection_retains_descriptive_and_custom_diagnostics(arguments):
    original = standard_experiment(ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=20),
    ))
    custom = SampleCountCheck()
    original = replace(original, diagnostic_checks=(*original.diagnostic_checks, custom))
    run = parse(arguments, original)

    assert run.baseline_models == ()
    assert run.comparison_pairs == ()
    assert [check.specification.name for check in run.diagnostic_checks] == [
        "monthly_losses", "largest_errors", "sample_count",
    ]
    assert run.diagnostic_checks[-1] is custom
    result = run.run(daily_sample())
    diagnostics = run.robustness_analyzer().analyze(result)
    assert diagnostics.check("sample_count").records() == ({"count": result.n_forecasts},)
    if arguments[0] == "--models":
        assert run.candidate_model == "EWMA"
        assert result.models == ("EWMA",)


def test_single_model_selection_supplies_descriptive_checks_if_all_checks_need_comparisons():
    original = standard_experiment()
    comparison_checks = tuple(check for check in original.diagnostic_checks
                              if check.specification.name in {"hac_sensitivity", "win_rates"})
    run = parse(["--models", "HAR"], replace(original, diagnostic_checks=comparison_checks))

    assert [check.specification.name for check in run.diagnostic_checks] == [
        "monthly_losses", "largest_errors",
    ]


def test_python_setup_can_supply_models_and_metrics_without_a_name_registry():
    class CustomMean(HistoricalMeanModel):
        name = "custom_mean"

    class AbsoluteError:
        name = "MAE"
        scale: ClassVar[ForecastScale] = "log_variance"

        def losses(self, actual: FloatArray, predicted: FloatArray) -> FloatArray:
            return np.abs(actual - predicted)

    model = CustomMean()
    metric = AbsoluteError()
    base = standard_experiment(ExperimentConfig(execution=WalkForwardSettings(min_train_size=20)))
    run = replace(
        base, models=(model,), metrics=(metric,), plot_metric="MAE", candidate_model=model.name,
        baseline_models=(), comparison_pairs=(), diagnostic_checks=(MonthlyLossCheck(),),
    )
    selected = parse(["--models", "custom_mean"], run)
    assert selected.models == (model,)
    assert selected.models[0] is model
    result = selected.run(daily_sample())
    evaluation = selected.evaluator().evaluate(result)

    assert result.models == ("custom_mean",)
    assert {summary.metric for summary in evaluation.summaries} == {"MAE"}
    assert [loss.loss for loss in evaluation.losses] == [
        abs(record.actual_log_rv - record.predicted_log_rv) for record in result.records
    ]
