import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rvol.domain import (
    ComponentSpecification,
    ExperimentConfig,
    RunIdentity,
    TargetHorizon,
    WalkForwardSettings,
)
from rvol.forecasting.composition import standard_diagnostic_checks
from rvol.forecasting.evaluation import (
    DiagnosticCheck,
    DiagnosticResult,
    DieboldMarianoTest,
    ForecastEvaluator,
    ForecastRobustnessAnalyzer,
    LagSensitivityCheck,
    LargestErrorsCheck,
    MonthlyLossCheck,
    WinRateCheck,
)
from rvol.forecasting.execution import WalkForwardExperiment
from rvol.forecasting.models import (
    AR1Forecaster,
    EWMAForecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)
from rvol.presentation.reporting import RobustnessArtifactWriter


def experiment_result():
    n_rows = 55
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n_rows + 1)
    matrix = rng.normal(loc=-10.0, scale=0.5, size=(n_rows, 3))
    target = -0.3 + matrix @ np.array([0.5, 0.3, 0.17])
    features = pd.DataFrame({
        "origin_date": dates[:-1], "target_date": dates[1:],
        "rv_daily": matrix[:, 0], "rv_weekly": matrix[:, 1], "rv_monthly": matrix[:, 2],
        "target": target,
    })
    return WalkForwardExperiment(
        models=(HistoricalMeanForecaster(), NaiveForecaster(), EWMAForecaster(),
                AR1Forecaster(), HARForecaster()),
        config=ExperimentConfig(execution=WalkForwardSettings(min_train_size=20)),
    ).run(features)


def analyzer():
    return ForecastRobustnessAnalyzer(
        checks=standard_diagnostic_checks(hac_lags=(0, 1), top_errors=2),
        candidate_model="HAR", baseline_models=("naive", "EWMA", "AR1", "historical_mean"),
    )


def test_robustness_analyzer_calculates_all_predefined_checks():
    result = analyzer().analyze(experiment_result())
    tables = {item.name: pd.DataFrame(item.rows, columns=item.columns) for item in result.checks}

    assert len(tables["hac_sensitivity"]) == 16
    assert set(tables["hac_sensitivity"].hac_lags) == {0, 1}
    wins = tables["win_rates"]
    assert len(wins) == 8
    assert set(wins.n_obs) == {35}
    assert wins.candidate_win_rate.between(0, 1).all()
    assert (wins.candidate_wins + wins.ties <= wins.n_obs).all()
    assert len(tables["monthly_losses"]) >= 8
    assert set(tables["monthly_losses"].metric) == {"QLIKE", "log-RV MSE"}
    assert len(tables["largest_errors"]) == 10
    assert set(tables["largest_errors"].model) == {"AR1", "EWMA", "HAR", "historical_mean", "naive"}


def test_robustness_reuses_supplied_evaluation_without_recalculating_losses(monkeypatch):
    experiment = experiment_result()
    analysis = analyzer()
    expected = analysis.analyze(experiment)
    evaluation = analysis.evaluator.evaluate(experiment)

    def unexpected_evaluation(self, experiment):
        raise AssertionError("supplied losses must be reused, including in lag sensitivity")

    monkeypatch.setattr(ForecastEvaluator, "evaluate", unexpected_evaluation)
    assert analysis.analyze(experiment, evaluation=evaluation) == expected


def test_lag_sensitivity_matches_independent_evaluations_for_each_test_setting():
    experiment = experiment_result()
    analysis = analyzer()
    table = analysis.analyze(experiment).check("hac_sensitivity")
    for lag in (0, 1):
        evaluation = ForecastEvaluator(
            comparison_pairs=tuple((baseline, "HAR") for baseline in analysis.baseline_models),
            comparison_test=DieboldMarianoTest(lag),
        ).evaluate(experiment)
        actual = {
            (row["metric"], row["baseline_model"], row["candidate_model"]): row
            for row in table.records() if row["hac_lags"] == lag
        }
        for comparison in evaluation.comparisons:
            row = actual[(comparison.metric, comparison.baseline_model, comparison.candidate_model)]
            assert row["n_obs"] == comparison.n_obs
            assert row["baseline_mean_loss"] == comparison.baseline_mean_loss
            assert row["candidate_mean_loss"] == comparison.candidate_mean_loss
            assert row["mean_loss_difference"] == comparison.mean_loss_difference
            assert row["statistic"] == comparison.statistic
            assert row["p_value"] == comparison.p_value
            assert row["test_parameters"] == json.dumps(
                dict(comparison.test_parameters), sort_keys=True,
            )


@pytest.mark.parametrize("change, message", [
    (lambda result: replace(result, identity=RunIdentity("other")), "identity"),
    (lambda result: replace(result, target_horizon=TargetHorizon(3, "cumulative")), "target"),
    (lambda result: replace(result, instrument="GBP/USD"), "instrument"),
    (lambda result: replace(result, losses=result.losses[:-1]), "one loss per forecast"),
    (lambda result: replace(result, losses=(*result.losses, result.losses[0])),
     "one loss per forecast"),
    (lambda result: replace(result, losses=(
        replace(result.losses[0], metric="other"), *result.losses[1:],
    )), "configured metric"),
    (lambda result: replace(result, losses=(
        replace(result.losses[0], origin_date=None), *result.losses[1:],
    )), "one loss per forecast"),
    (lambda result: replace(result, losses=(
        replace(result.losses[0], loss=-1.0), *result.losses[1:],
    )), "finite and non-negative"),
    (lambda result: replace(result, losses=(
        replace(result.losses[0], loss=float("nan")), *result.losses[1:],
    )), "finite and non-negative"),
])
def test_robustness_rejects_evaluation_that_does_not_match_the_experiment(change, message):
    experiment = experiment_result()
    analysis = analyzer()
    evaluation = analysis.evaluator.evaluate(experiment)
    with pytest.raises(ValueError, match=message):
        analysis.analyze(experiment, evaluation=change(evaluation))


def test_robustness_writer_is_deterministic_and_records_selected_check_settings(tmp_path):
    result = analyzer().analyze(experiment_result())
    writer = RobustnessArtifactWriter(tmp_path)
    paths = writer.write(result)
    first_run = {path.name: path.read_bytes() for path in paths.all()}
    repeated = writer.write(result)

    assert {path.name: path.read_bytes() for path in repeated.all()} == first_run
    assert len(pd.read_csv(paths.csv("hac_sensitivity"))) == 16
    assert len(pd.read_csv(paths.csv("win_rates"))) == 8
    assert len(pd.read_csv(paths.csv("largest_errors"))) == 10
    report = paths.report.read_text()
    assert "## Newey–West lag sensitivity" in report
    assert "## Monthly forecast losses" in report
    metadata = json.loads(paths.metadata.read_text())
    specs = {item["specification"]["name"]: item for item in metadata["checks"]}
    assert specs["largest_errors"]["specification"]["parameters"] == {"top_errors": 2}
    assert specs["hac_sensitivity"]["components"]["lag_1"]["parameters"]["hac_lags"] == 1


def test_descriptive_checks_work_on_one_session_without_comparison_tests(tmp_path):
    experiment = experiment_result()
    short = replace(experiment, records=experiment.records[:5])
    result = ForecastRobustnessAnalyzer(
        checks=(MonthlyLossCheck(), LargestErrorsCheck(2)),
    ).analyze(short)
    paths = RobustnessArtifactWriter(tmp_path).write(result)

    assert len(result.check("monthly_losses").rows) == 10
    assert len(result.check("largest_errors").rows) == 5
    assert {path.name for path in paths.all()} == {
        "monthly_losses.csv", "largest_errors.csv", "robustness.md", "robustness.json",
    }
    assert "lag sensitivity" not in paths.report.read_text()


def test_independent_diagnostic_can_be_injected_and_rendered_without_writer_changes(tmp_path):
    class CountCheck:
        specification = ComponentSpecification("counts", "test.CountCheck")

        def run(self, context):
            return DiagnosticResult(
                self.specification, "Forecast counts", ("count",),
                ((context.experiment.n_forecasts,),), (("Count", "{count}"),),
            )

    check = CountCheck()
    assert isinstance(check, DiagnosticCheck)
    result = ForecastRobustnessAnalyzer(checks=(check,)).analyze(experiment_result())
    paths = RobustnessArtifactWriter(tmp_path).write(result)

    assert pd.read_csv(paths.csv("counts")).loc[0, "count"] == 175
    assert "## Forecast counts" in paths.report.read_text()
    assert len(paths.diagnostics) == 1


@pytest.mark.parametrize("lags", [(), (-1,), (1, 1), (True,), (0.5,)])
def test_lag_check_validates_its_own_configuration(lags):
    with pytest.raises(ValueError, match="hac_lags"):
        LagSensitivityCheck(lags)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_error_check_validates_its_own_configuration(count):
    with pytest.raises(ValueError, match="top_errors"):
        LargestErrorsCheck(count)


def test_diagnostic_selection_rejects_empty_and_duplicate_checks():
    with pytest.raises(ValueError, match="non-empty"):
        ForecastRobustnessAnalyzer(checks=())
    with pytest.raises(ValueError, match="unique"):
        ForecastRobustnessAnalyzer(checks=(MonthlyLossCheck(), MonthlyLossCheck()))
    with pytest.raises(ValueError, match="unknown"):
        standard_diagnostic_checks(("missing",))


def test_comparison_diagnostics_validate_their_required_model_references():
    with pytest.raises(ValueError, match="candidate and baselines"):
        ForecastRobustnessAnalyzer(checks=(WinRateCheck(),)).analyze(experiment_result())
    with pytest.raises(ValueError, match="unknown models"):
        ForecastRobustnessAnalyzer(
            checks=(WinRateCheck(),), candidate_model="missing", baseline_models=("naive",),
        ).analyze(experiment_result())


def test_invalid_display_template_fails_before_writing(tmp_path):
    result = analyzer().analyze(experiment_result())
    invalid = replace(result.checks[0], report_columns=(("bad", "{missing}"),))
    output = tmp_path / "invalid"
    with pytest.raises(KeyError):
        RobustnessArtifactWriter(output).write(replace(result, checks=(invalid,)))
    assert not output.exists()
