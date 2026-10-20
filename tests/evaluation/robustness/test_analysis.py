import pandas as pd
import pytest

from rvol.evaluation import evaluate
from rvol.evaluation.robustness import (
    HacSensitivity,
    LargestErrors,
    MonthlyLosses,
    WinRates,
    analyze_robustness,
    standard_diagnostics,
)
from tests.evaluation.test_evaluator import sample_forecasts


def test_standard_diagnostics_return_plain_tables_keyed_by_name():
    forecasts = sample_forecasts(80)
    evaluation = evaluate(forecasts)

    tables = analyze_robustness(forecasts, evaluation, standard_diagnostics("candidate"))

    assert set(tables) == {"hac_sensitivity", "monthly_losses", "win_rates", "largest_errors"}
    assert all(isinstance(table, pd.DataFrame) for table in tables.values())
    assert all((table["asset"] == "default").all() for table in tables.values())


def test_without_diagnostics_only_descriptive_tables_are_built():
    forecasts = sample_forecasts(1).query("model == 'candidate'")

    tables = analyze_robustness(forecasts, evaluate(forecasts))

    assert set(tables) == {"monthly_losses", "largest_errors"}
    assert len(tables["largest_errors"]) == 1


def test_a_new_diagnostic_is_a_class_with_a_name_and_run():
    class LossCount:
        name = "loss_count"

        def run(self, forecasts, evaluation):
            forecasts["model"] = "mutated"
            return evaluation.losses.groupby("model").size().rename("n").reset_index()

    forecasts = sample_forecasts()
    evaluation = evaluate(forecasts)
    unchanged = forecasts.copy(deep=True)

    tables = analyze_robustness(forecasts, evaluation, (LossCount(), MonthlyLosses()))

    assert tables["loss_count"].n.tolist() == [20, 20]
    assert set(tables["monthly_losses"].model) == {"baseline", "candidate"}
    pd.testing.assert_frame_equal(forecasts, unchanged)


def test_diagnostics_must_return_tables():
    class NotATable:
        name = "broken"

        def run(self, forecasts, evaluation):
            return [1, 2, 3]

    forecasts = sample_forecasts()
    with pytest.raises(TypeError, match="broken"):
        analyze_robustness(forecasts, evaluate(forecasts), (NotATable(),))


@pytest.mark.parametrize(
    "diagnostics",
    [
        (),
        (LargestErrors(), LargestErrors(top=2)),
        (WinRates("missing"),),
        (WinRates("candidate", ("missing",)),),
        (HacSensitivity("missing"),),
    ],
)
def test_invalid_diagnostic_choices_are_rejected(diagnostics):
    forecasts = sample_forecasts()
    with pytest.raises(ValueError):
        analyze_robustness(forecasts, evaluate(forecasts), diagnostics)


def test_comparison_diagnostics_need_a_baseline():
    forecasts = sample_forecasts().query("model == 'candidate'")
    with pytest.raises(ValueError, match="baseline"):
        analyze_robustness(forecasts, evaluate(forecasts), (WinRates("candidate"),))


def test_diagnostics_reject_an_evaluation_of_different_forecasts():
    forecasts = sample_forecasts()
    evaluation = evaluate(forecasts)
    forecasts.predicted += 1
    with pytest.raises(ValueError):
        analyze_robustness(forecasts, evaluation)
