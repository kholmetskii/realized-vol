from typing import cast

import pytest

from rvol.evaluation import ComparisonOutcome, diebold_mariano, evaluate
from rvol.evaluation.robustness import HacSensitivity, analyze_robustness
from tests.evaluation.test_evaluator import sample_forecasts


def test_each_lag_repeats_the_dm_test_on_existing_losses():
    forecasts = sample_forecasts(80)
    evaluation = evaluate(forecasts)

    table = analyze_robustness(
        forecasts, evaluation, (HacSensitivity("candidate", lags=(0, 1, 3)),)
    )["hac_sensitivity"]

    assert table.hac_lags.tolist() == [0, 1, 3]
    losses = evaluation.losses.pivot(index="target_date", columns="model", values="loss")
    for row in table.itertuples():
        expected = diebold_mariano(
            losses.baseline, losses.candidate, hac_lags=cast(int, row.hac_lags)
        )
        assert row.statistic == expected.statistic and row.p_value == expected.p_value
        assert row.test_name == "DM"


def test_the_sensitivity_check_uses_the_supplied_test_family():
    class LaggedTest:
        name = "lagged"

        def __init__(self, lags: int) -> None:
            self.lags = lags

        def compare(self, baseline_loss, candidate_loss):
            return ComparisonOutcome(float(self.lags), 0.5, {"hac_lags": self.lags})

    forecasts = sample_forecasts()
    table = analyze_robustness(
        forecasts,
        evaluate(forecasts),
        (HacSensitivity("candidate", lags=(2, 4), make_test=LaggedTest),),
    )["hac_sensitivity"]

    assert table.test_name.tolist() == ["lagged", "lagged"]
    assert table.statistic.tolist() == [2.0, 4.0]


@pytest.mark.parametrize("lags", [(), (0, 0), (-1,), (1.5,), (True,)])
def test_invalid_lags_are_rejected_on_construction(lags):
    with pytest.raises(ValueError, match="lags"):
        HacSensitivity("candidate", lags=lags)
