from typing import cast

import numpy as np
import pandas as pd
import pytest

from rvol.evaluation import analyze_robustness, evaluate
from rvol.evaluation.comparison import diebold_mariano
from tests.test_evaluator import sample_forecasts


def test_diagnostics_use_existing_losses_and_return_plain_tables():
    forecasts = sample_forecasts(80)
    evaluation = evaluate(forecasts)
    tables = analyze_robustness(
        forecasts, evaluation, candidate_model="candidate", hac_lags=(0, 1, 3), top_errors=2
    )
    assert set(tables) == {"hac_sensitivity", "monthly_losses", "win_rates", "largest_errors"}
    assert all(isinstance(table, pd.DataFrame) for table in tables.values())
    wins = tables["win_rates"].iloc[0]
    assert wins.candidate_wins == wins.n_obs == 80
    assert wins.ties == 0 and wins.candidate_win_rate == 1
    errors = tables["largest_errors"]
    assert errors.groupby("model").size().to_dict() == {"baseline": 2, "candidate": 2}
    np.testing.assert_allclose(errors.absolute_error, (errors.actual - errors.predicted).abs())
    for row in tables["hac_sensitivity"].itertuples():
        losses = evaluation.losses.pivot(index="target_date", columns="model", values="loss")
        expected = diebold_mariano(
            losses.baseline, losses.candidate, hac_lags=cast(int, row.hac_lags)
        )
        assert row.statistic == expected.statistic and row.p_value == expected.p_value
    for row in tables["monthly_losses"].itertuples():
        subset = evaluation.losses.loc[
            (evaluation.losses.model == row.model)
            & (evaluation.losses.target_date.dt.strftime("%Y-%m") == row.month)
        ]
        assert row.n_obs == len(subset)
        assert row.mean_loss == pytest.approx(subset.loss.mean())


def test_a_short_single_model_run_needs_only_descriptive_checks():
    forecasts = sample_forecasts(1).query("model == 'candidate'")
    tables = analyze_robustness(forecasts, evaluate(forecasts))
    assert set(tables) == {"monthly_losses", "largest_errors"}
    assert len(tables["largest_errors"]) == 1


def test_ties_are_not_counted_as_wins():
    forecasts = sample_forecasts()
    forecasts["predicted"] = forecasts.actual + 0.25
    tables = analyze_robustness(
        forecasts, evaluate(forecasts), candidate_model="candidate", checks=("win_rates",)
    )
    row = tables["win_rates"].iloc[0]
    assert row.ties == 20 and row.candidate_wins == row.candidate_win_rate == 0


@pytest.mark.parametrize(
    "options",
    [
        {"checks": ()},
        {"checks": ("unknown",)},
        {"checks": ("largest_errors", "largest_errors")},
        {"checks": ("win_rates",)},
        {"top_errors": 0},
        {"top_errors": True},
        {"candidate_model": "missing"},
        {"candidate_model": "candidate", "baseline_models": ("candidate",)},
        {"candidate_model": "candidate", "baseline_models": ("missing",)},
        {"candidate_model": "candidate", "baseline_models": ("baseline", "baseline")},
        {"tie_rtol": -1},
        {"tie_atol": float("nan")},
        {"candidate_model": "candidate", "hac_lags": ()},
        {"candidate_model": "candidate", "hac_lags": (0, 0)},
    ],
)
def test_invalid_diagnostic_choices_are_rejected(options):
    forecasts = sample_forecasts()
    with pytest.raises(ValueError):
        analyze_robustness(forecasts, evaluate(forecasts), **options)
