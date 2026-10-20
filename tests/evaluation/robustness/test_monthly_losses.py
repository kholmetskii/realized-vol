import pytest

from rvol.evaluation import evaluate
from rvol.evaluation.robustness import MonthlyLosses, analyze_robustness
from tests.evaluation.test_evaluator import sample_forecasts


def test_monthly_means_match_the_evaluated_losses():
    forecasts = sample_forecasts(80)
    evaluation = evaluate(forecasts)

    table = analyze_robustness(forecasts, evaluation, (MonthlyLosses(),))["monthly_losses"]

    assert set(table.month) == {"2024-01", "2024-02", "2024-03", "2024-04"}
    for row in table.itertuples():
        subset = evaluation.losses.loc[
            (evaluation.losses.model == row.model)
            & (evaluation.losses.target_date.dt.strftime("%Y-%m") == row.month)
        ]
        assert row.n_obs == len(subset)
        assert row.mean_loss == pytest.approx(subset.loss.mean())
