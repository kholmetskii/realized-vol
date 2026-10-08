import numpy as np
import pytest

from rvol.domain import ComparisonOutcome
from rvol.forecasting.evaluation import DieboldMarianoTest
from rvol.forecasting.evaluation.comparison import diebold_mariano


def test_comparison_detects_lower_candidate_loss():
    n = 120
    rng = np.random.default_rng(31)
    candidate = np.square(rng.normal(0.0, 0.15, n))
    baseline = candidate + np.square(rng.normal(0.0, 0.45, n))

    result = diebold_mariano(
        baseline_loss=baseline,
        candidate_loss=candidate,
        hac_lags=3,
    )

    assert result.mean_loss_difference > 0
    assert result.statistic > 0
    assert result.p_value < 0.01
    assert result.hac_lags == 3


def test_identical_loss_sequences_give_no_evidence_of_a_difference():
    losses = np.linspace(0.1, 1.0, 20)

    result = diebold_mariano(losses, losses)

    assert result.mean_loss_difference == 0
    assert result.statistic == 0
    assert result.p_value == 0.5


def test_zero_lag_statistic_matches_the_iid_formula():
    baseline = np.array([0.8, 1.2, 0.7, 1.5, 0.9, 1.1])
    candidate = np.array([0.6, 0.9, 0.8, 1.0, 0.7, 0.8])
    differential = baseline - candidate
    expected = differential.mean() / (differential.std(ddof=0) / np.sqrt(len(differential)))

    result = diebold_mariano(baseline, candidate, hac_lags=0)

    assert np.isclose(result.statistic, expected)


def test_configured_dm_adapter_preserves_the_existing_formula_and_records_resolved_lags():
    baseline = np.linspace(0.1, 1.0, 20)
    candidate = baseline / 2
    test = DieboldMarianoTest()
    reference = diebold_mariano(baseline, candidate)

    outcome = test.compare(baseline, candidate)

    assert outcome.statistic == reference.statistic
    assert outcome.p_value == reference.p_value
    assert dict(outcome.parameters) == {"hac_lags": reference.hac_lags}
    assert dict(test.specification.parameters)["hac_lags"] is None


@pytest.mark.parametrize("statistic, p_value", [(float("nan"), 0.5), (0, -0.1), (0, 1.1)])
def test_comparison_outcomes_reject_invalid_values(statistic, p_value):
    with pytest.raises(ValueError):
        ComparisonOutcome(statistic, p_value)
