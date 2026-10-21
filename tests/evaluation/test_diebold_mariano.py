import numpy as np
import pytest

from rvol.evaluation import default_hac_lags, diebold_mariano


def test_a_clearly_lower_candidate_loss_is_detected():
    rng = np.random.default_rng(31)
    candidate = np.square(rng.normal(0.0, 0.15, 120))
    baseline = candidate + np.square(rng.normal(0.0, 0.45, 120))

    statistic, p_value = diebold_mariano(baseline, candidate, hac_lags=3)

    assert statistic > 0
    assert p_value < 0.01


def test_identical_loss_sequences_give_no_evidence_of_a_difference():
    losses = np.linspace(0.1, 1.0, 20)

    assert diebold_mariano(losses, losses) == (0.0, 0.5)


def test_zero_lag_statistic_matches_the_iid_formula():
    baseline = np.array([0.8, 1.2, 0.7, 1.5, 0.9, 1.1])
    candidate = np.array([0.6, 0.9, 0.8, 1.0, 0.7, 0.8])
    differential = baseline - candidate
    expected = differential.mean() / (differential.std(ddof=0) / np.sqrt(len(differential)))

    statistic, _ = diebold_mariano(baseline, candidate, hac_lags=0)

    assert np.isclose(statistic, expected)


@pytest.mark.parametrize("n_obs, lags", [(3, 1), (20, 2), (100, 4), (453, 5)])
def test_default_lags_follow_the_newey_west_rule(n_obs, lags):
    assert default_hac_lags(n_obs) == lags


@pytest.mark.parametrize("hac_lags", [-1, 19])
def test_lags_outside_the_sample_are_rejected(hac_lags):
    losses = np.linspace(0.1, 1.0, 20)
    with pytest.raises(ValueError, match="hac_lags"):
        diebold_mariano(losses, losses / 2, hac_lags=hac_lags)


def test_short_or_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(5), np.ones(6))
