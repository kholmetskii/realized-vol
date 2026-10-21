import numpy as np
import pytest

from rvol.evaluation import qlike


def losses(actual, predicted):
    return qlike(np.array(actual), np.array(predicted))


def test_perfect_forecasts_have_zero_loss():
    values = np.array([0.01, 0.02, 0.04])

    assert np.allclose(qlike(values, values), 0.0)


def test_qlike_matches_its_formula():
    assert losses([1.0], [2.0])[0] == pytest.approx(0.5 - np.log(0.5) - 1)


def test_underprediction_costs_more_than_reciprocal_overprediction():
    under = losses([1.0], [0.5])[0]
    over = losses([1.0], [2.0])[0]

    assert under > over > 0


@pytest.mark.parametrize("actual, predicted", [([0.0], [1.0]), ([1.0], [-1.0])])
def test_nonpositive_values_are_rejected(actual, predicted):
    with pytest.raises(ValueError, match="strictly positive"):
        losses(actual, predicted)
