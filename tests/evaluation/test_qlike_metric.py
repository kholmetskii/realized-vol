import numpy as np
import pytest

from rvol.evaluation import QLikeMetric


def qlike(actual, predicted):
    return QLikeMetric().losses(np.array(actual), np.array(predicted))


def test_perfect_forecasts_have_zero_qlike():
    values = np.array([0.01, 0.02, 0.04])

    assert np.allclose(qlike(values, values), 0.0)


def test_qlike_matches_its_formula():
    assert qlike([1.0], [2.0])[0] == pytest.approx(0.5 - np.log(0.5) - 1)


def test_qlike_penalizes_underprediction_more_than_reciprocal_overprediction():
    under = qlike([1.0], [0.5])[0]
    over = qlike([1.0], [2.0])[0]

    assert under > over > 0


@pytest.mark.parametrize("actual, predicted", [([0.0], [1.0]), ([1.0], [-1.0])])
def test_qlike_rejects_nonpositive_values(actual, predicted):
    with pytest.raises(ValueError, match="strictly positive"):
        qlike(actual, predicted)
