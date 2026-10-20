import numpy as np

from rvol.models import (
    LinearPredictor,
)


def test_linear_predictor_copies_coefficients_from_mutable_input():
    coefficients = [2.0, 3.0]
    predictor = LinearPredictor(intercept=1.0, coefficients=coefficients)

    coefficients[0] = 100.0
    coefficients.append(4.0)

    assert predictor.coefficients == (2.0, 3.0)
    np.testing.assert_array_equal(predictor.predict(np.array([[4.0, 5.0]])), [24.0])
