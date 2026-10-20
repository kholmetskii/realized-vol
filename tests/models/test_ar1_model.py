import numpy as np

from rvol.models import (
    AR1Model,
    LinearPredictor,
)


def test_ar1_recovers_known_coefficients():
    daily = np.linspace(-12.0, -8.0, 30).reshape(-1, 1)
    target = 0.7 + 0.6 * daily[:, 0]

    fitted = AR1Model().fit(daily, target)

    assert isinstance(fitted, LinearPredictor)
    assert np.isclose(fitted.intercept, 0.7)
    np.testing.assert_allclose(fitted.coefficients, [0.6])
