import numpy as np
import pytest

from rvol.models import (
    LinearRegressionModel,
)


@pytest.mark.parametrize("n_features", [0, 2, 4, 7])
def test_linear_regression_accepts_arbitrary_prepared_feature_counts(n_features):
    features = np.random.default_rng(24).normal(size=(50, n_features))
    coefficients = np.arange(n_features) + 0.5
    target = 2.5 + features @ coefficients

    fitted = LinearRegressionModel().fit(features, target)

    assert fitted.intercept == pytest.approx(2.5)
    np.testing.assert_allclose(fitted.coefficients, coefficients, atol=1e-12)
    np.testing.assert_allclose(fitted.predict(features), target, atol=1e-12)
