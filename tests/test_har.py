import numpy as np
import pandas as pd

from rvol.models.har import HarModel, fit_har


def synthetic_regression(n: int = 100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, 3))
    target = 0.7 + matrix @ np.array([0.5, 0.3, 0.1])
    return pd.DataFrame({
        "rv_daily": matrix[:, 0],
        "rv_weekly": matrix[:, 1],
        "rv_monthly": matrix[:, 2],
        "target": target,
    })


def test_har_recovers_known_linear_coefficients():
    model = fit_har(synthetic_regression())

    assert np.isclose(model.intercept, 0.7)
    assert np.allclose(model.coefficients, [0.5, 0.3, 0.1])


def test_fit_har_returns_the_arithmetic_variance_forecast():
    features = pd.DataFrame({
        "rv_daily": np.zeros(4),
        "rv_weekly": np.zeros(4),
        "rv_monthly": np.zeros(4),
        "target": np.log([1.0, 9.0, 1.0, 9.0]),
    })

    model = fit_har(features)

    np.testing.assert_allclose(np.exp(model.predict(features.iloc[:1])), [5.0])


def test_har_predicts_multiple_rows():
    model = HarModel(intercept=1.0, daily=0.5, weekly=0.25, monthly=0.1)
    features = synthetic_regression(n=4)

    predicted = model.predict(features)
    expected = 1.0 + features[["rv_daily", "rv_weekly", "rv_monthly"]].to_numpy() @ np.array(
        [0.5, 0.25, 0.1]
    )

    assert np.allclose(predicted, expected)
