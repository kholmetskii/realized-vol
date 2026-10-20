import numpy as np

from rvol.models import (
    NaiveModel,
)


def test_naive_model_returns_the_current_daily_feature():
    training = np.array([[-11.0], [-10.0], [-9.0]])
    fitted = NaiveModel().fit(training, np.array([-10.0, -9.0, -8.0]))
    future = np.array([[-8.5], [-8.0]])

    predicted = fitted.predict(future)

    np.testing.assert_array_equal(predicted, future[:, 0])
    assert not np.shares_memory(predicted, future)
