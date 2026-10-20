import numpy as np

from rvol.models import (
    HistoricalMeanModel,
)


def test_historical_mean_predicts_the_training_target_mean():
    target = np.array([-11.0, -10.0, -9.0])
    fitted = HistoricalMeanModel().fit(np.empty((3, 0)), target)

    predicted = fitted.predict(np.empty((2, 0)))

    np.testing.assert_allclose(predicted, [-10.0, -10.0])
