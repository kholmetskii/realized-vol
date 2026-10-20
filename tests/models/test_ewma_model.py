import numpy as np
import pytest

from rvol.models import (
    EWMAModel,
)


def test_ewma_model_applies_fixed_decay_to_supplied_targets():
    target = np.array([1.0, 4.0, 9.0])
    fitted = EWMAModel(decay=0.5).fit(np.empty((3, 0)), target)

    predicted = fitted.predict(np.empty((2, 0)))

    np.testing.assert_allclose(predicted, [5.75, 5.75])


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, float("nan")])
def test_ewma_model_rejects_invalid_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        EWMAModel(decay=decay)
