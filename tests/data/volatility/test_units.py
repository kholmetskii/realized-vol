import numpy as np
import pytest

from rvol.data.volatility import annualized_volatility_pct


def test_annualized_volatility_scales_daily_variance_to_a_yearly_percentage():
    # A daily variance of 1e-4 over 250 periods: sqrt(0.025) = 15.8%.
    result = annualized_volatility_pct([1e-4, 0.0], 250)

    np.testing.assert_allclose(result, [100 * np.sqrt(0.025), 0.0])


@pytest.mark.parametrize(("variance", "periods"), [([-1e-4], 250), ([1e-4], 0), ([1e-4], np.nan)])
def test_invalid_inputs_are_rejected(variance, periods):
    with pytest.raises(ValueError):
        annualized_volatility_pct(variance, periods)
