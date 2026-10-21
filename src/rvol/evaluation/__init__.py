"""Score forecasts and compare models: losses, the Diebold-Mariano test, robustness checks.

The package scores whatever forecasts it receives and knows nothing about what
they forecast. Every function after forecast_losses works on its losses table.
"""

from rvol.evaluation.diebold_mariano import default_hac_lags, diebold_mariano
from rvol.evaluation.qlike import qlike
from rvol.evaluation.robustness import hac_sensitivity, win_rates
from rvol.evaluation.scoring import Loss, compare_models, forecast_losses

__all__ = [
    "Loss",
    "compare_models",
    "default_hac_lags",
    "diebold_mariano",
    "forecast_losses",
    "hac_sensitivity",
    "qlike",
    "win_rates",
]
