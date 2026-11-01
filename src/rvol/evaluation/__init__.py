"""Score forecasts and compare models: losses, the Diebold-Mariano test, robustness checks.

The package scores whatever forecasts it receives and knows nothing about what
they forecast. Every function after forecast_losses works on its losses table;
``StudyDesign`` fixes a comparison's dates, loss and candidate and runs it on a
forecasting table.
"""

from rvol.evaluation.diebold_mariano import default_hac_lags, diebold_mariano, hac_lag_grid
from rvol.evaluation.evaluation import Evaluation, evaluate, results_table
from rvol.evaluation.qlike import qlike
from rvol.evaluation.robustness import hac_sensitivity, win_rates
from rvol.evaluation.scoring import Loss, compare_models, forecast_losses
from rvol.evaluation.study_design import StudyDesign
from rvol.evaluation.study_run import StudyRun

__all__ = [
    "Evaluation",
    "Loss",
    "StudyDesign",
    "StudyRun",
    "compare_models",
    "default_hac_lags",
    "diebold_mariano",
    "evaluate",
    "forecast_losses",
    "hac_lag_grid",
    "hac_sensitivity",
    "qlike",
    "results_table",
    "win_rates",
]
