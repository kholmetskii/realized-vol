import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rvol import experiment
from rvol.evaluation import win_rates
from rvol.models import ForecastModel

START, END = date(2024, 10, 1), date(2026, 9, 30)
EVALUATION_START = date(2025, 1, 1)


def test_forecasting_and_experiment_imports_do_not_load_matplotlib():
    code = """
import sys
import pandas as pd
from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.evaluation import forecast_losses
from rvol.models import ForecastModel, LinearRegression
from rvol.walkforward import walk_forward
from rvol import experiment
data = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=20), 'value': range(20)})
features = pd.DataFrame({'origin_date': data.date, 'lag_0': data.value.astype(float)})
prepared = prepare_forecast_data(features, build_next_observation_targets(data))
model = ForecastModel('AR1', LinearRegression(), ('lag_0',))
forecasts = walk_forward(prepared, model, min_train_size=5)
forecast_losses(forecasts, lambda actual, predicted: (actual - predicted) ** 2)
assert 'matplotlib' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


def test_importing_models_and_walk_forward_loads_no_preparation_or_evaluation():
    code = """
import sys
from rvol.walkforward import walk_forward
from rvol.models import ForecastModel, LinearRegression
for prefix in ('rvol.data', 'rvol.evaluation', 'rvol.experiment', 'matplotlib', 'scipy'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


def daily_series(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-03", periods=n),
            "log_rv": -10 + np.sin(np.arange(n) / 4) / 5 + rng.normal(0, 0.3, n),
        }
    )


def study_forecasts(models=None) -> pd.DataFrame:
    models = experiment.benchmark_models() if models is None else models
    prepared = experiment.prepare(daily_series())
    return experiment.forecast(prepared, models, start=START, end=END, asset="test")


@pytest.mark.parametrize("custom_model", [False, True])
def test_compare_scores_smeared_variance_forecasts_on_the_same_dates(custom_model):
    class CustomMean:
        def fit(self, features, target):
            self.mean = float(np.mean(target))
            return self

        def predict(self, features):
            return np.full(len(features), self.mean)

    models = experiment.benchmark_models()
    if custom_model:
        models = (*models[1:], ForecastModel("custom_mean", CustomMean()))

    comparison = experiment.compare(study_forecasts(models), evaluation_start=EVALUATION_START)

    names = {model.name for model in models}
    assert set(comparison.summary.model) == names
    assert set(comparison.losses.model) == names
    assert comparison.summary.set_index("model").loc["HAR", ["p_value"]].isna().all()
    variance = comparison.forecasts
    assert variance.target_date.min() >= pd.Timestamp(EVALUATION_START)
    assert variance.groupby("model").target_date.apply(tuple).nunique() == 1
    assert (variance.smearing_factor > 0).all()
    assert (variance.actual > 0).all() and (variance.predicted > 0).all()
    assert set(variance.series) == {"test"}


def test_results_table_summarises_the_comparison():
    comparison = experiment.compare(study_forecasts(), evaluation_start=EVALUATION_START)

    table = experiment.results_table(comparison)

    summary = comparison.summary.set_index("model")
    pd.testing.assert_series_equal(
        table["QLIKE"], summary.mean_loss.loc[table.index], check_names=False
    )
    assert table["QLIKE"].is_monotonic_increasing
    assert table.loc["HAR", "HAR lower by"] == 0
    np.testing.assert_allclose(table["DM p-value"], summary.p_value.loc[table.index])
    wins = win_rates(comparison.losses, "HAR").set_index("baseline_model").candidate_win_rate
    np.testing.assert_allclose(table["HAR wins"].drop("HAR"), wins.loc[table.index.drop("HAR")])
    assert pd.isna(table.loc["HAR", "HAR wins"])


def test_compare_rejects_a_missing_candidate_or_an_empty_window():
    forecasts = study_forecasts(experiment.benchmark_models()[:2])
    with pytest.raises(ValueError, match="candidate"):
        experiment.compare(forecasts, evaluation_start=EVALUATION_START)
    with pytest.raises(ValueError, match="evaluation_start"):
        experiment.compare(forecasts, evaluation_start=date(2030, 1, 1), candidate="naive")


def test_forecast_requires_at_least_one_model():
    with pytest.raises(ValueError, match="at least one model"):
        experiment.forecast(experiment.prepare(daily_series()), (), start=START, end=END)
