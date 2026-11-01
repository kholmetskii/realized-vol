import inspect
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rvol import study
from rvol.evaluation import win_rates
from rvol.models import ForecastModel, LinearRegression
from tests.forecasting_helpers import with_semivariances

START, END = date(2024, 10, 1), date(2026, 9, 30)
EVALUATION_START = date(2025, 1, 1)


def test_forecasting_and_study_imports_do_not_load_matplotlib():
    code = """
import sys
import pandas as pd
from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.evaluation import forecast_losses
from rvol.models import ForecastModel, LinearRegression
from rvol.walkforward import walk_forward
from rvol import study
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
for prefix in ('rvol.data', 'rvol.evaluation', 'rvol.study', 'matplotlib', 'scipy'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


def daily_series(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-03", periods=n),
            "log_rv": -10 + np.sin(np.arange(n) / 4) / 5 + rng.normal(0, 0.3, n),
        }
    )
    return with_semivariances(daily)


def study_forecasts(models=None) -> pd.DataFrame:
    models = study.benchmark_models() if models is None else models
    prepared = study.prepare(daily_series())
    return study.forecast(prepared, models, start=START, end=END, asset="test")


@pytest.mark.parametrize("custom_model", [False, True])
def test_compare_scores_smeared_variance_forecasts_on_the_same_dates(custom_model):
    class CustomMean:
        def fit(self, features, target):
            self.mean = float(np.mean(target))
            return self

        def predict(self, features):
            return np.full(len(features), self.mean)

    models = study.benchmark_models()
    if custom_model:
        models = (*models[1:], ForecastModel("custom_mean", CustomMean()))

    comparison = study.compare(study_forecasts(models), evaluation_start=EVALUATION_START)

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
    comparison = study.compare(study_forecasts(), evaluation_start=EVALUATION_START)

    table = study.results_table(comparison)

    summary = comparison.summary.set_index("model")
    pd.testing.assert_series_equal(
        table[study.LOSS_LABEL], summary.mean_loss.loc[table.index], check_names=False
    )
    assert table[study.LOSS_LABEL].is_monotonic_increasing
    assert table.loc["HAR", "HAR lower by"] == 0
    np.testing.assert_allclose(table["DM p-value"], summary.p_value.loc[table.index])
    wins = win_rates(comparison.losses, "HAR").set_index("baseline_model").candidate_win_rate
    np.testing.assert_allclose(table["HAR wins"].drop("HAR"), wins.loc[table.index.drop("HAR")])
    assert pd.isna(table.loc["HAR", "HAR wins"])


def test_compare_rejects_a_missing_candidate_or_an_empty_window():
    forecasts = study_forecasts(study.benchmark_models()[:2])
    with pytest.raises(ValueError, match="candidate"):
        study.compare(forecasts, evaluation_start=EVALUATION_START)
    with pytest.raises(ValueError, match="evaluation_start"):
        study.compare(forecasts, evaluation_start=date(2030, 1, 1), candidate="naive")


def test_forecast_requires_at_least_one_model():
    with pytest.raises(ValueError, match="at least one model"):
        study.forecast(study.prepare(daily_series()), (), start=START, end=END)


def test_the_ridge_penalty_is_the_calibration_minimum():
    prepared = study.prepare(daily_series())

    scores = study.ridge_calibration(prepared)
    models = study.study_models(prepared)

    assert scores.index.tolist() == list(study.RIDGE_GRID)
    assert np.isfinite(scores).all()
    ridge = models[-1]
    assert ridge.name == "HAR-ridge" and ridge.regressors == study.HAR_FEATURES
    assert isinstance(ridge.rule, LinearRegression) and ridge.rule.ridge == scores.idxmin()
    assert [model.name for model in models[:-1]] == [
        model.name for model in study.benchmark_models()
    ]


def test_the_ridge_calibration_never_sees_the_evaluation_period():
    daily = daily_series()
    changed = daily.copy()
    later = changed["date"] >= pd.Timestamp(EVALUATION_START)
    changed.loc[later, ["log_rv", "log_rs_pos", "log_rs_neg"]] += 2.0

    before = study.ridge_calibration(study.prepare(daily))
    after = study.ridge_calibration(study.prepare(changed))

    pd.testing.assert_series_equal(before, after)


def test_ridge_calibration_needs_a_penalty():
    with pytest.raises(ValueError, match="penalty"):
        study.ridge_calibration(study.prepare(daily_series()), ())


def specification(models):
    """What defines each model before fitting: name, rule, its settings, regressors.

    Settings are the rule's constructor arguments, so fitting a model (which adds
    estimated attributes) does not change its specification.
    """
    return [
        (
            m.name,
            type(m.rule),
            {name: getattr(m.rule, name) for name in inspect.signature(type(m.rule)).parameters},
            m.regressors,
        )
        for m in models
    ]


def test_every_study_choice_is_made_before_the_evaluation_period():
    prepared = study.prepare(daily_series())
    before = prepared.loc[prepared["target_date"] < pd.Timestamp(study.EVALUATION_START)]

    assert len(before) < len(prepared)
    assert specification(study.study_models(before)) == specification(study.study_models(prepared))
