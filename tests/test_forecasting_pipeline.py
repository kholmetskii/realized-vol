import subprocess
import sys
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from rvol import experiment
from rvol.backtesting import WalkForwardPlanner, WalkForwardRunner
from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.data.volatility.features import VarianceFeatureBuilder
from rvol.evaluation import evaluate
from tests.forecasting_helpers import benchmark_models, daily_sample, prepared_variance


def test_changing_future_data_cannot_change_earlier_predictions():
    daily = daily_sample()
    changed = daily.copy()
    changed.loc[65:, "log_rv"] += 1
    cutoff = daily.date.iloc[65]
    runner = WalkForwardRunner(WalkForwardPlanner(min_train_size=15, refit_every=5))
    for model, columns in benchmark_models():

        def prepare(frame, selected=columns):
            return prepare_forecast_data(
                VarianceFeatureBuilder(feature_names=selected)(frame),
                build_next_observation_targets(frame, value_col="log_rv"),
            )

        first = runner.run(prepare(daily), model=model, feature_columns=columns)
        second = runner.run(prepare(changed), model=model, feature_columns=columns)
        pd.testing.assert_series_equal(
            first.loc[first.origin_date < cutoff, "predicted"],
            second.loc[second.origin_date < cutoff, "predicted"],
        )


def test_forecasting_and_experiment_imports_do_not_load_matplotlib():
    code = """
import sys
import pandas as pd
from rvol.data.preparation import prepare_forecast_data
from rvol.backtesting import WalkForwardPlanner, WalkForwardRunner
from rvol.data.preparation import build_next_observation_targets
from rvol.models import AR1Model
from rvol.evaluation import evaluate
from rvol import experiment
data = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=20), 'value': range(20)})
features = pd.DataFrame({'origin_date': data.date, 'lag_0': data.value.astype(float)})
prepared = prepare_forecast_data(features, build_next_observation_targets(data))
forecasts = WalkForwardRunner(WalkForwardPlanner(min_train_size=5)).run(
    prepared, model=AR1Model(), feature_columns=("lag_0",)
)
evaluate(forecasts)
assert 'matplotlib' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


def test_importing_models_and_runner_loads_no_preparation_or_evaluation():
    code = """
import sys
from rvol.backtesting import WalkForwardPlanner, WalkForwardRunner
from rvol.models import AR1Model
for prefix in ('rvol.data', 'rvol.evaluation', 'rvol.experiment', 'rvol.simulation',
               'matplotlib', 'scipy'):
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


def runner_for(window: Literal["expanding", "rolling"] = "expanding") -> WalkForwardRunner:
    return WalkForwardRunner(
        WalkForwardPlanner(
            window=window,
            window_size=252 if window == "rolling" else None,
            min_train_size=252,
            forecast_start=date(2024, 10, 1),
            forecast_end=date(2026, 9, 30),
        )
    )


EVALUATION_START = date(2025, 1, 1)


@pytest.mark.parametrize(
    "window, custom_model", [("expanding", False), ("rolling", False), ("expanding", True)]
)
def test_compare_scores_log_and_smeared_variance_forecasts_on_the_same_dates(window, custom_model):
    from rvol.models import ConstantPredictor

    class CustomMean:
        name = "custom_mean"

        def fit(self, features, target):
            return ConstantPredictor(float(np.mean(target)), n_features=features.shape[1])

    models = experiment.benchmark_models()
    if custom_model:
        models = (*models[1:], (CustomMean(), ()))
    forecasts = experiment.forecast(
        experiment.prepare(daily_series()), models, runner_for(window), run_id="test"
    )

    comparison = experiment.compare(forecasts, evaluation_start=EVALUATION_START)

    names = {model.name for model, _ in models}
    scales = ((comparison.log_rv, "log-RV MSE"), (comparison.variance, "QLIKE"))
    for scale, metric in scales:
        assert set(scale.evaluation.summaries.model) == names
        assert set(scale.evaluation.summaries.metric) == {metric}
        assert set(scale.evaluation.comparisons.candidate_model) == {"HAR"}
        assert scale.forecasts.target_date.min() >= pd.Timestamp(EVALUATION_START)
        assert {"win_rates", "hac_sensitivity", "monthly_losses"} <= set(scale.robustness)
    pd.testing.assert_frame_equal(
        comparison.log_rv.forecasts[["model", "target_date"]],
        comparison.variance.forecasts[["model", "target_date"]],
    )
    variance = comparison.variance.forecasts
    assert (variance.smearing_factor > 0).all()
    assert (variance.actual > 0).all() and (variance.predicted > 0).all()
    if window == "rolling":
        assert set(variance.n_train) == {252}


def test_results_table_summarises_both_evaluations():
    forecasts = experiment.forecast(
        experiment.prepare(daily_series()), experiment.benchmark_models(), runner_for()
    )
    comparison = experiment.compare(forecasts, evaluation_start=EVALUATION_START)

    table = experiment.results_table(comparison)

    qlike = comparison.variance.evaluation.summaries.set_index("model").mean_loss
    pd.testing.assert_series_equal(table["QLIKE"], qlike.loc[table.index], check_names=False)
    assert table["QLIKE"].is_monotonic_increasing
    assert table.loc["HAR", "QLIKE: HAR lower by"] == 0
    tests = comparison.log_rv.evaluation.comparisons.set_index("baseline_model").p_value
    np.testing.assert_allclose(table.loc[tests.index, "log-RV MSE: DM p-value"], tests)
    assert pd.isna(table.loc["HAR", "HAR wins (QLIKE)"])


def test_compare_rejects_a_missing_candidate_or_an_empty_window():
    forecasts = experiment.forecast(
        experiment.prepare(daily_series()), experiment.benchmark_models()[:2], runner_for()
    )
    with pytest.raises(ValueError, match="candidate"):
        experiment.compare(forecasts, evaluation_start=EVALUATION_START)
    with pytest.raises(ValueError, match="evaluation_start"):
        experiment.compare(forecasts, evaluation_start=date(2030, 1, 1), candidate="naive")


def test_forecast_requires_at_least_one_model():
    with pytest.raises(ValueError, match="at least one model"):
        experiment.forecast(experiment.prepare(daily_series()), (), runner_for())


def test_separate_models_share_an_explicit_comparison_sample():
    data = prepared_variance()
    runner = WalkForwardRunner(WalkForwardPlanner(min_train_size=15))
    forecasts = pd.concat(
        [
            runner.run(data, model=model, feature_columns=columns)
            for model, columns in benchmark_models()
        ],
        ignore_index=True,
    )
    evaluation = evaluate(forecasts, comparison_pairs=(("naive", "HAR"),))
    assert len(evaluation.summaries) == 5
    assert forecasts.groupby("model").target_date.apply(list).map(tuple).nunique() == 1
    assert len(evaluation.comparisons) == 1
