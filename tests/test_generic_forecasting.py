import numpy as np
import pandas as pd
import pytest

from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.evaluation import MAEMetric, MSEMetric, evaluate
from rvol.models import (
    AR1Model,
    ConstantPredictor,
    EWMAModel,
    HistoricalMeanModel,
    LinearRegressionModel,
    NaiveModel,
)
from rvol.walkforward import WalkForwardPlanner, WalkForwardRunner
from tests.forecasting_helpers import lag_features


@pytest.mark.parametrize("window", ["expanding", "rolling"])
@pytest.mark.parametrize("subject, shift", [("apples", 0), ("price", 100), ("temperature", -20)])
def test_same_model_and_execution_work_for_arbitrary_subjects(window, subject, shift):
    data = pd.DataFrame(
        {"observed_at": pd.date_range("2024-01-01", periods=40), subject: np.arange(40) + shift}
    )
    prepared = prepare_forecast_data(
        lag_features(data, date_col="observed_at", value_col=subject),
        build_next_observation_targets(data, date_col="observed_at", value_col=subject),
    )
    model = AR1Model()
    runner = WalkForwardRunner(
        WalkForwardPlanner(
            window=window,
            window_size=8 if window == "rolling" else None,
            min_train_size=5,
        )
    )
    forecasts = runner.run(prepared, model=model, feature_columns=("lag_0",))
    assert isinstance(forecasts, pd.DataFrame)
    np.testing.assert_allclose(forecasts.predicted, forecasts.actual, atol=1e-10)
    evaluation = evaluate(forecasts)
    assert evaluation.summaries.metric.tolist() == ["MSE"]
    assert evaluation.summaries.mean_loss.iloc[0] < 1e-15


def test_finite_negative_zero_and_large_values_are_preserved():
    data = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=45), "value": (np.arange(45) - 20) * 1000.0}
    )
    prepared = prepare_forecast_data(lag_features(data), build_next_observation_targets(data))
    forecasts = WalkForwardRunner(WalkForwardPlanner(min_train_size=5)).run(
        prepared, model=AR1Model(), feature_columns=("lag_0",)
    )
    assert forecasts.actual.min() < 0 < forecasts.actual.max()
    assert 0 in forecasts.actual.values
    np.testing.assert_allclose(forecasts.actual, forecasts.predicted, atol=1e-8)


def test_independent_model_implements_only_name_fit_and_predict():
    class Mean:
        name = "custom_mean"

        def fit(self, features, target):
            return ConstantPredictor(float(np.mean(target)))

    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=15), "value": np.arange(15)})
    prepared = prepare_forecast_data(
        lag_features(data, lags=()), build_next_observation_targets(data)
    )
    forecasts = WalkForwardRunner(WalkForwardPlanner(min_train_size=3)).run(prepared, model=Mean())
    assert set(forecasts.model) == {"custom_mean"}
    assert forecasts.predicted.iloc[0] == 2
    assert not hasattr(Mean(), "specification")


def test_ewma_preserves_numeric_and_logarithmic_target_scales():
    values = np.array([1.0, 9.0])
    features = np.empty((len(values), 0))
    raw = EWMAModel(decay=0.5).fit(features, values)
    logged = EWMAModel(decay=0.5).fit(features, np.log(values))
    np.testing.assert_allclose(raw.predict(features), [5.0, 5.0])
    np.testing.assert_allclose(logged.predict(features), np.log([3.0, 3.0]))
    signed = EWMAModel(decay=0.5).fit(np.empty((3, 0)), np.array([-3.0, 0.0, 9.0]))
    assert signed.value == pytest.approx(3.75)
    assert signed.update(np.empty((1, 0)), np.array([-9.0])).value == pytest.approx(-2.625)
    assert signed.value == pytest.approx(3.75)


def test_linear_regression_preserves_supplied_target_scale():
    model = LinearRegressionModel()
    features = np.zeros((4, 3))
    predictor = model.fit(features, np.array([-3.0, -1.0, 1.0, 3.0]))
    np.testing.assert_allclose(predictor.predict(features), 0, atol=1e-15)


def test_naive_and_metrics_do_not_apply_implicit_transformations():
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=12), "value": np.arange(12)})
    prepared = prepare_forecast_data(lag_features(data), build_next_observation_targets(data))
    forecasts = WalkForwardRunner(WalkForwardPlanner(min_train_size=3)).run(
        prepared, model=NaiveModel(), feature_columns=("lag_0",)
    )
    metrics = evaluate(forecasts, metrics=(MSEMetric(), MAEMetric()))
    np.testing.assert_allclose(metrics.summaries.mean_loss, 1)


def test_next_observation_targets_preserve_values_without_transformations():
    data = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=6), "value": [-5.0, 0.0, 10.0, -2.0, 3.0, 4.0]}
    )
    targets = build_next_observation_targets(data)
    np.testing.assert_array_equal(targets.target.iloc[:-1], [0.0, 10.0, -2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(targets.target_date, data.date.shift(-1), check_names=False)
    assert pd.isna(targets.target.iloc[-1])


def test_solo_models_prepare_only_their_required_inputs():
    from tests.forecasting_helpers import benchmark_models, prepared_variance

    runner = WalkForwardRunner(WalkForwardPlanner(min_train_size=5))
    for model, columns in benchmark_models():
        data = prepared_variance(n=50, feature_columns=columns)
        forecasts = runner.run(data, model=model, feature_columns=columns)
        assert (
            forecasts.target_date.iloc[0]
            == pd.bdate_range("2023-01-02", periods=50)[27 if model.name == "HAR" else 6]
        )
        assert set(forecasts.model) == {model.name}


@pytest.mark.parametrize("window", ["expanding", "rolling"])
def test_reusing_a_runner_after_other_models_and_data_preserves_fresh_state(window):
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20), "value": np.arange(20)})
    prepared = prepare_forecast_data(
        lag_features(data, lags=()), build_next_observation_targets(data)
    )
    runner = WalkForwardRunner(
        WalkForwardPlanner(
            window=window,
            window_size=8 if window == "rolling" else None,
            min_train_size=5,
            refit_every=7,
        )
    )
    model = EWMAModel(decay=0.8)
    first = runner.run(prepared, model=model)
    changed = prepared.copy(deep=True)
    changed["target"] += 100
    shifted = runner.run(changed, model=model)
    runner.run(prepared, model=HistoricalMeanModel())
    repeated = runner.run(prepared, model=model)

    pd.testing.assert_frame_equal(first, repeated)
    np.testing.assert_allclose(shifted.predicted, first.predicted + 100)


def test_a_custom_model_uses_ready_data_in_the_callers_feature_order():
    class Learned:
        def __init__(self, intercept):
            self.intercept = intercept

        def predict(self, features):
            return self.intercept + features[:, 0] ** 2 + 2 * features[:, 1]

    class CustomModel:
        name = "custom_nonlinear"

        def fit(self, features, target):
            residuals = target - features[:, 0] ** 2 - 2 * features[:, 1]
            return Learned(float(np.mean(residuals)))

    dates = pd.bdate_range("2024-01-01", periods=41)
    first, second = np.random.default_rng(12).normal(size=(2, 40))
    ready = pd.DataFrame(
        {
            "origin_date": dates[:-1],
            "target_date": dates[1:],
            "target": 7 + second**2 + 2 * first,
            "first": first,
            "second": second,
        }
    )
    forecasts = WalkForwardRunner(WalkForwardPlanner(min_train_size=5)).run(
        ready, model=CustomModel(), feature_columns=("second", "first")
    )

    np.testing.assert_allclose(forecasts.predicted, forecasts.actual, atol=1e-12)
    assert not hasattr(CustomModel(), "feature_names")
    assert set(forecasts.model) == {"custom_nonlinear"}


def test_a_model_cannot_corrupt_subsequent_training_by_mutating_its_arrays():
    class MutatingModel:
        name = "mutating_mean"

        def fit(self, features, target):
            predictor = ConstantPredictor(float(np.mean(target)), n_features=features.shape[1])
            features[:] = 1e9
            target[:] = 1e9
            return predictor

    dates = pd.bdate_range("2024-01-01", periods=21)
    ready = pd.DataFrame(
        {
            "origin_date": dates[:-1],
            "target_date": dates[1:],
            "target": np.arange(20, dtype=float),
            "input": np.arange(20, dtype=float),
        }
    )
    unchanged = ready.copy(deep=True)
    forecasts = WalkForwardRunner(WalkForwardPlanner(min_train_size=3)).run(
        ready, model=MutatingModel(), feature_columns=("input",)
    )

    for row in forecasts.itertuples():
        history = ready.loc[ready.target_date <= row.origin_date]
        assert row.predicted == pytest.approx(history.target.mean())
    pd.testing.assert_frame_equal(ready, unchanged)
