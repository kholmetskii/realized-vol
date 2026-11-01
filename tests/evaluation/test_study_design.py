from datetime import date

import numpy as np
import pandas as pd
import pytest

from rvol.data.preparation import build_forecast_data
from rvol.evaluation import (
    StudyDesign,
    evaluate,
    qlike,
    results_table,
    win_rates,
)
from rvol.models import ForecastModel
from rvol.walkforward import smeared_forecasts
from tests.forecasting_helpers import FEATURE_TABLES, benchmark_models, with_semivariances

DESIGN = StudyDesign(
    forecast_start=date(2024, 10, 1),
    evaluation_start=date(2025, 1, 1),
    forecast_end=date(2026, 9, 30),
    min_train_size=252,
    candidate="HAR",
    loss=qlike,
    loss_name="QLIKE",
)


def daily_series(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-03", periods=n),
            "log_rv": -10 + np.sin(np.arange(n) / 4) / 5 + rng.normal(0, 0.3, n),
        }
    )
    return with_semivariances(daily)


def prepared() -> pd.DataFrame:
    return build_forecast_data(daily_series(), FEATURE_TABLES, value_col="log_rv")


def test_a_run_forecasts_the_design_dates_and_scores_the_evaluation_period():
    data = prepared()
    models = benchmark_models()

    run = DESIGN.run(data, models, series="test")

    assert run.series == "test" and run.data is data
    pd.testing.assert_frame_equal(run.forecasts, DESIGN.forecast(data, models, series="test"))
    assert run.forecasts["target_date"].min() >= pd.Timestamp(DESIGN.forecast_start)
    assert run.forecasts["target_date"].max() <= pd.Timestamp(DESIGN.forecast_end)
    evaluation = run.evaluation
    assert evaluation.forecasts["target_date"].min() >= pd.Timestamp(DESIGN.evaluation_start)
    assert evaluation.forecasts.groupby("model").target_date.apply(tuple).nunique() == 1
    assert set(evaluation.summary.model) == {model.name for model in models}
    assert evaluation.summary.set_index("model").loc["HAR", ["p_value"]].isna().all()
    assert list(run.losses().columns) == sorted(model.name for model in models)


def test_the_evaluation_smears_with_errors_from_the_calibration_window():
    forecasts = DESIGN.forecast(prepared(), benchmark_models(), series="test")

    evaluation = DESIGN.evaluate(forecasts)

    smeared = smeared_forecasts(forecasts)
    expected = smeared.loc[smeared["target_date"] >= pd.Timestamp(DESIGN.evaluation_start)]
    pd.testing.assert_frame_equal(
        evaluation.forecasts, expected.sort_values(["model", "target_date"]).reset_index(drop=True)
    )


def test_the_results_table_summarises_the_evaluation():
    evaluation = DESIGN.run(prepared(), benchmark_models(), series="test").evaluation

    table = results_table(evaluation)

    summary = evaluation.summary.set_index("model")
    pd.testing.assert_series_equal(
        table["QLIKE"], summary.mean_loss.loc[table.index], check_names=False
    )
    assert table["QLIKE"].is_monotonic_increasing
    assert table.loc["HAR", "HAR lower by"] == 0 and pd.isna(table.loc["HAR", "HAR wins"])
    np.testing.assert_allclose(table["DM p-value"], summary.p_value.loc[table.index])
    wins = win_rates(evaluation.losses, "HAR").set_index("baseline_model").candidate_win_rate
    np.testing.assert_allclose(table["HAR wins"].drop("HAR"), wins.loc[table.index.drop("HAR")])


def test_a_run_needs_the_candidate_and_an_evaluation_period():
    models = [model for model in benchmark_models() if model.name != "HAR"]
    with pytest.raises(ValueError, match="candidate"):
        DESIGN.run(prepared(), models, series="test")
    forecasts = DESIGN.forecast(prepared(), benchmark_models()[:2], series="test")
    with pytest.raises(ValueError, match="evaluation_start"):
        evaluate(forecasts, candidate="naive", loss=qlike, evaluation_start=date(2030, 1, 1))


def test_the_loss_name_defaults_to_the_loss_function_name():
    forecasts = DESIGN.forecast(prepared(), benchmark_models()[:2], series="test")

    evaluation = evaluate(
        forecasts, candidate="naive", loss=qlike, evaluation_start=date(2025, 1, 1)
    )

    assert evaluation.loss_name == "qlike"


@pytest.mark.parametrize(
    "change",
    [
        {"evaluation_start": date(2024, 1, 1)},
        {"forecast_end": date(2024, 12, 31)},
        {"min_train_size": 0},
        {"min_train_size": True},
    ],
)
def test_invalid_designs_are_rejected(change):
    settings = {field: getattr(DESIGN, field) for field in DESIGN.__dataclass_fields__} | change
    with pytest.raises(ValueError):
        StudyDesign(**settings)


def test_a_custom_rule_takes_part_like_any_other():
    class CustomMean:
        def fit(self, features, target):
            self.mean = float(np.mean(target))
            return self

        def predict(self, features):
            return np.full(len(features), self.mean)

    models = (*benchmark_models(), ForecastModel("custom_mean", CustomMean()))

    run = DESIGN.run(prepared(), models, series="test")

    assert "custom_mean" in set(run.evaluation.summary.model)
