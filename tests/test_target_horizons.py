import json
from collections.abc import Callable
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from rvol.domain import (
    EveryNSessions,
    ExperimentConfig,
    ExperimentResult,
    ForecastRecord,
    HoldoutSettings,
    RollingWindow,
    RunIdentity,
    TargetHorizon,
    WalkForwardSettings,
)
from rvol.domain.contracts import FloatArray
from rvol.forecasting.composition import standard_experiment
from rvol.forecasting.evaluation import ForecastEvaluator
from rvol.forecasting.execution import (
    HoldoutExperiment,
    WalkForwardExperiment,
    prepare_forecast_data,
)
from rvol.forecasting.features import HARFeatureBuilder
from rvol.forecasting.models import EWMAForecaster, HistoricalMeanForecaster
from rvol.forecasting.models.predictors import ConstantPredictor
from rvol.forecasting.targets import RealizedVarianceTargetBuilder
from rvol.presentation.plotting.forecasting import plot_forecast_evaluation
from rvol.presentation.reporting import (
    DatasetSnapshot,
    ForecastArtifactWriter,
    RobustnessArtifactWriter,
    ValidationArtifactWriter,
)


def daily_sample(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=n),
        "log_rv": -10 + 0.2 * np.sin(np.arange(n) / 4) + rng.normal(0, 0.05, n),
    })


def prepared_sample(daily: pd.DataFrame, horizon: TargetHorizon) -> pd.DataFrame:
    return prepare_forecast_data(
        HARFeatureBuilder(weekly_window=2, monthly_window=4)(daily),
        RealizedVarianceTargetBuilder(horizon)(daily),
    )


@pytest.mark.parametrize("kind", ["session", "cumulative"])
@pytest.mark.parametrize("sessions", [1, 3, 5, 100])
def test_targets_match_future_observed_sessions_and_drop_incomplete_intervals(kind, sessions):
    daily = daily_sample(16).drop(index=10).reset_index(drop=True)
    daily["log_rv"] = np.log(np.arange(1, len(daily) + 1))
    horizon = TargetHorizon(sessions, kind)
    features = prepared_sample(daily, horizon)

    assert len(features) == max(0, len(daily) - 3 - sessions)
    assert features.attrs["target_horizon"] == {"sessions": sessions, "kind": kind}
    for offset, row in enumerate(features.itertuples(), start=3):
        expected = float(offset + sessions + 1)
        if kind == "cumulative":
            expected = sum(range(offset + 2, offset + sessions + 2))
        assert row.target == pytest.approx(np.log(expected))
        assert row.origin_date == daily["date"].iloc[offset]
        assert row.target_date == daily["date"].iloc[offset + sessions]
        assert row.target_start_date == daily["date"].iloc[
            offset + (1 if kind == "cumulative" else sessions)
        ]
        assert row.rv_monthly == pytest.approx(daily["log_rv"].iloc[offset - 3:offset + 1].mean())


def test_cumulative_targets_sum_variances_stably_rather_than_logs():
    daily = daily_sample(12)
    daily["log_rv"] = 1000.0
    features = prepared_sample(daily, TargetHorizon(3, "cumulative"))
    np.testing.assert_allclose(features["target"], 1000 + np.log(3))


class RecordingMean(HistoricalMeanForecaster):
    def __init__(self) -> None:
        self.training_targets: list[FloatArray] = []

    def fit(self, features: FloatArray, target: FloatArray) -> ConstantPredictor:
        self.training_targets.append(target.copy())
        return super().fit(features, target)


@pytest.mark.parametrize("kind", ["session", "cumulative"])
def test_rolling_refits_admit_only_labels_whose_final_session_is_known(kind):
    features = prepared_sample(daily_sample(45), TargetHorizon(5, kind))
    model = RecordingMean()
    result = WalkForwardExperiment((model,), ExperimentConfig(execution=WalkForwardSettings(
        RollingWindow(10), EveryNSessions(3), min_train_size=8,
    ))).run(features)
    fit_dates = sorted({r.fit_date for r in result.records if r.fit_date is not None})

    assert len(fit_dates) == len(model.training_targets) > 1
    for fit_date, observed in zip(fit_dates, model.training_targets, strict=True):
        assert fit_date is not None
        eligible = features[features["target_date"] <= pd.Timestamp(fit_date)].tail(10)
        # Rows originating before the fit but ending after it are still unavailable.
        assert ((features.origin_date < pd.Timestamp(fit_date)) & (
            features.target_date > pd.Timestamp(fit_date)
        )).any()
        np.testing.assert_array_equal(observed, eligible["target"])
        for record in (r for r in result.records if r.fit_date == fit_date):
            assert record.n_train == len(eligible)
            assert record.train_end_date == eligible["target_date"].iloc[-1].date()
            assert record.predicted_log_rv == pytest.approx(eligible["target"].mean())


@pytest.mark.parametrize("kind", ["session", "cumulative"])
@pytest.mark.parametrize("update", [False, True])
def test_holdout_ewma_waits_for_complete_labels_and_handles_delayed_reporting(kind, update):
    daily = daily_sample(45)
    horizon = TargetHorizon(5, kind)
    features = prepared_sample(daily, horizon)
    cutoff = daily["date"].iloc[20].date()
    config = ExperimentConfig(execution=HoldoutSettings(cutoff, 8, update))
    runner = HoldoutExperiment((EWMAForecaster(decay=0.5),), config)
    full = runner.run(features)
    state = None
    consumed = 0
    for record in full.records:
        available_date = record.origin_date if update else cutoff
        available = features[features.target_date <= pd.Timestamp(available_date)]
        for variance in np.exp(available["target"].iloc[consumed:]):
            state = variance if state is None else 0.5 * state + 0.5 * variance
        consumed = len(available)
        assert record.predicted_rv == pytest.approx(state)
        assert record.train_end_date == cutoff
        assert record.n_train == len(features[features.target_date <= pd.Timestamp(cutoff)])

    bounded_config = replace(config, forecast_start=full.records[7].target_date)
    bounded = HoldoutExperiment((EWMAForecaster(decay=0.5),), bounded_config).run(features)
    assert bounded.records == full.records[7:]


@pytest.mark.parametrize("kind", ["session", "cumulative"])
@pytest.mark.parametrize("strategy", ["expanding", "rolling", "holdout", "holdout_updated"])
def test_future_values_cannot_change_any_models_current_forecast(kind, strategy):
    daily = daily_sample()
    changed = daily.copy()
    horizon = TargetHorizon(3, kind)
    origin_position = 65
    future_position = origin_position + (1 if kind == "cumulative" else horizon.sessions)
    changed.loc[future_position, "log_rv"] = float(changed["log_rv"].iloc[future_position]) + 2.0
    origin = daily["date"].iloc[origin_position].date()
    run_daily: Callable[[pd.DataFrame], ExperimentResult]
    if strategy.startswith("holdout"):
        holdout_config = ExperimentConfig(execution=HoldoutSettings(
            daily["date"].iloc[45].date(), 20, strategy == "holdout_updated",
        ))
        run_daily = standard_experiment(holdout_config, target_horizon=horizon).run
    else:
        settings = WalkForwardSettings(retrain=EveryNSessions(4), min_train_size=20)
        if strategy == "rolling":
            settings = replace(settings, training_window=RollingWindow(25))
        walk_forward_config = ExperimentConfig(execution=settings)
        run_daily = standard_experiment(walk_forward_config, target_horizon=horizon).run
    before = run_daily(daily)
    after = run_daily(changed)
    assert [(r.model, r.origin_date, r.predicted_log_rv) for r in before.records
            if r.origin_date <= origin] == [
        (r.model, r.origin_date, r.predicted_log_rv)
        for r in after.records if r.origin_date <= origin
    ]
    assert [r.actual_log_rv for r in before.records if r.origin_date == origin] != [
        r.actual_log_rv for r in after.records if r.origin_date == origin
    ]
    assert before.target_horizon == horizon
    assert {r.target_horizon for r in before.records} == {horizon}
    for record in before.records:
        if record.model == "naive":
            current = daily.loc[daily.date == pd.Timestamp(record.origin_date), "log_rv"].iloc[0]
            multiplier = horizon.sessions if kind == "cumulative" else 1
            assert record.predicted_rv == pytest.approx(np.exp(current) * multiplier)


def test_multisession_walk_forward_reporting_bounds_preserve_fit_schedule_and_updates():
    daily = daily_sample()
    config = ExperimentConfig(execution=WalkForwardSettings(
        RollingWindow(25), EveryNSessions(4), min_train_size=20,
    ))
    workflow = standard_experiment(config, target_horizon=TargetHorizon(5, "cumulative"))
    full = workflow.run(daily)
    start, end = daily["date"].iloc[61].date(), daily["date"].iloc[72].date()
    bounded = replace(workflow, config=replace(
        config, forecast_start=start, forecast_end=end,
    )).run(daily)
    assert bounded.records == tuple(r for r in full.records if start <= r.target_date <= end)
    assert bounded.execution == full.execution


def test_empty_walk_forward_result_preserves_its_target_definition():
    horizon = TargetHorizon(100, "cumulative")
    result = standard_experiment(
        ExperimentConfig(execution=WalkForwardSettings(min_train_size=5)),
        target_horizon=horizon,
    ).run(daily_sample(30))
    assert result.records == ()
    assert result.target_horizon == horizon


@pytest.mark.parametrize("sessions, kind", [
    (0, "session"), (-1, "session"), (True, "session"), (1.5, "session"), (1, "unknown"),
])
def test_target_definitions_reject_invalid_options(sessions, kind):
    with pytest.raises(ValueError):
        TargetHorizon(sessions, kind)


@pytest.mark.parametrize(
    "corruption", ["partial", "mixed", "missing_start", "reversed", "same_end"],
)
def test_executors_reject_ambiguous_or_invalid_prepared_targets(corruption):
    features = standard_experiment(
        target_horizon=TargetHorizon(3, "cumulative"),
    ).prepare(daily_sample(40))
    if corruption == "partial":
        features = features.drop(columns="target_kind")
    elif corruption == "mixed":
        features.loc[0, "horizon_sessions"] = 5
    elif corruption == "missing_start":
        features = features.drop(columns="target_start_date")
    else:
        features.loc[0, "target_start_date"] = features.loc[
            0, "origin_date" if corruption == "reversed" else "target_date"
        ]
    with pytest.raises(ValueError):
        WalkForwardExperiment((HistoricalMeanForecaster(),), ExperimentConfig(
            execution=WalkForwardSettings(min_train_size=5),
        )).run(features)


def test_records_require_a_consistent_target_interval_and_experiment_horizon():
    record = ForecastRecord(
        "model", date(2024, 1, 1), date(2024, 1, 4), 10, -10.0, -10.0,
        target_horizon=TargetHorizon(3, "cumulative"), target_start_date=date(2024, 1, 2),
    )
    with pytest.raises(ValueError, match="explicit target start"):
        replace(record, target_start_date=None)
    with pytest.raises(ValueError, match="start before their end"):
        replace(record, target_start_date=record.target_date)
    with pytest.raises(ValueError, match="must match the experiment"):
        ExperimentResult((record,))
    with pytest.raises(ValueError, match="start and end"):
        replace(record, target_horizon=TargetHorizon(3, "session"))


def test_default_comparison_accounts_for_horizon_and_explicit_lags_remain_configurable():
    daily = daily_sample()
    config = ExperimentConfig(
        execution=WalkForwardSettings(min_train_size=20),
        forecast_start=daily["date"].iloc[70].date(),
        forecast_end=daily["date"].iloc[79].date(),
    )
    workflow = standard_experiment(config, target_horizon=TargetHorizon(5, "cumulative"))
    experiment = workflow.run(daily)
    evaluation = workflow.evaluator().evaluate(experiment)
    direct = ForecastEvaluator(comparison_pairs=workflow.comparison_pairs).evaluate(experiment)
    assert direct == evaluation
    assert {dict(item.test_parameters)["hac_lags"] for item in evaluation.comparisons} == {4}
    overridden = workflow.evaluator(hac_lags=0).evaluate(experiment)
    assert {dict(item.test_parameters)["hac_lags"] for item in overridden.comparisons} == {0}


@pytest.mark.parametrize("column", ["origin_date", "target_start_date"])
def test_evaluation_rejects_models_predicting_different_intervals(column):
    workflow = standard_experiment(
        ExperimentConfig(execution=WalkForwardSettings(min_train_size=20)),
        target_horizon=TargetHorizon(5, "cumulative"),
    )
    result = workflow.run(daily_sample())
    first = result.records[0]
    if column == "origin_date":
        altered = replace(
            first, origin_date=first.origin_date - timedelta(days=1),
            fit_date=None, train_start_date=None, train_end_date=None,
        )
    else:
        altered = replace(first, target_start_date=first.target_start + timedelta(days=1))
    with pytest.raises(ValueError, match="identical forecast origins and target starts"):
        ForecastEvaluator().evaluate(replace(result, records=(altered, *result.records[1:])))


def test_horizon_is_exported_and_mismatched_evaluations_and_validation_are_rejected(tmp_path):
    horizon = TargetHorizon(3, "cumulative")
    daily = daily_sample()
    source = tmp_path / "daily.parquet"
    daily.to_parquet(source, index=False)
    config = ExperimentConfig(execution=WalkForwardSettings(min_train_size=20))
    workflow = standard_experiment(config, target_horizon=horizon)
    experiment = workflow.run(daily)
    evaluation = workflow.evaluator().evaluate(experiment)
    snapshot = DatasetSnapshot.from_frame(source, daily)
    writer = ForecastArtifactWriter(tmp_path / "forecast")
    with pytest.raises(ValueError, match="evaluation target must match"):
        writer.write(
            experiment, replace(evaluation, target_horizon=TargetHorizon()),
            config=config, dataset=snapshot, specification=workflow.specification,
        )
    assert not writer.output_dir.exists()
    with pytest.raises(ValueError, match="evaluation target must match"):
        plot_forecast_evaluation(
            experiment, replace(evaluation, target_horizon=TargetHorizon()), candidate_model="HAR",
        )
    paths = writer.write(
        experiment, evaluation, config=config, dataset=snapshot,
        specification=workflow.specification,
    )
    metadata = json.loads(paths.experiment.read_text())
    assert metadata["target"]["parameters"] == {"sessions": 3, "kind": "cumulative"}
    forecasts = pd.read_csv(paths.forecasts)
    assert list(forecasts.target_start_date) == [str(r.target_start) for r in sorted(
        experiment.records, key=lambda r: (r.target_date, r.model),
    )]
    robust = workflow.robustness_analyzer().analyze(experiment)
    robust_paths = RobustnessArtifactWriter(tmp_path / "robust").write(robust)
    assert json.loads(robust_paths.metadata.read_text())["target"] == {
        "sessions": 3, "kind": "cumulative",
    }
    other = replace(experiment, identity=RunIdentity("other"), config=None)
    validation = workflow.evaluator().evaluate_runs((experiment, other))
    validation_paths = ValidationArtifactWriter(tmp_path / "validation").write(validation)
    assert json.loads(validation_paths.metadata.read_text())["target"]["sessions"] == 3
    for path in (paths.forecasts, paths.loss_summary, paths.comparisons,
                 robust_paths.csv("monthly_losses"), validation_paths.losses,
                 validation_paths.aggregate_summaries):
        frame = pd.read_csv(path)
        assert set(frame.horizon_sessions) == {3}
        assert set(frame.target_kind) == {"cumulative"}
    incompatible = replace(
        other, target_horizon=TargetHorizon(5, "cumulative"), records=tuple(
            replace(r, target_horizon=TargetHorizon(5, "cumulative")) for r in other.records
        ),
    )
    with pytest.raises(ValueError, match="identical target horizons"):
        workflow.evaluator().evaluate_runs((experiment, incompatible))
