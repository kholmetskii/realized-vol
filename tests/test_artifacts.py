import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

from rvol.domain import (
    ComparisonOutcome,
    ComponentSpecification,
    EveryNSessions,
    ExecutionMetadata,
    ExecutionSpecification,
    ExperimentConfig,
    RollingWindow,
    RunIdentity,
    WalkForwardSettings,
)
from rvol.forecasting.composition import standard_experiment_definition
from rvol.forecasting.evaluation import ForecastEvaluator
from rvol.forecasting.features import HARFeatureBuilder
from rvol.forecasting.models import EWMAForecaster, HARForecaster
from rvol.forecasting.targets import RealizedVarianceTargetBuilder
from rvol.presentation.reporting import DatasetSnapshot, ForecastArtifactWriter


@pytest.fixture
def configured_run(tmp_path):
    step = np.arange(40, dtype="float64")
    dates = pd.bdate_range("2024-01-02", periods=len(step))
    daily = pd.DataFrame({
        "session": dates,
        "value": -10.0 + 0.2 * np.sin(step / 4) + 0.01 * step,
    })
    source = tmp_path / "daily.parquet"
    daily.to_parquet(source, index=False)
    definition = replace(
        standard_experiment_definition(),
        models=(EWMAForecaster(decay=0.8), HARForecaster()),
        feature_builder=HARFeatureBuilder(
            date_col="session", rv_col="value", weekly_window=2,
            monthly_window=4, max_gap_days=None,
        ),
        target_builder=RealizedVarianceTargetBuilder(date_col="session", rv_col="value"),
        comparison_pairs=(("EWMA", "HAR"),),
        baseline_models=("EWMA",),
    )
    config = ExperimentConfig(
        execution=WalkForwardSettings(RollingWindow(12), EveryNSessions(3), min_train_size=10),
        forecast_start=dates[22].date(),
        forecast_end=dates[30].date(),
    )
    specification = definition.specification
    experiment = definition.run(daily, config)
    evaluation = definition.evaluator(hac_lags=0).evaluate(experiment)
    snapshot = DatasetSnapshot.from_frame(source, daily, date_column="session")
    return experiment, evaluation, config, specification, snapshot


def test_artifacts_record_nondefault_settings_and_remain_deterministic(tmp_path, configured_run):
    experiment, evaluation, config, specification, snapshot = configured_run
    writer = ForecastArtifactWriter(tmp_path / "artifacts")
    paths = writer.write(
        experiment, evaluation, config=config, dataset=snapshot, specification=specification,
    )
    first = {path.name: path.read_bytes() for path in paths.all()}
    metadata = json.loads(paths.experiment.read_text())

    assert metadata["schema_version"] == 9
    assert experiment.execution is not None
    assert metadata["execution"] == {
        "executor": "rvol.forecasting.execution.walk_forward.WalkForwardExperiment",
        "settings": {
            "name": "walk_forward",
            "implementation": "rvol.domain.settings.WalkForwardSettings",
            "feature_names": [],
            "parameters": {"min_train_size": 10},
        },
        "policies": {
            "training_window": {
                "name": "rolling",
                "implementation": "rvol.domain.policies.RollingWindow",
                "feature_names": [],
                "parameters": {"size": 12},
            },
            "retrain": {
                "name": "every_n_sessions",
                "implementation": "rvol.domain.policies.EveryNSessions",
                "feature_names": [],
                "parameters": {"interval": 3},
            },
        },
        "facts": {
            "schedule_anchor": dict(experiment.execution.facts)["schedule_anchor"],
            "schedule_anchor_policy": "first_eligible_origin",
            "observation_update_policy": "newly_available_targets_between_refits",
        },
    }
    forecasts = pd.read_csv(paths.forecasts)
    records = sorted(experiment.records, key=lambda record: (record.target_date, record.model))
    for column in ("fit_date", "train_start_date", "train_end_date"):
        assert list(forecasts[column]) == [str(getattr(record, column)) for record in records]
    assert list(forecasts["n_train"]) == [record.n_train for record in records]
    assert records[0].fit_date is not None
    assert records[0].fit_date < records[0].origin_date
    assert metadata["specification"]["features"]["parameters"] == {
        "date_col": "session", "rv_col": "value", "weekly_window": 2,
        "monthly_window": 4, "max_gap_days": None,
    }
    assert metadata["specification"]["target"]["parameters"] == {
        "date_col": "session", "rv_col": "value",
        "horizon_sessions": 1, "target_kind": "session",
    }
    models = metadata["specification"]["models"]
    assert [model["name"] for model in models] == ["EWMA", "HAR"]
    assert models[0]["implementation"] == "rvol.forecasting.models.ewma.EWMAForecaster"
    assert models[0]["parameters"] == {"decay": 0.8}
    assert models[1]["implementation"] == "rvol.forecasting.models.har.HARForecaster"
    assert models[1]["parameters"] == {"variance_correction": "duan_smearing"}
    assert "intercept" not in models[1]["parameters"]  # Fitted estimates are not configuration.

    repeated = writer.write(
        experiment, evaluation, config=config, dataset=snapshot, specification=specification,
    )
    assert {path.name: path.read_bytes() for path in repeated.all()} == first


def test_writer_rejects_metadata_for_different_models_before_writing(tmp_path, configured_run):
    experiment, evaluation, config, specification, snapshot = configured_run
    output = tmp_path / "artifacts"
    incomplete = replace(specification, models=(specification.models[0],))

    with pytest.raises(ValueError, match="must match the forecast models"):
        ForecastArtifactWriter(output).write(
            experiment, evaluation, config=config, dataset=snapshot, specification=incomplete,
        )

    assert not output.exists()


def test_writer_requires_execution_metadata_before_writing(tmp_path, configured_run):
    experiment, evaluation, config, specification, snapshot = configured_run
    output = tmp_path / "artifacts"
    incomplete = replace(experiment, execution=None)

    with pytest.raises(ValueError, match="must include execution metadata"):
        ForecastArtifactWriter(output).write(
            incomplete, evaluation, config=config, dataset=snapshot, specification=specification,
        )

    assert not output.exists()


def test_writer_rejects_configuration_for_a_different_run_before_writing(tmp_path, configured_run):
    experiment, evaluation, config, specification, snapshot = configured_run
    output = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="configuration must match the completed experiment"):
        ForecastArtifactWriter(output).write(
            experiment, evaluation,
            config=replace(config, execution=replace(config.execution, min_train_size=11)),
            dataset=snapshot, specification=specification,
        )

    assert not output.exists()


def test_writer_rejects_changed_execution_settings_before_writing(tmp_path, configured_run):
    experiment, evaluation, config, specification, snapshot = configured_run
    output = tmp_path / "artifacts"
    inconsistent = replace(
        experiment, execution=replace(
            experiment.execution, settings=WalkForwardSettings().specification,
        ),
    )

    with pytest.raises(ValueError, match="execution specification must match"):
        ForecastArtifactWriter(output).write(
            inconsistent, evaluation, config=config, dataset=snapshot, specification=specification,
        )

    assert not output.exists()


def test_writer_requires_fit_boundaries_before_writing(tmp_path, configured_run):
    experiment, evaluation, config, specification, snapshot = configured_run
    output = tmp_path / "artifacts"
    unaudited = replace(experiment, records=tuple(
        replace(record, fit_date=None, train_start_date=None, train_end_date=None)
        for record in experiment.records
    ))

    with pytest.raises(ValueError, match="must include fit dates and training boundaries"):
        ForecastArtifactWriter(output).write(
            unaudited, evaluation, config=config, dataset=snapshot, specification=specification,
        )

    assert not output.exists()


def test_writer_exports_independent_executor_metadata_without_walk_forward_rules(
    tmp_path, configured_run,
):
    experiment, evaluation, config, specification, snapshot = configured_run

    @dataclass(frozen=True)
    class CustomSettings:
        batches: int = 2

        @property
        def specification(self) -> ExecutionSpecification:
            return ExecutionSpecification(ComponentSpecification(
                "custom", "test.CustomSettings", parameters=(("batches", self.batches),),
            ))

    custom_config = replace(config, execution=CustomSettings())
    metadata = ExecutionMetadata(
        executor="test.CustomExecutor", settings=custom_config.execution.specification,
        facts=(("completed_batches", 2),),
    )
    custom_run = replace(experiment, config=custom_config, execution=metadata)
    paths = ForecastArtifactWriter(tmp_path / "custom").write(
        custom_run, evaluation, config=custom_config, dataset=snapshot, specification=specification,
    )

    exported = json.loads(paths.experiment.read_text())["execution"]
    assert exported["executor"] == "test.CustomExecutor"
    assert exported["settings"]["parameters"] == {"batches": 2}
    assert exported["policies"] == {}
    assert exported["facts"] == {"completed_batches": 2}


def test_writer_exports_an_independent_comparison_test(tmp_path, configured_run):
    experiment, _, config, specification, snapshot = configured_run

    class CustomTest:
        specification = ComponentSpecification(
            "custom", "test.CustomTest", parameters=(("alpha", 0.1),),
        )

        def compare(self, baseline_loss, candidate_loss):
            return ComparisonOutcome(7.0, 0.1, parameters=(("draws", 100),))

    evaluation = ForecastEvaluator(
        comparison_pairs=(("EWMA", "HAR"),), comparison_test=CustomTest(),
    ).evaluate(experiment)
    paths = ForecastArtifactWriter(tmp_path / "custom_test").write(
        experiment, evaluation, config=config, dataset=snapshot, specification=specification,
    )
    metadata = json.loads(paths.experiment.read_text())
    comparisons = pd.read_csv(paths.comparisons)
    assert metadata["evaluation"]["comparison_test"]["parameters"] == {"alpha": 0.1}
    assert "hac_lags" not in metadata["evaluation"]
    assert set(comparisons["test_name"]) == {"custom"}
    assert set(comparisons["statistic"]) == {7.0}
    assert [json.loads(value) for value in comparisons["test_parameters"]] == [{"draws": 100}] * 2


def test_writer_rejects_evaluation_for_another_run_before_writing(tmp_path, configured_run):
    experiment, evaluation, config, specification, snapshot = configured_run
    output = tmp_path / "wrong_identity"
    wrong = replace(evaluation, identity=RunIdentity("another_run"))
    with pytest.raises(ValueError, match="evaluation identity must match"):
        ForecastArtifactWriter(output).write(
            experiment, wrong, config=config, dataset=snapshot, specification=specification,
        )
    assert not output.exists()
