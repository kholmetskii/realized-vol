import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rvol.composition import standard_experiment_definition
from rvol.domain import EveryNSessions, ExperimentConfig, RollingWindow, WalkForwardStrategy
from rvol.features.forecasting import HARFeatureBuilder
from rvol.models import EWMAForecaster, HARForecaster
from rvol.reporting import DatasetSnapshot, ForecastArtifactWriter


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
        comparison_pairs=(("EWMA", "HAR"),),
        baseline_models=("EWMA",),
    )
    config = ExperimentConfig(
        min_train_size=10,
        forecast_start=dates[22].date(),
        forecast_end=dates[30].date(),
        strategy=WalkForwardStrategy(RollingWindow(12), EveryNSessions(3)),
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

    assert metadata["schema_version"] == 3
    assert experiment.strategy_anchor is not None
    assert metadata["execution"] == {
        "implementation": "rvol.domain.policies.WalkForwardStrategy",
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
        "schedule_anchor": experiment.strategy_anchor.isoformat(),
        "schedule_anchor_policy": "first_eligible_origin",
        "observation_update_policy": "newly_available_targets_between_refits",
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
    models = metadata["specification"]["models"]
    assert [model["name"] for model in models] == ["EWMA", "HAR"]
    assert models[0]["implementation"] == "rvol.models.ewma.EWMAForecaster"
    assert models[0]["parameters"] == {"decay": 0.8}
    assert models[1]["implementation"] == "rvol.models.har.HARForecaster"
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


@pytest.mark.parametrize("missing", ["execution_specification", "strategy_anchor"])
def test_writer_requires_execution_metadata_before_writing(tmp_path, configured_run, missing):
    experiment, evaluation, config, specification, snapshot = configured_run
    output = tmp_path / "artifacts"
    incomplete = replace(experiment, **{missing: None})

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
            experiment, evaluation, config=replace(config, min_train_size=11),
            dataset=snapshot, specification=specification,
        )

    assert not output.exists()


def test_writer_rejects_changed_execution_settings_before_writing(tmp_path, configured_run):
    experiment, evaluation, config, specification, snapshot = configured_run
    output = tmp_path / "artifacts"
    inconsistent = replace(
        experiment, execution_specification=WalkForwardStrategy().specification,
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
