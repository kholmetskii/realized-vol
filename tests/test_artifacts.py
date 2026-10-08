import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from rvol.composition import standard_experiment_definition
from rvol.domain import ExperimentConfig
from rvol.features.forecasting import HARFeatureBuilder
from rvol.models import EWMAForecaster, HARForecaster
from rvol.reporting import DatasetSnapshot, ForecastArtifactWriter


@pytest.fixture
def configured_run(tmp_path):
    step = np.arange(40, dtype="float64")
    daily = pd.DataFrame({
        "session": pd.bdate_range("2024-01-02", periods=len(step)),
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
    config = ExperimentConfig(min_train_size=10)
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

    assert metadata["schema_version"] == 2
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
