from dataclasses import FrozenInstanceError
from datetime import date

import numpy as np
import pytest

from rvol.domain import (
    DatasetRepository,
    ExperimentConfig,
    ExperimentResult,
    FittedForecaster,
    Forecaster,
    ForecastMetric,
    ForecastRecord,
    ModelLossSummary,
)


def assign_attribute(instance: object, name: str, value: object) -> None:
    setattr(instance, name, value)


class MeanPredictor:
    def __init__(self, value: float):
        self.value = value

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value, dtype="float64")


class MeanForecaster:
    name = "mean"
    feature_names: tuple[str, ...] = ()

    def fit(self, features: np.ndarray, target: np.ndarray) -> MeanPredictor:
        return MeanPredictor(float(np.mean(target)))


class SquaredError:
    name = "MSE"
    scale = "log_variance"

    def losses(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return np.square(actual - predicted)


class MemoryRepository:
    def __init__(self):
        self.dataset: list[float] = []

    def load(self) -> list[float]:
        return self.dataset.copy()

    def save(self, dataset: list[float]) -> None:
        self.dataset = dataset.copy()


def sample_record(model: str = "HAR") -> ForecastRecord:
    return ForecastRecord(
        model=model,
        origin_date=date(2024, 1, 2),
        target_date=date(2024, 1, 3),
        n_train=252,
        actual_log_rv=-10.0,
        predicted_log_rv=-10.1,
        actual_rv=float(np.exp(-10.0)),
        predicted_rv=float(np.exp(-10.1)),
    )


def test_structural_contracts_accept_independent_implementations():
    forecaster = MeanForecaster()
    fitted = forecaster.fit(np.empty((3, 0)), np.array([-10.0, -9.0, -8.0]))
    metric = SquaredError()
    repository = MemoryRepository()

    assert isinstance(forecaster, Forecaster)
    assert isinstance(fitted, FittedForecaster)
    assert isinstance(metric, ForecastMetric)
    assert isinstance(repository, DatasetRepository)
    assert np.allclose(fitted.predict(np.empty((2, 0))), -9.0)
    assert np.allclose(metric.losses(np.array([1.0]), np.array([0.5])), 0.25)


def test_experiment_config_is_immutable_and_validates_date_bounds():
    config = ExperimentConfig(
        min_train_size=100,
        forecast_start=date(2023, 1, 1),
        forecast_end=date(2023, 12, 31),
    )

    with pytest.raises(FrozenInstanceError):
        assign_attribute(config, "min_train_size", 10)
    with pytest.raises(ValueError, match="positive"):
        ExperimentConfig(min_train_size=0)
    with pytest.raises(ValueError, match="must not be after"):
        ExperimentConfig(
            forecast_start=date(2024, 1, 1),
            forecast_end=date(2023, 1, 1),
        )


def test_result_objects_are_immutable_and_collect_model_names():
    record = sample_record()
    loss = ModelLossSummary(model="naive", metric="QLIKE", n_obs=20, mean_loss=0.1)
    result = ExperimentResult(records=(record,), loss_summaries=(loss,))

    assert result.models == ("HAR", "naive")
    assert result.n_forecasts == 1
    with pytest.raises(FrozenInstanceError):
        assign_attribute(record, "n_train", 10)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"model": ""}, "model"),
        ({"origin_date": date(2024, 1, 3)}, "precede"),
        ({"n_train": 0}, "positive"),
        ({"actual_rv": 0.0}, "strictly positive"),
        ({"predicted_log_rv": float("nan")}, "finite"),
    ],
)
def test_forecast_record_rejects_invalid_domain_values(changes, message):
    values = {
        "model": "HAR",
        "origin_date": date(2024, 1, 2),
        "target_date": date(2024, 1, 3),
        "n_train": 252,
        "actual_log_rv": -10.0,
        "predicted_log_rv": -10.1,
        "actual_rv": float(np.exp(-10.0)),
        "predicted_rv": float(np.exp(-10.1)),
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        ForecastRecord(**values)
