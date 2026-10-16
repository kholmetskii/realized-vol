"""Resolve declarative experiment settings into the existing component interfaces."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import Any, TypeVar

from rvol.domain import (
    EveryNSessions,
    ExpandingWindow,
    ExperimentConfig,
    HoldoutSettings,
    RollingWindow,
    RunIdentity,
    TargetHorizon,
    WalkForwardSettings,
)
from rvol.domain.contracts import ForecastComparisonTest, Forecaster, ForecastMetric
from rvol.forecasting.composition.forecasting import (
    Experiment,
    standard_diagnostic_checks,
    standard_experiment,
)
from rvol.forecasting.evaluation import (
    DieboldMarianoTest,
    LagSensitivityCheck,
    LargestErrorsCheck,
    LogMSEMetric,
    QLikeMetric,
)
from rvol.forecasting.features import HARFeatureBuilder
from rvol.forecasting.models import (
    AR1Forecaster,
    EWMAForecaster,
    HARForecaster,
    HistoricalMeanForecaster,
    NaiveForecaster,
)
from rvol.forecasting.targets import RealizedVarianceTargetBuilder

T = TypeVar("T")
MODEL_FACTORIES: Mapping[str, Callable[..., Forecaster]] = {
    "historical_mean": HistoricalMeanForecaster, "naive": NaiveForecaster,
    "EWMA": EWMAForecaster, "AR1": AR1Forecaster, "HAR": HARForecaster,
}
METRIC_FACTORIES: Mapping[str, Callable[[], ForecastMetric]] = {
    "QLIKE": QLikeMetric, "log-RV MSE": LogMSEMetric,
}
FEATURE_FACTORIES = {"HAR": HARFeatureBuilder}
TARGET_FACTORIES = {"realized_variance": RealizedVarianceTargetBuilder}
TEST_FACTORIES: Mapping[str, Callable[..., ForecastComparisonTest]] = {"DM": DieboldMarianoTest}


def _keys(values: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(values).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {location} settings: {', '.join(sorted(unknown))}")


def _table(
    document: Mapping[str, Any], name: str, allowed: set[str] | None = None,
) -> dict[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a table")
    if allowed is not None:
        _keys(value, allowed, name)
    return value.copy()


def _names(value: Any, location: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{location} must be an array of non-empty names")
    if len(set(value)) != len(value):
        raise ValueError(f"{location} names must be unique")
    return tuple(value)


def _date(value: Any, location: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"{location} must be a date in YYYY-MM-DD format")


def _component(
    values: dict[str, Any], factories: Mapping[str, Callable[..., T]],
    location: str, default_name: str, defaults: dict[str, Any] | None = None,
) -> T:
    settings = values.copy()
    name = settings.pop("name", default_name)
    if not isinstance(name, str) or name not in factories:
        raise ValueError(f"unknown {location} component: {name!r}")
    factory = factories[name]
    parameters = inspect.signature(factory).parameters
    _keys(settings, set(parameters), location)
    for key, value in settings.items():
        if isinstance(value, bool) and key in {
            "decay", "variance_multiplier", "weekly_window", "monthly_window", "max_gap_days",
            "hac_lags", "min_hac_lags",
        }:
            raise ValueError(f"{location}.{key} must be numeric, not boolean")
    try:
        return factory(**((defaults or {}) | settings))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {location} settings: {error}") from error


def experiment_from_mapping(
    document: Mapping[str, Any],
) -> Experiment:
    """Resolve defaults and a validated document without reading a dataset.

    Names select registered built-in components; Python callers can continue
    constructing Experiment directly with independent models and builders.
    """
    _keys(document, {
        "version", "dataset", "run", "target", "execution", "features", "models",
        "evaluation", "robustness", "plot",
    }, "configuration")
    version = document.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("configuration version must be 1")
    dataset = _table(document, "dataset", {"instrument"})
    run = _table(document, "run", {"run_id", "fold_id", "forecast_start", "forecast_end"})
    target_values = _table(document, "target")
    horizon = TargetHorizon(target_values.pop("sessions", 1), target_values.pop("kind", "session"))
    if "horizon" in target_values:
        raise ValueError("configure target horizon using target.sessions and target.kind")
    target = _component(
        target_values, TARGET_FACTORIES, "target", "realized_variance", {"horizon": horizon},
    )
    defaults = standard_experiment(target_horizon=horizon)
    feature_values = _table(document, "features")
    check_gaps = feature_values.pop("check_gaps", True)
    if not isinstance(check_gaps, bool):
        raise ValueError("features.check_gaps must be boolean")
    if "horizon" in feature_values:
        raise ValueError("configure the forecast horizon in the target table")
    if not check_gaps:
        feature_values["max_gap_days"] = None
    features = _component(
        feature_values, FEATURE_FACTORIES, "features", "HAR",
    )

    if "models" not in document:
        models = defaults.models
    else:
        values = document["models"]
        if not isinstance(values, list) or not values or any(
            not isinstance(value, dict) or "name" not in value for value in values
        ):
            raise ValueError("models must be a non-empty array of tables with a name")
        models = tuple(_component(
            value, MODEL_FACTORIES, "model", "", {
                "variance_multiplier": float(horizon.sessions) if horizon.kind == "cumulative"
                else 1.0,
            } if value["name"] == "naive" else None,
        ) for value in values)
    model_names = {model.name for model in models}
    evaluation = _table(document, "evaluation", {
        "metrics", "candidate", "baselines", "comparison_pairs", "test",
    })
    metric_names = _names(evaluation.get("metrics", [metric.name for metric in defaults.metrics]),
                          "evaluation.metrics")
    if set(metric_names).difference(METRIC_FACTORIES):
        raise ValueError("unknown evaluation metric")
    metrics = tuple(METRIC_FACTORIES[name]() for name in metric_names)
    candidate = evaluation.get("candidate", defaults.candidate_model)
    if not isinstance(candidate, str) or candidate not in model_names:
        raise ValueError("evaluation.candidate must name a selected model")
    baselines = _names(evaluation.get("baselines", [
        name for name in (*defaults.baseline_models, defaults.candidate_model)
        if name in model_names and name != candidate
    ]), "evaluation.baselines", allow_empty=True)
    default_pairs = [[baselines[0], candidate]] if baselines else []
    pairs = evaluation.get("comparison_pairs", default_pairs)
    if not isinstance(pairs, list) or any(
        not isinstance(pair, list) or len(pair) != 2 or any(
            not isinstance(name, str) for name in pair
        ) for pair in pairs
    ):
        raise ValueError("evaluation.comparison_pairs must contain [baseline, candidate] pairs")
    test_values = _table(evaluation, "test")
    test = _component(test_values, TEST_FACTORIES, "evaluation.test", "DM", {
        "min_hac_lags": dict(defaults.comparison_test.specification.parameters)["min_hac_lags"],
    })
    robust = _table(document, "robustness", {"checks", "hac_lags", "top_errors"})
    check_names = None if "checks" not in robust else _names(robust["checks"], "robustness.checks")
    if check_names is None and not baselines:
        check_names = ("monthly_losses", "largest_errors")
    lags = robust.get("hac_lags", list(range(6)))
    if not isinstance(lags, list) or any(
        isinstance(lag, bool) or not isinstance(lag, int) or lag < 0 for lag in lags
    ):
        raise ValueError("robustness.hac_lags must be an array of non-negative integers")
    LagSensitivityCheck(tuple(lags))
    LargestErrorsCheck(robust.get("top_errors", 5))
    checks = standard_diagnostic_checks(
        check_names, hac_lags=lags, top_errors=robust.get("top_errors", 5),
    )
    if not baselines and any(check.specification.name in {"hac_sensitivity", "win_rates"}
                             for check in checks):
        raise ValueError("comparison diagnostics require at least one baseline model")
    plot = _table(document, "plot", {"metric"})
    execution = _table(document, "execution", {
        "name", "min_train_size", "training_window", "window_size", "retrain_every",
        "train_end", "update_observations",
    })

    return Experiment(
        config=ExperimentConfig(
            execution=_execution_settings(execution),
            forecast_start=_date(run.get("forecast_start"), "run.forecast_start"),
            forecast_end=_date(run.get("forecast_end"), "run.forecast_end"),
            identity=RunIdentity(run.get("run_id", "default"), run.get("fold_id")),
            instrument=dataset.get("instrument"),
        ),
        models=models, feature_builder=features, target_builder=target, metrics=metrics,
        comparison_pairs=tuple((pair[0], pair[1]) for pair in pairs),
        candidate_model=candidate, baseline_models=baselines, comparison_test=test,
        diagnostic_checks=checks, plot_metric=plot.get("metric"),
    )


def _execution_settings(execution: Mapping[str, Any]) -> WalkForwardSettings | HoldoutSettings:
    """Parse the two supported chronological execution strategies."""
    name = execution.get("name", "walk-forward")
    if name == "holdout":
        for key in ("training_window", "window_size", "retrain_every"):
            if key in execution:
                raise ValueError(f"execution.{key} requires execution.name = 'walk-forward'")
        train_end = _date(execution.get("train_end"), "execution.train_end")
        if train_end is None:
            raise ValueError("execution.train_end is required when execution.name = 'holdout'")
        return HoldoutSettings(
            train_end, execution.get("min_train_size", HoldoutSettings(train_end).min_train_size),
            execution.get("update_observations", False),
        )
    if name != "walk-forward":
        raise ValueError("execution.name must be walk-forward or holdout")
    if "train_end" in execution:
        raise ValueError("execution.train_end requires execution.name = 'holdout'")
    if "update_observations" in execution:
        raise ValueError("execution.update_observations requires execution.name = 'holdout'")
    window_name = execution.get("training_window", "expanding")
    if window_name not in ("expanding", "rolling"):
        raise ValueError("execution.training_window must be expanding or rolling")
    if window_name == "rolling" and "window_size" not in execution:
        raise ValueError(
            "execution.window_size is required when execution.training_window = 'rolling'"
        )
    if window_name != "rolling" and "window_size" in execution:
        raise ValueError("execution.window_size requires execution.training_window = 'rolling'")
    window = (
        RollingWindow(execution["window_size"]) if window_name == "rolling" else ExpandingWindow()
    )
    return WalkForwardSettings(
        window, EveryNSessions(execution.get("retrain_every", 1)),
        execution.get("min_train_size", WalkForwardSettings().min_train_size),
    )
