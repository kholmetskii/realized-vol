from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from rvol.domain import ComponentSpecification, ExperimentSpecification
from rvol.models import EWMAForecaster


def test_parameter_snapshots_are_immutable_and_describe_the_configured_model():
    model = EWMAForecaster(decay=0.8)
    original = model.specification
    changed = replace(model, decay=0.9).specification

    assert dict(original.parameters) == {"decay": 0.8}
    assert dict(changed.parameters) == {"decay": 0.9}
    with pytest.raises(FrozenInstanceError):
        cast(Any, original).parameters = (("decay", 0.1),)


@pytest.mark.parametrize(
    "parameters, message",
    [
        ((("decay", float("nan")),), "finite"),
        ((("decay", float("inf")),), "finite"),
        ((("decay", 0.8), ("decay", 0.9)), "unique"),
        ((("windows", [5, 22]),), "JSON scalar"),
    ],
)
def test_specifications_reject_ambiguous_or_non_json_parameters(parameters, message):
    with pytest.raises(ValueError, match=message):
        ComponentSpecification("model", "test.Model", parameters=cast(Any, parameters))


def test_experiment_specification_rejects_missing_feature_inputs():
    with pytest.raises(ValueError, match="missing model inputs: rv_daily"):
        ExperimentSpecification(
            features=ComponentSpecification("features", "test.Features"),
            models=(ComponentSpecification("AR1", "test.AR1", feature_names=("rv_daily",)),),
        )
