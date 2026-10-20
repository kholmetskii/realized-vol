import pytest

from rvol.evaluation import ComparisonOutcome


@pytest.mark.parametrize("statistic, p_value", [(float("nan"), 0.5), (0, -0.1), (0, 1.1)])
def test_comparison_outcomes_reject_invalid_values(statistic, p_value):
    with pytest.raises(ValueError):
        ComparisonOutcome(statistic, p_value)


def test_default_comparison_parameters_are_independent_for_each_result():
    first = ComparisonOutcome(0.0, 0.5)
    second = ComparisonOutcome(0.0, 0.5)
    first.parameters["option"] = 1

    assert second.parameters == {}
