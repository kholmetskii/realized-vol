import numpy as np
import pytest

from rvol.evaluation import evaluate
from rvol.evaluation.robustness import LargestErrors, analyze_robustness
from tests.evaluation.test_evaluator import sample_forecasts


def test_the_largest_absolute_errors_are_listed_per_model():
    forecasts = sample_forecasts(80)
    forecasts.loc[5, "predicted"] += 10

    table = analyze_robustness(forecasts, evaluate(forecasts), (LargestErrors(top=2),))[
        "largest_errors"
    ]

    assert table.groupby("model").size().to_dict() == {"baseline": 2, "candidate": 2}
    np.testing.assert_allclose(table.absolute_error, (table.actual - table.predicted).abs())
    assert table.iloc[0].absolute_error == pytest.approx(10.5)


@pytest.mark.parametrize("top", [0, -1, 1.5, True])
def test_top_must_be_a_positive_integer(top):
    with pytest.raises(ValueError, match="top"):
        LargestErrors(top=top)
