import pytest

from rvol.evaluation import evaluate
from rvol.evaluation.robustness import WinRates, analyze_robustness
from tests.evaluation.test_evaluator import sample_forecasts


def win_rates(forecasts, diagnostic):
    return analyze_robustness(forecasts, evaluate(forecasts), (diagnostic,))["win_rates"]


def test_a_uniformly_better_candidate_wins_every_date():
    row = win_rates(sample_forecasts(80), WinRates("candidate")).iloc[0]

    assert row.baseline_model == "baseline"
    assert row.candidate_wins == row.n_obs == 80
    assert row.ties == 0 and row.candidate_win_rate == 1


def test_ties_are_not_counted_as_wins():
    forecasts = sample_forecasts()
    forecasts["predicted"] = forecasts.actual + 0.25

    row = win_rates(forecasts, WinRates("candidate")).iloc[0]

    assert row.ties == 20 and row.candidate_wins == row.candidate_win_rate == 0


@pytest.mark.parametrize(
    "options",
    [
        {"candidate": ""},
        {"candidate": "candidate", "baselines": ()},
        {"candidate": "candidate", "baselines": ("candidate",)},
        {"candidate": "candidate", "baselines": ("baseline", "baseline")},
        {"candidate": "candidate", "baselines": "baseline"},
        {"candidate": "candidate", "tie_rtol": -1},
        {"candidate": "candidate", "tie_atol": float("nan")},
    ],
)
def test_invalid_settings_are_rejected_on_construction(options):
    with pytest.raises(ValueError):
        WinRates(**options)
