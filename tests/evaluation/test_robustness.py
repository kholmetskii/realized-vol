import pandas as pd
import pytest

from rvol.evaluation import diebold_mariano, forecast_losses, hac_sensitivity, win_rates
from tests.evaluation.test_scoring import sample_forecasts
from tests.losses import squared_error


def losses_of(forecasts):
    return forecast_losses(forecasts, squared_error)


def test_a_uniformly_better_candidate_wins_every_date():
    row = win_rates(losses_of(sample_forecasts(80)), "candidate").iloc[0]

    assert row.baseline_model == "baseline"
    assert row.candidate_wins == row.n_obs == 80
    assert row.ties == 0 and row.candidate_win_rate == 1


def test_ties_are_not_counted_as_wins():
    forecasts = sample_forecasts()
    forecasts["predicted"] = forecasts.actual + 0.25

    row = win_rates(losses_of(forecasts), "candidate").iloc[0]

    assert row.ties == 20 and row.candidate_wins == row.candidate_win_rate == 0


def test_each_lag_repeats_the_dm_test_on_existing_losses():
    losses = losses_of(sample_forecasts(80))

    table = hac_sensitivity(losses, "candidate", lags=(0, 1, 3))

    assert table.hac_lags.tolist() == [0, 1, 3]
    pivot = losses.pivot(index="target_date", columns="model", values="loss")
    for row in table.itertuples():
        statistic, p_value = diebold_mariano(
            pivot.baseline, pivot.candidate, hac_lags=int(row.hac_lags)
        )
        assert row.statistic == statistic and row.p_value == p_value


@pytest.mark.parametrize("lags", [(), (0, 0), (-1,), (1.5,), (True,)])
def test_invalid_lags_are_rejected(lags):
    with pytest.raises(ValueError, match="lags"):
        hac_sensitivity(losses_of(sample_forecasts()), "candidate", lags=lags)


@pytest.mark.parametrize("options", [{"tie_rtol": -1}, {"tie_atol": float("nan")}])
def test_invalid_tie_tolerances_are_rejected(options):
    with pytest.raises(ValueError, match="tie tolerances"):
        win_rates(losses_of(sample_forecasts()), "candidate", **options)


@pytest.mark.parametrize("check", [win_rates, hac_sensitivity])
def test_checks_need_a_known_candidate_and_a_baseline(check):
    forecasts = sample_forecasts()
    with pytest.raises(ValueError, match="unknown candidate"):
        check(losses_of(forecasts), "missing")
    with pytest.raises(ValueError, match="baseline"):
        check(losses_of(forecasts.query("model == 'candidate'")), "candidate")


def test_every_other_model_is_a_baseline():
    forecasts = sample_forecasts()
    third = forecasts.query("model == 'baseline'").assign(model="other")
    losses = losses_of(pd.concat([forecasts, third], ignore_index=True))

    assert win_rates(losses, "candidate").baseline_model.tolist() == ["baseline", "other"]
