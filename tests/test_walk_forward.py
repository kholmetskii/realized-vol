import numpy as np
import pandas as pd

from rvol.evaluation.walk_forward import walk_forward_forecasts


def feature_sample(n: int = 45, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n + 1)
    matrix = rng.normal(loc=-10.0, scale=0.5, size=(n, 3))
    target = -0.4 + matrix @ np.array([0.5, 0.3, 0.15])
    return pd.DataFrame({
        "origin_date": dates[:-1],
        "target_date": dates[1:],
        "rv_daily": matrix[:, 0],
        "rv_weekly": matrix[:, 1],
        "rv_monthly": matrix[:, 2],
        "target": target,
    })


def test_walk_forward_uses_an_expanding_training_window():
    features = feature_sample()

    forecasts = walk_forward_forecasts(features, min_train_size=20)

    assert len(forecasts) == len(features) - 20
    assert list(forecasts["n_train"]) == list(range(20, len(features)))
    assert forecasts["target_date"].is_monotonic_increasing


def test_naive_forecast_is_current_daily_value_and_variances_are_positive():
    features = feature_sample()

    forecasts = walk_forward_forecasts(features, min_train_size=20)
    expected = features.set_index("target_date").loc[
        forecasts["target_date"], "rv_daily"
    ].to_numpy()

    assert np.allclose(forecasts["naive_log_prediction"], expected)
    assert (forecasts[["actual_rv", "naive_rv_prediction", "har_rv_prediction"]] > 0).all().all()


def test_current_target_is_not_used_to_make_its_own_forecast():
    original = feature_sample()
    changed = original.copy()
    changed.loc[20, "target"] += 100.0

    before = walk_forward_forecasts(original, min_train_size=20)
    after = walk_forward_forecasts(changed, min_train_size=20)

    assert before.loc[0, "har_log_prediction"] == after.loc[0, "har_log_prediction"]
    assert before.loc[0, "actual_log_rv"] != after.loc[0, "actual_log_rv"]


def test_forecast_start_preserves_all_prior_training_data():
    features = feature_sample()
    start = features.loc[30, "target_date"]

    forecasts = walk_forward_forecasts(
        features,
        min_train_size=20,
        forecast_start=start,
    )

    assert forecasts.loc[0, "target_date"] == start
    assert forecasts.loc[0, "n_train"] == 30


def test_forecast_end_is_inclusive():
    features = feature_sample()
    end = features.loc[35, "target_date"]

    forecasts = walk_forward_forecasts(
        features,
        min_train_size=20,
        forecast_end=end,
    )

    assert forecasts["target_date"].iloc[-1] == end
    assert len(forecasts) == 16
