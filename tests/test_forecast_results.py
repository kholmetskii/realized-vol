import pandas as pd
import pytest

from rvol.backtesting.results import validated_forecasts


@pytest.fixture
def forecast_table():
    return pd.DataFrame(
        {
            "model": ["AR1", "AR1", "HAR", "HAR"],
            "origin_date": pd.to_datetime(["2026-10-01", "2026-10-02"] * 2),
            "target_date": pd.to_datetime(["2026-10-02", "2026-10-03"] * 2),
            "actual": [10.0, 20.0, 10.0, 20.0],
            "predicted": [11.0, 19.0, 12.0, 18.0],
        }
    )


@pytest.mark.parametrize("difference", ["dates", "actual", "origin"])
def test_forecast_records_allow_models_with_different_samples(forecast_table, difference):
    selected = forecast_table.model == "HAR"
    if difference == "dates":
        forecast_table.loc[selected, ["origin_date", "target_date"]] += pd.Timedelta(days=30)
    elif difference == "actual":
        forecast_table.loc[selected, "actual"] += 100.0
    else:
        forecast_table.loc[selected, "origin_date"] -= pd.Timedelta(days=1)
    original = forecast_table.copy(deep=True)

    validated = validated_forecasts(forecast_table)

    pd.testing.assert_frame_equal(validated[original.columns], original)
    assert validated.run_id.tolist() == ["default"] * 4
    assert validated.fold_id.isna().all()
    pd.testing.assert_frame_equal(forecast_table, original)


@pytest.mark.parametrize("identity", ["run_id", "fold_id"])
def test_overlapping_dates_are_unique_within_each_run_fold(forecast_table, identity):
    first = forecast_table.assign(run_id="first", fold_id=None)
    if identity == "fold_id":
        first["fold_id"] = "first_fold"
    second = first.copy()
    second[identity] = "second"
    combined = pd.concat([first, second], ignore_index=True)

    validated = validated_forecasts(combined)

    assert len(validated) == 8
    assert len(validated[["run_id", "fold_id"]].drop_duplicates()) == 2
    duplicate = pd.concat([combined, first.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="within a run/fold"):
        validated_forecasts(duplicate)
