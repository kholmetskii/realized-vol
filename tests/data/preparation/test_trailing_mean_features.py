import dataclasses

import numpy as np
import pandas as pd
import pytest

from rvol.data.preparation import TrailingMeanFeatures

MEANS = TrailingMeanFeatures("value", (1, 5, 22))


def sample(n: int = 35) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=n),
            "value": np.linspace(-12.0, -9.0, n),
        }
    )


def test_columns_are_named_by_series_and_window_in_window_order():
    assert MEANS.columns == ("value_mean_1", "value_mean_5", "value_mean_22")
    assert MEANS.column(5) == "value_mean_5"
    assert list(MEANS(sample()).columns) == ["origin_date", *MEANS.columns]


def test_only_existing_windows_have_a_column():
    with pytest.raises(ValueError, match="no window of 2"):
        MEANS.column(2)


def test_features_include_current_values_and_the_latest_origin():
    data = sample()

    features = MEANS(data)

    first_origin = 21
    assert features.loc[0, "origin_date"] == data.loc[first_origin, "date"]
    assert features.loc[0, "value_mean_1"] == data.loc[first_origin, "value"]
    assert len(features) == len(data) - 21
    assert features.iloc[-1]["origin_date"] == data.iloc[-1]["date"]


def test_trailing_means_use_only_information_at_the_origin():
    data = sample()
    origin = 24

    features = MEANS(data)
    row = features[features["origin_date"] == data.loc[origin, "date"]].iloc[0]

    assert np.isclose(row["value_mean_5"], data.loc[origin - 4 : origin, "value"].mean())
    assert np.isclose(row["value_mean_22"], data.loc[origin - 21 : origin, "value"].mean())


def test_changing_a_future_value_cannot_change_earlier_features():
    original = sample()
    changed = original.copy()
    changed_at = 30
    changed.loc[changed_at, "value"] = 100.0

    before = MEANS(original)
    after = MEANS(changed)
    earlier = before["origin_date"] < original.loc[changed_at, "date"]

    pd.testing.assert_frame_equal(
        before.loc[earlier].reset_index(drop=True), after.loc[earlier].reset_index(drop=True)
    )


def test_windows_count_observations_not_calendar_days():
    data = sample().drop(index=10).reset_index(drop=True)

    first = MEANS(data).iloc[0]

    assert first["origin_date"] == data.loc[21, "date"]
    assert np.isclose(first["value_mean_22"], data.loc[:21, "value"].mean())


def test_unordered_input_is_sorted_by_date():
    data = sample()

    shuffled = MEANS(data.sample(frac=1, random_state=3))

    pd.testing.assert_frame_equal(shuffled, MEANS(data))


def test_duplicate_dates_are_rejected():
    data = sample()
    data.loc[1, "date"] = data.loc[0, "date"]

    with pytest.raises(ValueError, match="one row per date"):
        MEANS(data)


def test_large_gaps_are_rejected_only_when_a_limit_is_set():
    data = sample()
    data.loc[20:, "date"] += pd.Timedelta(days=30)

    with pytest.raises(ValueError, match=r"\d+-day gap"):
        dataclasses.replace(MEANS, max_gap_days=7)(data)
    assert len(MEANS(data)) == len(data) - 21


def test_configured_columns_are_used():
    data = sample(8).rename(columns={"date": "session", "value": "x"})

    features = TrailingMeanFeatures("x", (2, 4), date_col="session")(data)

    assert list(features.columns) == ["origin_date", "x_mean_2", "x_mean_4"]
    assert len(features) == 5
    assert features.loc[0, "origin_date"] == data.loc[3, "session"]
    assert features.loc[0, "x_mean_2"] == pytest.approx(data.loc[2:3, "x"].mean())
    assert features.loc[0, "x_mean_4"] == pytest.approx(data.loc[:3, "x"].mean())


@pytest.mark.parametrize("windows", [(), (1,)])
def test_features_without_longer_windows_keep_short_histories(windows):
    data = sample(4)

    features = TrailingMeanFeatures("value", windows)(data)

    assert features["origin_date"].equals(data["date"].rename("origin_date"))
    if windows:
        np.testing.assert_array_equal(features["value_mean_1"], data["value"])


def test_only_selected_windows_determine_the_warmup():
    data = sample(8)

    features = TrailingMeanFeatures("value", (5,))(data)

    assert list(features.columns) == ["origin_date", "value_mean_5"]
    assert features["origin_date"].tolist() == data.loc[4:, "date"].tolist()
    assert features.loc[0, "value_mean_5"] == pytest.approx(data.loc[:4, "value"].mean())


@pytest.mark.parametrize(
    "options",
    [
        {"windows": (0,)},
        {"windows": (1.5,)},
        {"windows": (True,)},
        {"windows": (5, 1)},
        {"windows": (1, 1)},
        {"value_col": ""},
        {"value_col": "date"},
        {"max_gap_days": 0},
    ],
)
def test_invalid_settings_are_rejected(options):
    settings = {"value_col": "value", "windows": (1, 5)} | options
    with pytest.raises(ValueError):
        TrailingMeanFeatures(**settings)


def test_settings_cannot_be_edited_after_validation():
    with pytest.raises(dataclasses.FrozenInstanceError):
        MEANS.windows = (0,)  # type: ignore[misc]


@pytest.mark.parametrize("column", ["date", "value"])
def test_ambiguous_duplicate_columns_are_rejected(column):
    data = sample()
    ambiguous = pd.concat([data, data[[column]]], axis=1)

    with pytest.raises(ValueError, match="columns must be unique"):
        MEANS(ambiguous)


def test_missing_nullable_values_are_rejected():
    data = sample()
    data["value"] = data["value"].astype("Float64")
    data.loc[0, "value"] = pd.NA

    with pytest.raises(ValueError, match="invalid dates or 'value' values"):
        MEANS(data)
