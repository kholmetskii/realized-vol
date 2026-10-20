import numpy as np
import pandas as pd
import pytest

from rvol.data.preparation import TrailingMeanFeatures

WINDOWS = {"current": 1, "short": 5, "long": 22}


def sample(n: int = 35) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=n),
            "value": np.linspace(-12.0, -9.0, n),
        }
    )


def test_features_include_current_values_and_the_latest_origin():
    data = sample()

    features = TrailingMeanFeatures(WINDOWS)(data)

    first_origin = 21
    assert features.loc[0, "origin_date"] == data.loc[first_origin, "date"]
    assert features.loc[0, "current"] == data.loc[first_origin, "value"]
    assert len(features) == len(data) - 21
    assert features.iloc[-1]["origin_date"] == data.iloc[-1]["date"]
    assert list(features.columns) == ["origin_date", "current", "short", "long"]


def test_trailing_means_use_only_information_at_the_origin():
    data = sample()
    origin = 24

    features = TrailingMeanFeatures(WINDOWS)(data)
    row = features[features["origin_date"] == data.loc[origin, "date"]].iloc[0]

    assert np.isclose(row["short"], data.loc[origin - 4 : origin, "value"].mean())
    assert np.isclose(row["long"], data.loc[origin - 21 : origin, "value"].mean())


def test_changing_a_future_value_cannot_change_earlier_features():
    original = sample()
    changed = original.copy()
    changed_at = 30
    changed.loc[changed_at, "value"] = 100.0

    before = TrailingMeanFeatures(WINDOWS)(original)
    after = TrailingMeanFeatures(WINDOWS)(changed)
    earlier = before["origin_date"] < original.loc[changed_at, "date"]

    pd.testing.assert_frame_equal(
        before.loc[earlier].reset_index(drop=True), after.loc[earlier].reset_index(drop=True)
    )


def test_windows_count_observations_not_calendar_days():
    data = sample().drop(index=10).reset_index(drop=True)

    first = TrailingMeanFeatures(WINDOWS)(data).iloc[0]

    assert first["origin_date"] == data.loc[21, "date"]
    assert np.isclose(first["long"], data.loc[:21, "value"].mean())


def test_unordered_input_is_sorted_by_date():
    data = sample()

    shuffled = TrailingMeanFeatures(WINDOWS)(data.sample(frac=1, random_state=3))

    pd.testing.assert_frame_equal(shuffled, TrailingMeanFeatures(WINDOWS)(data))


def test_duplicate_dates_are_rejected():
    data = sample()
    data.loc[1, "date"] = data.loc[0, "date"]

    with pytest.raises(ValueError, match="one row per date"):
        TrailingMeanFeatures(WINDOWS)(data)


def test_large_gaps_are_rejected_only_when_a_limit_is_set():
    data = sample()
    data.loc[20:, "date"] += pd.Timedelta(days=30)

    with pytest.raises(ValueError, match=r"\d+-day gap"):
        TrailingMeanFeatures(WINDOWS, max_gap_days=7)(data)
    assert len(TrailingMeanFeatures(WINDOWS)(data)) == len(data) - 21


def test_configured_columns_are_used():
    data = sample(8).rename(columns={"date": "session", "value": "x"})

    features = TrailingMeanFeatures({"a": 2, "b": 4}, date_col="session", value_col="x")(data)

    assert len(features) == 5
    assert features.loc[0, "origin_date"] == data.loc[3, "session"]
    assert features.loc[0, "a"] == pytest.approx(data.loc[2:3, "x"].mean())
    assert features.loc[0, "b"] == pytest.approx(data.loc[:3, "x"].mean())


@pytest.mark.parametrize("windows", [{}, {"current": 1}])
def test_features_without_longer_windows_keep_short_histories(windows):
    data = sample(4)

    features = TrailingMeanFeatures(windows)(data)

    assert list(features.columns) == ["origin_date", *windows]
    assert features["origin_date"].equals(data["date"].rename("origin_date"))
    if windows:
        np.testing.assert_array_equal(features["current"], data["value"])


def test_only_selected_windows_determine_the_warmup():
    data = sample(8)

    features = TrailingMeanFeatures({"short": 5})(data)

    assert list(features.columns) == ["origin_date", "short"]
    assert features["origin_date"].tolist() == data.loc[4:, "date"].tolist()
    assert features.loc[0, "short"] == pytest.approx(data.loc[:4, "value"].mean())


@pytest.mark.parametrize(
    "options",
    [
        {"windows": {"a": 0}},
        {"windows": {"a": 1.5}},
        {"windows": {"a": True}},
        {"windows": {"": 1}},
        {"windows": {"origin_date": 1}},
        {"windows": {}, "date_col": "x", "value_col": "x"},
        {"windows": {}, "max_gap_days": 0},
    ],
)
def test_invalid_settings_are_rejected(options):
    with pytest.raises(ValueError):
        TrailingMeanFeatures(**options)


def test_edited_settings_are_applied_and_validated_before_building():
    data = sample(8)
    builder = TrailingMeanFeatures(WINDOWS)
    builder.windows = {"short": 2, "long": 4}

    expected = TrailingMeanFeatures({"short": 2, "long": 4})(data)
    pd.testing.assert_frame_equal(builder(data), expected)

    builder.windows = {"short": 0}
    with pytest.raises(ValueError, match="positive integer"):
        builder(data)


@pytest.mark.parametrize("column", ["date", "value"])
def test_ambiguous_duplicate_columns_are_rejected(column):
    data = sample()
    ambiguous = pd.concat([data, data[[column]]], axis=1)

    with pytest.raises(ValueError, match="columns must be unique"):
        TrailingMeanFeatures(WINDOWS)(ambiguous)


def test_missing_nullable_values_are_rejected():
    data = sample()
    data["value"] = data["value"].astype("Float64")
    data.loc[0, "value"] = pd.NA

    with pytest.raises(ValueError, match="invalid dates or 'value' values"):
        TrailingMeanFeatures(WINDOWS)(data)
