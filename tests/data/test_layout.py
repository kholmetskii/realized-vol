from pathlib import Path

from rvol.data.layout import DataLayout


def test_paths_are_below_the_root_and_named_by_upper_case_asset():
    layout = DataLayout(Path("store"))

    assert layout.root == Path("store")
    assert layout.ticks("eurusd") == Path("store/ticks/EURUSD")
    assert layout.raw_cache("dukascopy") == Path("store/raw/dukascopy")
    assert layout.daily_rv("eurusd", "5min") == Path("store/derived/EURUSD/daily_rv_5min.parquet")


def test_the_default_root_is_the_data_directory():
    assert DataLayout().ticks("GBPUSD") == Path("data/ticks/GBPUSD")
