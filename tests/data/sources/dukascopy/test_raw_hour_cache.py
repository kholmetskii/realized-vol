import datetime as dt

from rvol.data.sources.dukascopy import RawHourCache
from tests.data.sources.dukascopy.fakes import HOUR


def test_cache_distinguishes_empty_hours_from_failures(tmp_path):
    """An hour with no data is remembered; a failed download leaves no trace."""
    cache = RawHourCache(tmp_path)
    assert cache.read("EURUSD", HOUR) == (False, None)

    cache.write("EURUSD", HOUR, None)
    assert cache.read("EURUSD", HOUR) == (True, None)

    later = HOUR + dt.timedelta(hours=1)
    cache.write("EURUSD", later, b"body")
    assert cache.read("EURUSD", later) == (True, b"body")


def test_cache_files_live_under_the_symbol_and_hour(tmp_path):
    cache = RawHourCache(tmp_path)
    cache.write("eurusd", HOUR, b"body")
    cache.write("eurusd", HOUR + dt.timedelta(hours=1), None)

    assert (tmp_path / "EURUSD/2024/01/15/14h_ticks.bi5").read_bytes() == b"body"
    assert (tmp_path / "EURUSD/2024/01/15/15h_ticks.empty").is_file()
    assert cache.root == tmp_path
