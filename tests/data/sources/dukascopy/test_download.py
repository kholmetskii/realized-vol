import datetime as dt
import urllib.request

from rvol.data.sources.dukascopy import (
    DukascopyClient,
    RawHourCache,
    fetch_range,
    trading_hours,
)
from tests.data.sources.dukascopy.fakes import HOUR, http_error, make_bi5


def test_cached_hours_are_not_downloaded_again(tmp_path, monkeypatch):
    cache = RawHourCache(tmp_path)
    raw = make_bi5([(0, 109123, 109118, 1.0, 1.0)])
    for hour in range(24):
        when = HOUR.replace(hour=hour)
        cache.write("EURUSD", when, raw if when == HOUR else None)

    def fake_urlopen(req, timeout=None):
        raise AssertionError("cache hit must not hit the network")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ticks, failed = fetch_range(
        "EURUSD", HOUR.date(), HOUR.date(), cache=cache, progress=False,
    )
    assert failed == []
    assert len(ticks) == 1


def test_one_bad_hour_does_not_lose_the_rest(tmp_path, monkeypatch):
    cache = RawHourCache(tmp_path)
    good = make_bi5([(0, 109123, 109118, 1.0, 1.0)])
    for hour in range(24):
        if hour != 14:
            cache.write("EURUSD", HOUR.replace(hour=hour), good)

    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(http_error(503)),
    )

    ticks, failed = fetch_range(
        "EURUSD", HOUR.date(), HOUR.date(),
        client=DukascopyClient(retries=2, sleep=lambda _: None),
        cache=cache, max_workers=2, progress=False,
    )
    assert failed == [HOUR]
    assert len(ticks) == 23


def test_saturday_is_skipped_but_sunday_is_not():
    """The trading week opens on Sunday evening UTC, so Sunday must be fetched."""
    saturday = dt.date(2024, 1, 13)
    sunday = dt.date(2024, 1, 14)
    hours = trading_hours(saturday, sunday)
    assert {h.date() for h in hours} == {sunday}
    assert len(hours) == 24
