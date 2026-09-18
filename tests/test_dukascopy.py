"""Decoder tests: build a synthetic .bi5 from known values and check we get
exactly those values back."""

import datetime as dt
import lzma
import struct
import urllib.error
import urllib.request

import pandas as pd
import pytest

from rvol.data.dukascopy import (
    FetchError,
    decode_hour,
    fetch_hour,
    fetch_range,
    hour_url,
    point_divider,
    read_cache,
    trading_hours,
    write_cache,
)

HOUR = dt.datetime(2024, 1, 15, 14)


def make_bi5(records: list[tuple[int, int, int, float, float]]) -> bytes:
    body = b"".join(struct.pack(">IIIff", *r) for r in records)
    return lzma.compress(body, format=lzma.FORMAT_ALONE)


def test_decode_roundtrip():
    raw = make_bi5([
        (0, 109123, 109118, 1.5, 2.0),
        (1500, 109125, 109120, 0.7, 0.9),
        (3_599_999, 109130, 109124, 3.0, 1.1),
    ])
    df = decode_hour(raw, HOUR, point_divider("EURUSD"))

    assert len(df) == 3
    assert df["ts"].iloc[0] == pd.Timestamp("2024-01-15 14:00:00", tz="UTC")
    assert df["ts"].iloc[-1] == pd.Timestamp("2024-01-15 14:59:59.999", tz="UTC")
    assert abs(df["ask"].iloc[0] - 1.09123) < 1e-12
    assert abs(df["bid"].iloc[0] - 1.09118) < 1e-12
    assert abs(df["ask_vol"].iloc[0] - 1.5) < 1e-6
    assert (df["ask"] >= df["bid"]).all()


def test_empty_hour_is_not_an_error():
    """Weekends and holidays come back as a 404 or a zero-length body."""
    assert len(decode_hour(b"", HOUR, 1e5)) == 0
    assert len(decode_hour(make_bi5([]), HOUR, 1e5)) == 0


def test_jpy_uses_a_different_divider():
    assert point_divider("USDJPY") == 1e3
    assert point_divider("EURUSD") == 1e5


def test_url_month_is_zero_indexed():
    """The classic Dukascopy trap: month counts from zero, day and hour do not."""
    assert hour_url("EURUSD", dt.datetime(2015, 1, 15, 14)).endswith(
        "/EURUSD/2015/00/15/14h_ticks.bi5"
    )
    assert hour_url("EURUSD", dt.datetime(2015, 12, 3, 9)).endswith(
        "/EURUSD/2015/11/03/09h_ticks.bi5"
    )


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("url", code, "boom", {}, None)  # type: ignore[arg-type]


def test_404_means_no_data_not_an_error(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        raise http_error(404)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert fetch_hour("EURUSD", HOUR) is None
    assert len(calls) == 1, "a 404 must not be retried"


def test_503_is_retried_then_succeeds(monkeypatch):
    """The feed throttles under load; one 503 must not lose the run."""
    attempts = {"n": 0}

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise http_error(503)
        return FakeResponse(b"payload")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    body = fetch_hour("EURUSD", HOUR, sleep=slept.append)

    assert body == b"payload"
    assert attempts["n"] == 3
    assert len(slept) == 2 and slept[0] < slept[1], "backoff should grow"


def test_persistent_failure_raises_fetch_error(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(http_error(503)),
    )
    with pytest.raises(FetchError):
        fetch_hour("EURUSD", HOUR, retries=3, sleep=lambda _: None)


def test_client_error_is_not_retried(monkeypatch):
    attempts = {"n": 0}

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        raise http_error(403)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(FetchError):
        fetch_hour("EURUSD", HOUR, sleep=lambda _: None)
    assert attempts["n"] == 1


def test_cache_distinguishes_empty_hours_from_failures(tmp_path):
    """An hour with no data is remembered; a failed download leaves no trace."""
    assert read_cache(tmp_path, "EURUSD", HOUR) == (False, None)

    write_cache(tmp_path, "EURUSD", HOUR, None)
    assert read_cache(tmp_path, "EURUSD", HOUR) == (True, None)

    later = HOUR + dt.timedelta(hours=1)
    write_cache(tmp_path, "EURUSD", later, b"body")
    assert read_cache(tmp_path, "EURUSD", later) == (True, b"body")


def test_cached_hours_are_not_downloaded_again(tmp_path, monkeypatch):
    raw = make_bi5([(0, 109123, 109118, 1.0, 1.0)])
    for hour in range(24):
        when = HOUR.replace(hour=hour)
        write_cache(tmp_path, "EURUSD", when, raw if when == HOUR else None)

    def fake_urlopen(req, timeout=None):
        raise AssertionError("cache hit must not hit the network")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ticks, failed = fetch_range(
        "EURUSD", HOUR.date(), HOUR.date(), cache=tmp_path, progress=False,
    )
    assert failed == []
    assert len(ticks) == 1


def test_one_bad_hour_does_not_lose_the_rest(tmp_path, monkeypatch):
    good = make_bi5([(0, 109123, 109118, 1.0, 1.0)])
    for hour in range(24):
        if hour != 14:
            write_cache(tmp_path, "EURUSD", HOUR.replace(hour=hour), good)

    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(http_error(503)),
    )
    monkeypatch.setattr("rvol.data.dukascopy.time.sleep", lambda _: None)

    ticks, failed = fetch_range(
        "EURUSD", HOUR.date(), HOUR.date(), cache=tmp_path,
        max_workers=2, retries=2, progress=False,
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
