import datetime as dt
import urllib.request

import pytest

from rvol.data.sources.dukascopy import DukascopyClient, FetchError, hour_url
from tests.data.sources.dukascopy.fakes import HOUR, FakeResponse, http_error


def no_sleep(_: float) -> None:
    return None


def test_url_month_is_zero_indexed():
    """The classic Dukascopy trap: month counts from zero, day and hour do not."""
    assert hour_url("EURUSD", dt.datetime(2015, 1, 15, 14)).endswith(
        "/EURUSD/2015/00/15/14h_ticks.bi5"
    )
    assert hour_url("EURUSD", dt.datetime(2015, 12, 3, 9)).endswith(
        "/EURUSD/2015/11/03/09h_ticks.bi5"
    )


def test_404_means_no_data_not_an_error(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        raise http_error(404)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert DukascopyClient().fetch_hour("EURUSD", HOUR) is None
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
    body = DukascopyClient(sleep=slept.append).fetch_hour("EURUSD", HOUR)

    assert body == b"payload"
    assert attempts["n"] == 3
    assert len(slept) == 2 and slept[0] < slept[1], "backoff should grow"


def test_persistent_failure_raises_fetch_error(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(http_error(503)),
    )
    with pytest.raises(FetchError):
        DukascopyClient(retries=3, sleep=no_sleep).fetch_hour("EURUSD", HOUR)


def test_client_error_is_not_retried(monkeypatch):
    attempts = {"n": 0}

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        raise http_error(403)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(FetchError):
        DukascopyClient(sleep=no_sleep).fetch_hour("EURUSD", HOUR)
    assert attempts["n"] == 1


def test_configured_timeout_reaches_every_request(monkeypatch):
    timeouts = []

    def fake_urlopen(req, timeout=None):
        timeouts.append(timeout)
        return FakeResponse(b"payload")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    DukascopyClient(timeout=5.0).fetch_hour("EURUSD", HOUR)

    assert timeouts == [5.0]
