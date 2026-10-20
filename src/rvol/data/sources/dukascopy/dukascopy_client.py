"""HTTP access to the Dukascopy datafeed, with retries for throttling."""

from __future__ import annotations

import datetime as dt
import http.client
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from rvol.data.sources.dukascopy.exceptions import FetchError

BASE = "https://datafeed.dukascopy.com/datafeed"

#: Statuses worth another attempt: throttling and transient server faults.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def hour_url(symbol: str, when: dt.datetime) -> str:
    return (
        f"{BASE}/{symbol.upper()}/{when.year:04d}/{when.month - 1:02d}/"
        f"{when.day:02d}/{when.hour:02d}h_ticks.bi5"
    )


class DukascopyClient:
    """Fetch raw hourly bodies with one retry policy for every request.

    Throttling (429) and transient server faults (5xx) are retried with
    exponential backoff plus jitter. ``sleep`` is injectable so tests and
    callers can control waiting without patching the time module.
    """

    def __init__(
        self,
        retries: int = 6,
        base_delay: float = 1.0,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.retries = retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.sleep = sleep

    def fetch_hour(self, symbol: str, when: dt.datetime) -> bytes | None:
        """Fetch one hourly file.

        Returns the raw (still compressed) body, or None when the server says the
        hour does not exist — a weekend, a holiday, or before history starts.
        Anything still failing after ``retries`` attempts raises FetchError, so
        the caller can record the hour and carry on.
        """
        url = hour_url(symbol, when)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        last: Exception | None = None

        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body: bytes = resp.read()
                    return body
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                last = e
                if e.code not in RETRYABLE_STATUS:
                    raise FetchError(f"{url}: HTTP {e.code} {e.reason}") from e
            except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as e:
                last = e

            if attempt < self.retries - 1:
                # 1, 2, 4, 8, 16 ... seconds, jittered so parallel workers do not
                # retry in lockstep and trip the rate limit again.
                self.sleep(self.base_delay * 2**attempt * (1.0 + 0.25 * random.random()))

        raise FetchError(f"{url}: giving up after {self.retries} attempts ({last})")
