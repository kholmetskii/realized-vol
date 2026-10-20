"""Synthetic Dukascopy bodies and HTTP responses shared by the Dukascopy tests."""

import datetime as dt
import lzma
import struct
import urllib.error

HOUR = dt.datetime(2024, 1, 15, 14)


def make_bi5(records: list[tuple[int, int, int, float, float]]) -> bytes:
    body = b"".join(struct.pack(">IIIff", *r) for r in records)
    return lzma.compress(body, format=lzma.FORMAT_ALONE)


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
