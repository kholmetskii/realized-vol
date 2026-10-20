"""Download Dukascopy tick data.

Format: https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
  - MM is zero-indexed: January = 00, December = 11. DD and HH are ordinary,
    zero-padded.
  - The body is LZMA-compressed (the .bi5 extension tells you nothing).
  - Decompressed, it is a stream of 20-byte records, big-endian:
        uint32   milliseconds since the start of the hour
        uint32   ask, integer (divide by the point divider)
        uint32   bid, integer
        float32  ask volume
        float32  bid volume
  - An empty hour (weekend, holiday) is a zero-length body or a 404.
  - Timestamps are UTC.

The server throttles and answers 503 under load, so downloads retry with
exponential backoff, a whole run is never lost to one failing hour, and the
cache lets an interrupted run continue where it stopped.

Usage:
    python -m rvol.data.sources.dukascopy EURUSD 2024-01-15 2024-01-19

Raw hourly bodies are cached under data/raw/dukascopy/<SYMBOL>/ and decoded
ticks are written to data/ticks/<SYMBOL>/ unless --cache or --out say otherwise.
"""

from rvol.data.sources.dukascopy.decoding import decode_hour, point_divider
from rvol.data.sources.dukascopy.download import fetch_range, trading_hours
from rvol.data.sources.dukascopy.dukascopy_client import DukascopyClient, hour_url
from rvol.data.sources.dukascopy.exceptions import FetchError
from rvol.data.sources.dukascopy.raw_hour_cache import RawHourCache

__all__ = [
    "DukascopyClient",
    "FetchError",
    "RawHourCache",
    "decode_hour",
    "fetch_range",
    "hour_url",
    "point_divider",
    "trading_hours",
]
