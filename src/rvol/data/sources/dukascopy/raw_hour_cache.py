"""On-disk cache of raw hourly Dukascopy bodies."""

from __future__ import annotations

import datetime as dt
import pathlib


class RawHourCache:
    """Raw hourly bodies under ``root``, plus markers for hours known to be empty.

    A marker file means the hour is known to hold no data; a failed download
    leaves nothing behind, so it is retried next run.
    """

    def __init__(self, root: str | pathlib.Path) -> None:
        self._root = pathlib.Path(root)

    @property
    def root(self) -> pathlib.Path:
        return self._root

    def read(self, symbol: str, when: dt.datetime) -> tuple[bool, bytes | None]:
        """Look one hour up. Returns (hit, body)."""
        body_path, marker = self._paths(symbol, when)
        if marker.exists():
            return True, None
        if body_path.exists() and body_path.stat().st_size > 0:
            return True, body_path.read_bytes()
        return False, None

    def write(self, symbol: str, when: dt.datetime, raw: bytes | None) -> None:
        body_path, marker = self._paths(symbol, when)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        if raw:
            body_path.write_bytes(raw)
        else:
            marker.touch()

    def _paths(self, symbol: str, when: dt.datetime) -> tuple[pathlib.Path, pathlib.Path]:
        """Where an hour's body and its "no data here" marker live."""
        body = self._root / symbol.upper() / f"{when:%Y/%m/%d/%H}h_ticks.bi5"
        return body, body.with_suffix(".empty")
