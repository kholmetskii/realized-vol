"""Where each asset's data files live below one data directory.

The download command line and the notebooks both use this layout, so every
path is spelled once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataLayout:
    """Downloaded ticks, the raw download cache and derived tables below ``root``."""

    root: Path = Path("data")

    def ticks(self, asset: str) -> Path:
        """Directory of an asset's tick files, one parquet file per download."""
        return self.root / "ticks" / asset.upper()

    def raw_cache(self, source: str) -> Path:
        """Directory of a data source's raw downloads, kept to avoid fetching twice."""
        return self.root / "raw" / source

    def daily_rv(self, asset: str, frequency: str) -> Path:
        """The daily realized-variance table built from an asset's ticks."""
        return self.root / "derived" / asset.upper() / f"daily_rv_{frequency}.parquet"
