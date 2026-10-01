"""Parquet implementation of the dataset persistence boundary."""

from __future__ import annotations

import os
import pathlib
import tempfile
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ParquetDatasetRepository:
    """Load and atomically save one pandas dataset at a configured path."""

    path: pathlib.Path

    def __init__(self, path: str | pathlib.Path) -> None:
        object.__setattr__(self, "path", pathlib.Path(path))

    def load(self) -> pd.DataFrame:
        """Load the configured Parquet file into a data frame."""
        return pd.read_parquet(self.path)

    def save(self, dataset: pd.DataFrame) -> None:
        """Atomically replace the configured Parquet file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.stem}-",
            suffix=self.path.suffix or ".parquet",
            dir=self.path.parent,
        )
        os.close(descriptor)
        temporary = pathlib.Path(temporary_name)
        try:
            dataset.to_parquet(temporary, index=False)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
