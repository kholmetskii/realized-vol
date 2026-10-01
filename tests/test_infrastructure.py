import pathlib

import pandas as pd
import pytest

from rvol.domain import DatasetRepository
from rvol.infrastructure import ParquetDatasetRepository


def test_parquet_repository_round_trip_creates_parent_directory(tmp_path):
    destination = tmp_path / "derived" / "daily.parquet"
    expected = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "rv_5min": [1.2e-5, 1.4e-5],
    })
    repository = ParquetDatasetRepository(destination)

    assert isinstance(repository, DatasetRepository)
    repository.save(expected)

    pd.testing.assert_frame_equal(repository.load(), expected)


def test_parquet_repository_preserves_existing_file_when_write_fails(tmp_path, monkeypatch):
    destination = tmp_path / "daily.parquet"
    repository = ParquetDatasetRepository(destination)
    original = pd.DataFrame({"value": [1.0]})
    repository.save(original)

    def fail_after_partial_write(
        self: pd.DataFrame,
        path: pathlib.Path,
        *,
        index: bool,
    ) -> None:
        path.write_bytes(b"incomplete")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="simulated write failure"):
        repository.save(pd.DataFrame({"value": [2.0]}))

    monkeypatch.undo()
    pd.testing.assert_frame_equal(repository.load(), original)
    assert list(tmp_path.glob(".daily-*")) == []
