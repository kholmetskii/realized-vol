"""Atomic file helpers shared by reporting adapters."""

from __future__ import annotations

import os
import pathlib
import tempfile

import pandas as pd


def atomic_csv(frame: pd.DataFrame, destination: pathlib.Path) -> None:
    """Atomically write a deterministic CSV file."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        frame.to_csv(
            temporary,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(content: str, destination: pathlib.Path) -> None:
    """Atomically write UTF-8 text."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=destination.suffix,
        dir=destination.parent,
        text=True,
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
