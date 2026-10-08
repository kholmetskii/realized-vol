"""Read saved experiment settings without constructing application components."""

from __future__ import annotations

import pathlib
import tomllib
from typing import Any


def load_experiment_configuration(path: str | pathlib.Path) -> dict[str, Any]:
    """Load TOML; callers validate its schema and resolve experiment components."""
    source = pathlib.Path(path)
    try:
        with source.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot read experiment configuration {source}: {error}") from error
