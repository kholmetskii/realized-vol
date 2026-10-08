"""JSON normalization shared by experiment artifact writers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def configuration_payload(configuration: Mapping[str, object] | None) -> dict[str, Any] | None:
    """Copy a supplied snapshot and reject invalid JSON before writing any artifacts."""
    if configuration is None:
        return None
    return json.loads(json.dumps(dict(configuration), allow_nan=False))
