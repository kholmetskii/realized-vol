"""The interface every fitted predictor implements."""

from __future__ import annotations

from typing import Protocol

from rvol.models.arrays import FloatArray


class Predictor(Protocol):
    """Return one finite prediction per feature row without changing inputs."""

    def predict(self, features: FloatArray) -> FloatArray: ...
