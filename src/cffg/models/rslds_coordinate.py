"""Checkpoint-compatible coordinate views for observed-only rSLDS fitting."""

from __future__ import annotations

import numpy as np

from cffg.learn import Learnables


def legacy_snapshot(
    learning: Learnables,
    points: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Preserve the historical posterior-plus-reporting residual view."""
    return {
        **{
            f"posterior/{name}": np.asarray(value)
            for name, value in learning.hyperparameter_snapshot().items()
        },
        **{f"point/{name}": np.asarray(value) for name, value in points.items()},
    }
