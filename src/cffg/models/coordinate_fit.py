"""Host-side evidence for parameter fixed-point coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


ParameterBlocks = Mapping[str, Mapping[str, np.ndarray]]


@dataclass(frozen=True)
class CoordinateResidual:
    """Named residual evidence for one complete undamped parameter map."""

    maximum: float
    root_mean_square: float
    per_family: dict[str, float]
    per_array: dict[str, float]
    vector: tuple[float, ...]


def coordinate_residual(
    before: ParameterBlocks,
    after: ParameterBlocks,
) -> CoordinateResidual:
    """Compare named model parameters in deterministic family/array order.

    Each array uses ``max(1, |before|, |after|)`` as its scale. The unit term
    deliberately makes changes to probability and near-zero location
    parameters absolute, while larger model parameters use relative change.
    """
    if before.keys() != after.keys():
        raise ValueError("coordinate views disagree on parameter families")

    per_family: dict[str, float] = {}
    per_array: dict[str, float] = {}
    pieces: list[np.ndarray] = []
    for family in sorted(before):
        before_arrays = before[family]
        after_arrays = after[family]
        if before_arrays.keys() != after_arrays.keys():
            raise ValueError(
                f"coordinate family {family!r} disagrees on parameter arrays"
            )
        family_maximum = 0.0
        for name in sorted(before_arrays):
            left = np.asarray(before_arrays[name])
            right = np.asarray(after_arrays[name])
            if left.shape != right.shape:
                raise ValueError(
                    f"coordinate array {family}/{name} changed shape "
                    f"from {left.shape} to {right.shape}"
                )
            if not np.issubdtype(left.dtype, np.inexact) or not np.issubdtype(
                right.dtype, np.inexact
            ):
                raise TypeError(
                    f"coordinate array {family}/{name} must use an inexact dtype"
                )
            if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                raise ValueError(
                    f"coordinate array {family}/{name} contains a non-finite value"
                )
            scale = max(
                1.0,
                float(np.max(np.abs(left), initial=0.0)),
                float(np.max(np.abs(right), initial=0.0)),
            )
            delta = np.ravel((right - left) / scale).astype(np.float64)
            array_maximum = float(np.max(np.abs(delta), initial=0.0))
            per_array[f"{family}/{name}"] = array_maximum
            family_maximum = max(family_maximum, array_maximum)
            pieces.append(delta)
        per_family[family] = family_maximum

    vector = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)
    maximum = float(np.max(np.abs(vector), initial=0.0))
    root_mean_square = (
        float(np.sqrt(np.mean(np.square(vector)))) if vector.size else 0.0
    )
    return CoordinateResidual(
        maximum=maximum,
        root_mean_square=root_mean_square,
        per_family=per_family,
        per_array=per_array,
        vector=tuple(float(value) for value in vector),
    )
