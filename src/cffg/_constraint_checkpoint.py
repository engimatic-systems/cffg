"""Reconstruct fixed constraint arrays absent from older prepared caches."""

import json
from collections.abc import Collection, Mapping

import jax
import numpy as np


def legacy_constraint_keys(
    data: Mapping[str, np.ndarray],
    metadata: dict,
    expected_keys: Collection[str],
    constraint_keys: Collection[str],
) -> frozenset[str]:
    """Identify only the complete parameter layout before State owned constraints."""
    keys = frozenset(constraint_keys)
    if not keys or set(metadata) != {"params"}:
        return frozenset()
    legacy_names = sorted(set(expected_keys) - keys)
    if metadata["params"] != legacy_names:
        return frozenset()
    if any(f"mu3:state:param:{key}" in data for key in keys):
        return frozenset()
    return keys


def restore_legacy_constraint_arrays(
    data: Mapping[str, np.ndarray],
    metadata: dict,
    expected: Mapping[str, jax.Array],
    constraint_keys: Collection[str],
) -> Mapping[str, np.ndarray]:
    """Reconstruct fixed arrays on a private copy of a recognized older layout.

    The outer decoder retains its envelope and available model-data identity
    checks. The normal state decoder still checks every other field, array
    shape, dtype and cached-result invariant afterward. Legacy envelopes
    without fixed-data hashes retain their identity limits.
    """
    keys = legacy_constraint_keys(data, metadata, expected.keys(), constraint_keys)
    if not keys:
        return data
    restored = dict(data)
    for key in keys:
        restored[f"mu3:state:param:{key}"] = np.asarray(expected[key]).copy()
    restored["mu3:state:metadata"] = np.asarray(
        json.dumps({"params": sorted(expected)}, sort_keys=True)
    )
    return restored
