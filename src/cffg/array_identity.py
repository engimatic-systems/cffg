"""Numeric content descriptors shared by portable bundles and proposal records."""

import hashlib
import json


def array_descriptor(value):
    """Identify the exact shape, dtype and logical C-order bytes of an array."""
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def json_digest(value):
    """Hash canonical JSON records, preserving sequence order."""
    data = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()
