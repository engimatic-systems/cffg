"""Numeric content descriptors shared by portable bundles and proposal records."""

import hashlib


def array_descriptor(value):
    """Identify the exact shape, dtype and logical C-order bytes of an array."""
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }
