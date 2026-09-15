"""Strict, versioned numeric artifacts for deterministic neural workflows."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def array_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        descriptor = json.dumps([name, value.dtype.str, value.shape], separators=(",", ":"))
        digest.update(descriptor.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def write_artifact(path, metadata, arrays):
    """Publish a complete snapshot atomically; never serialize Python objects."""
    path = Path(path)
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_artifact(path, *, format_name, fields):
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("incompatible neural checkpoint: missing versioned metadata; legacy migration required")
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != {"metadata", *fields}:
            raise ValueError("neural checkpoint array inventory mismatch")
        raw = archive["metadata"]
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("neural checkpoint metadata must be a scalar JSON string")
        metadata = json.loads(str(raw))
        if (not isinstance(metadata, dict) or metadata.get("format") != format_name
                or type(metadata.get("version")) is not int or metadata["version"] != 1):
            raise ValueError("incompatible neural checkpoint format or version")
        arrays = {name: archive[name].copy() for name in fields}
    return metadata, arrays
