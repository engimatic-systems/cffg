"""The `.cffgz` portable model bundle (docs/ARCHITECTURE.md).

A zip containing:
- `model.json`  — canonical template serialization, blob paths rewritten to
  the bundle-internal `blobs/<name>.npy`
- `blobs/*.npy` — one array per blob
- `manifest.json` — schema version, structure hash, per-blob sha256,
  creation time, generator

Unzipping anywhere and pointing the engine at it is a full restore; loading
verifies every hash, so silent corruption or a mismatched model/blob pairing
fails loudly. The canonical schema is ours — no third-party interchange
format (decision recorded in ARCHITECTURE.md).
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from cffg.core.ir import Template
from cffg.core.serialize import (
    SCHEMA_VERSION,
    from_json,
    structure_hash,
    to_json,
)


class BundleError(ValueError):
    pass


def save_bundle(path: str | Path, template: Template, arrays: dict[str, np.ndarray]) -> None:
    """Write template + parameter arrays as a self-contained .cffgz zip."""
    missing = sorted(set(template.blobs) - set(arrays))
    if missing:
        raise BundleError(f"arrays missing for blobs: {missing}")

    bundled = Template(
        name=template.name,
        variables=dict(template.variables),
        factors=dict(template.factors),
        constraints=list(template.constraints),
        ports=dict(template.ports),
        blobs={
            name: replace(ref, path=f"blobs/{name}.npy")
            for name, ref in template.blobs.items()
        },
        schema_version=template.schema_version,
    )

    blob_sha: dict[str, str] = {}
    blob_bytes: dict[str, bytes] = {}
    for name in sorted(template.blobs):
        buf = io.BytesIO()
        np.save(buf, np.asarray(arrays[name]))
        raw = buf.getvalue()
        blob_bytes[name] = raw
        blob_sha[name] = hashlib.sha256(raw).hexdigest()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "structure_hash": structure_hash(bundled),
        "blob_sha256": blob_sha,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": "cffg",
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.json", to_json(bundled) + "\n")
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for name, raw in blob_bytes.items():
            zf.writestr(f"blobs/{name}.npy", raw)


def load_bundle(path: str | Path) -> tuple[Template, dict[str, np.ndarray], dict]:
    """Read and verify a .cffgz; returns (template, arrays, manifest)."""
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for required in ("model.json", "manifest.json"):
            if required not in names:
                raise BundleError(f"{path}: missing {required}")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise BundleError(
                f"{path}: schema {manifest.get('schema_version')!r} != {SCHEMA_VERSION!r}"
            )
        template = from_json(zf.read("model.json").decode())

        actual_hash = structure_hash(template)
        if actual_hash != manifest.get("structure_hash"):
            raise BundleError(
                f"{path}: structure hash mismatch (manifest {manifest.get('structure_hash')!r}, "
                f"model {actual_hash!r})"
            )

        arrays: dict[str, np.ndarray] = {}
        for name, ref in template.blobs.items():
            member = ref.path
            if member not in names:
                raise BundleError(f"{path}: missing blob file {member}")
            raw = zf.read(member)
            digest = hashlib.sha256(raw).hexdigest()
            want = manifest.get("blob_sha256", {}).get(name)
            if digest != want:
                raise BundleError(
                    f"{path}: blob {name!r} sha256 mismatch — corrupted or tampered"
                )
            arrays[name] = np.load(io.BytesIO(raw))
    return template, arrays, manifest
