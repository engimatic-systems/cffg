"""The `.cffgz` portable model bundle (docs/ARCHITECTURE.md).

A zip containing:
- `model.json`  — canonical template serialization, blob paths rewritten to
  the bundle-internal `blobs/<name>.npy`
- `blobs/*.npy` — one array per blob
- `manifest.json` — schema version, structure hash, per-blob sha256,
  creation time, generator

``load_bundle`` verifies every hash and returns the template and arrays for
``compile_template`` and ``Engine``. An extracted bundle may instead compile
its serialized model with the extraction directory as the blob root. Silent
corruption or a mismatched model/blob pairing fails loudly. The canonical
schema is ours — no third-party interchange format (decision recorded in
ARCHITECTURE.md).
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

import numpy as np

from cffg.core.ir import Template
from cffg.core.plates import PLATE_INDEX_TOKEN, expand_plates
from cffg.core.serialize import (
    SCHEMA_VERSION,
    from_json,
    legacy_learning_structure_hash,
    structure_hash,
    to_json,
)
from cffg.core.validate import validate
from cffg.learning_schema import validate_learning_declarations


class BundleError(ValueError):
    pass


def _path_segment(value: str) -> str:
    """Encode one logical-name segment into a safe, reversible ZIP segment."""
    if value in ("", ".", ".."):
        raise BundleError(f"blob name contains unsafe path segment {value!r}")
    return quote(value, safe="-_")


def _blob_member(name: str) -> str:
    """Collision-free member path for an expanded blob/array key."""
    return "blobs/" + "/".join(_path_segment(p) for p in name.split("/")) + ".npy"


def _per_copy_pattern(plate_name: str, blob_name: str) -> str:
    plate = "/".join(_path_segment(p) for p in plate_name.split("/"))
    blob = "/".join(_path_segment(p) for p in blob_name.split("/"))
    return f"blobs/{plate}/{PLATE_INDEX_TOKEN}/{blob}.npy"


def _rewrite_blob_paths(
    template: Template,
    *,
    enclosing_plate: str | None = None,
    per_copy: frozenset[str] = frozenset(),
) -> Template:
    """Copy a template hierarchy and point every declaration into the ZIP.

    A tied fragment blob resolves to its expanded array name.  A per-copy
    declaration retains one compact path containing ``{index}``; plate
    grounding replaces that marker with the concrete 1-D/2-D copy index.
    """
    blobs = {}
    for name, ref in template.blobs.items():
        if enclosing_plate is not None and name in per_copy:
            path = _per_copy_pattern(enclosing_plate, name)
        else:
            path = _blob_member(name)
        blobs[name] = replace(ref, path=path)

    plates = {
        name: replace(
            spec,
            fragment=_rewrite_blob_paths(
                spec.fragment,
                enclosing_plate=name,
                per_copy=frozenset(spec.per_copy_blobs),
            ),
        )
        for name, spec in template.plates.items()
    }
    return Template(
        name=template.name,
        variables=dict(template.variables),
        factors=dict(template.factors),
        constraints=list(template.constraints),
        ports=dict(template.ports),
        blobs=blobs,
        plates=plates,
        learnables=dict(template.learnables),
        agent=template.agent,
        schema_version=template.schema_version,
    )


def save_bundle(path: str | Path, template: Template, arrays: dict[str, np.ndarray]) -> None:
    """Write template + parameter arrays as a self-contained .cffgz zip."""
    # Grounding is the source of truth for the concrete array namespace: tied
    # fragment blobs appear once, while per-copy blobs acquire indexed names.
    # It also validates the plate spec before the destination is opened.
    expanded = expand_plates(template)
    validate(expanded)
    required = set(expanded.blobs)
    missing = sorted(required - set(arrays))
    if missing:
        raise BundleError(f"arrays missing for blobs: {missing}")
    validate_learning_declarations(expanded, arrays)

    bundled = _rewrite_blob_paths(template)

    blob_sha: dict[str, str] = {}
    blob_bytes: dict[str, bytes] = {}
    blob_paths = {name: _blob_member(name) for name in sorted(required)}
    if len(set(blob_paths.values())) != len(blob_paths):  # pragma: no cover - defensive
        raise BundleError("expanded blob names collide after path encoding")
    for name in sorted(required):
        buf = io.BytesIO()
        np.save(buf, np.asarray(arrays[name]))
        raw = buf.getvalue()
        blob_bytes[name] = raw
        blob_sha[name] = hashlib.sha256(raw).hexdigest()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "structure_hash": structure_hash(bundled),
        "blob_sha256": blob_sha,
        "blob_paths": blob_paths,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": "cffg",
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.json", to_json(bundled) + "\n")
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for name, raw in blob_bytes.items():
            zf.writestr(blob_paths[name], raw)


def load_bundle(path: str | Path) -> tuple[Template, dict[str, np.ndarray], dict]:
    """Read and verify a .cffgz; returns (template, arrays, manifest)."""
    with zipfile.ZipFile(path) as zf:
        listed_names = zf.namelist()
        if len(listed_names) != len(set(listed_names)):
            raise BundleError(f"{path}: duplicate ZIP member names")
        names = set(listed_names)
        for required in ("model.json", "manifest.json"):
            if required not in names:
                raise BundleError(f"{path}: missing {required}")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise BundleError(
                f"{path}: schema {manifest.get('schema_version')!r} != {SCHEMA_VERSION!r}"
            )
        model_json = zf.read("model.json").decode()
        template = from_json(model_json)

        actual_hash = structure_hash(template)
        legacy_hash = (
            legacy_learning_structure_hash(template)
            if '"semantic_identity"' not in model_json else None
        )
        recorded_hash = manifest.get("structure_hash")
        if recorded_hash is None or recorded_hash not in {actual_hash, legacy_hash}:
            raise BundleError(
                f"{path}: structure hash mismatch (manifest {manifest.get('structure_hash')!r}, "
                f"model {actual_hash!r})"
            )

        arrays: dict[str, np.ndarray] = {}
        blob_paths = manifest.get("blob_paths")
        if blob_paths is None:
            # Original schema-1 flat bundles predate plates.  Their manifest
            # maps top-level blob names directly and remains fully loadable.
            if template.plates:
                raise BundleError(
                    f"{path}: plate model uses a legacy flat blob manifest; "
                    "re-export it with a plate-aware cffg version"
                )
            entries = {name: ref.path for name, ref in template.blobs.items()}
        else:
            if not isinstance(blob_paths, dict):
                raise BundleError(f"{path}: manifest blob_paths must be an object")
            if any(not isinstance(member, str) for member in blob_paths.values()):
                raise BundleError(f"{path}: manifest blob_paths values must be strings")
            if len(set(blob_paths.values())) != len(blob_paths):
                raise BundleError(f"{path}: manifest blob paths collide")
            try:
                expected = set(expand_plates(template).blobs)
            except (ValueError, KeyError) as exc:
                raise BundleError(f"{path}: invalid bundled plate structure: {exc}") from None
            declared = set(blob_paths)
            if declared != expected:
                missing = sorted(expected - declared)
                extra = sorted(declared - expected)
                raise BundleError(
                    f"{path}: blob inventory mismatch (missing {missing}, extra {extra})"
                )
            entries = blob_paths

        for name, member in sorted(entries.items()):
            if member not in names:
                raise BundleError(f"{path}: missing blob file {member}")
            raw = zf.read(member)
            digest = hashlib.sha256(raw).hexdigest()
            want = manifest.get("blob_sha256", {}).get(name)
            if digest != want:
                raise BundleError(
                    f"{path}: blob {name!r} sha256 mismatch — corrupted or tampered"
                )
            arrays[name] = np.load(io.BytesIO(raw), allow_pickle=False)
    return template, arrays, manifest
