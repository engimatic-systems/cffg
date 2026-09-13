"""Portable model and declared-learning archives, validated before publication.

Schema 2 stores model.json, point blobs, complete immutable priors and current
posteriors, and a manifest binding logical ownership to exact array bytes.
Earlier bundle schemas are rejected; no missing learning history is inferred.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
import datetime
import hashlib
import io
import json
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from urllib.parse import quote

import numpy as np

from cffg.core.ir import Template
from cffg.core.plates import PLATE_INDEX_TOKEN, expand_plates
from cffg.core.serialize import (
    document_structure_hash,
    from_dict,
    structure_hash,
    to_json,
)
from cffg.core.validate import validate
from cffg.learning_schema import (
    encode_persistent_manifest, validate_learning_declarations,
    validate_persistent_manifest,
)

if TYPE_CHECKING:
    from cffg.learn import Learnables

BUNDLE_SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class LoadedBundle:
    """Validated model, point arrays and complete learning history.

    Tuple unpacking retains the historical (template, arrays, manifest) API.
    Learning state is always an explicit field, never a point-array entry.
    """

    template: Template
    arrays: dict[str, np.ndarray]
    manifest: dict
    learnables: Learnables

    @property
    def resumable(self) -> bool:
        return self.manifest["resumable"]

    def __iter__(self):
        yield self.template
        yield self.arrays
        yield self.manifest


class BundleError(ValueError):
    pass


def _path_segment(value: str) -> str:
    """Encode one logical-name segment into a safe, reversible ZIP segment."""
    if value in ("", ".", "..") or "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
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


def _npy(value):
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _array_descriptor(value):
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def _point_arrays(template, arrays):
    expected = set(template.blobs)
    if set(arrays) != expected:
        raise BundleError(
            f"blob inventory mismatch: arrays missing for blobs: {sorted(expected - set(arrays))}, "
            f"extra {sorted(set(arrays) - expected)}"
        )
    for name, value in arrays.items():
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise BundleError(f"blob {name!r}: expected finite real numeric values")


def _learning_snapshot(learning):
    if learning is None:
        return json.loads(encode_persistent_manifest(()).item()), {}
    descriptors = learning.persistent_descriptors()
    if set(learning.params) != set(learning.plans) or set(learning.initial_params) != set(learning.plans):
        raise BundleError("learning state inventory does not match declared owners")
    document = json.loads(encode_persistent_manifest(descriptors).item())
    values = {}
    for descriptor in descriptors:
        for row in descriptor.arrays:
            state = learning.initial_params if row.role == "prior" else learning.params
            value = np.asarray(state[descriptor.name][row.key]).copy()
            if _array_descriptor(value) != {
                "shape": list(row.shape), "dtype": row.dtype, "sha256": row.sha256,
            }:
                raise BundleError("learning state changed while taking its snapshot")
            values[row.name] = value
    return document, values


def save_bundle(
    path: str | Path,
    template: Template,
    arrays: dict[str, np.ndarray],
    *,
    learnables: Learnables | None = None,
) -> None:
    """Atomically save point arrays and complete declared-learning state.

    Pass ``learnables`` whenever the expanded template declares learning.
    Point arrays are the serialized model parameters supplied by the caller;
    they may be initialization points or values exported after a learning trial.
    Public learner operations are excluded during snapshot capture. Callers
    must synchronize direct edits to their template, arrays or learner mappings.
    """
    with learnables._lock if learnables is not None else nullcontext():
        template = copy.deepcopy(template)
        arrays = {name: np.asarray(value).copy() for name, value in arrays.items()}
        expanded = expand_plates(template)
        validate(expanded)
        _point_arrays(expanded, arrays)
        validate_learning_declarations(expanded, arrays)
        if expanded.learnables and learnables is None:
            raise BundleError("declared learnables require complete prior and posterior learning state")
        learning, learning_arrays = _learning_snapshot(learnables)
    bundled = _rewrite_blob_paths(template)
    blob_paths = {name: _blob_member(name) for name in sorted(arrays)}
    raw_blobs = {name: _npy(value) for name, value in arrays.items()}
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "structure_hash": structure_hash(bundled),
        "blob_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in raw_blobs.items()},
        "blob_paths": blob_paths,
        "blob_arrays": {name: _array_descriptor(value) for name, value in arrays.items()},
        "learning": learning,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": "cffg",
    }
    members = {
        "model.json": (to_json(bundled) + "\n").encode(),
        "manifest.json": (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        **{blob_paths[name]: raw for name, raw in raw_blobs.items()},
        **{name: _npy(value) for name, value in learning_arrays.items()},
    }
    # Validate exactly the bytes to be published, before creating a destination
    # or sibling. This also reconstructs and checks compiler-owned shapes.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    buffer.seek(0)
    load_bundle(buffer)
    path = Path(path)
    sibling = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
            sibling = Path(output.name)
            output.write(buffer.getvalue())
            output.flush()
            os.fsync(output.fileno())
        os.replace(sibling, path)
    finally:
        if sibling is not None:
            sibling.unlink(missing_ok=True)


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BundleError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def _check_members(archive):
    listed = archive.infolist()
    names = {info.filename for info in listed}
    if len(names) != len(listed):
        raise BundleError("duplicate ZIP member names")
    for info in listed:
        name = info.filename
        if (
            info.orig_filename != name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or "\\" in name or ":" in name
            or any(ord(c) < 32 or ord(c) == 127 for c in name)
            or stat.S_ISLNK(info.external_attr >> 16)
        ):
            raise BundleError(f"unsafe ZIP member {name!r}")
    for required in ("model.json", "manifest.json"):
        if required not in names:
            raise BundleError(f"missing {required}")
    return names


def _read_array(archive, member):
    raw = archive.read(member)
    stream = io.BytesIO(raw)
    value = np.load(stream, allow_pickle=False)
    if not isinstance(value, np.ndarray) or stream.tell() != len(raw):
        raise BundleError(f"invalid NPY member {member!r}")
    return value


def _restore_learning(document, values, arena):
    from cffg.learn import Learnables

    if not isinstance(document, dict) or set(document) != {"schema_version", "learnables"}:
        raise BundleError("invalid learning inventory")
    records = document["learnables"]
    if not isinstance(records, list):
        raise BundleError("learning inventory must be a list")
    priors, posteriors = {}, {}
    for record in records:
        name = record["name"]
        if name in priors:
            raise BundleError(f"duplicate learning owner {name!r}")
        priors[name], posteriors[name] = {}, {}
        for row in record["arrays"]:
            if row["role"] not in ("prior", "posterior"):
                raise BundleError("invalid learning array role")
            state = priors[name] if row["role"] == "prior" else posteriors[name]
            if row["key"] in state:
                raise BundleError("duplicate learning hyperparameter")
            state[row["key"]] = values[row["name"]]
    learning = Learnables.from_persistent_state(SimpleNamespace(arena=arena), priors, posteriors)
    descriptors = learning.persistent_descriptors()
    validate_persistent_manifest(np.asarray(json.dumps(document)), descriptors)
    expected = {row.name: row for d in descriptors for row in d.arrays}
    for record in records:
        for row in record["arrays"]:
            actual = expected[row["name"]]
            if row["dtype"] != actual.dtype or row["sha256"] != actual.sha256:
                raise BundleError(f"learning array {row['name']!r}: dtype or value digest mismatch")
    return learning


def load_bundle(path: str | Path) -> LoadedBundle:
    """Validate and load a complete schema-2 archive.

    No caller-supplied priors are needed for schema 2. Compilation validates
    point shapes and reconstructs logical learning plans; no inference runs.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            return _load_archive(archive)
    except BundleError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, EOFError, zipfile.BadZipFile) as error:
        raise BundleError(f"invalid bundle: {error}") from error


def _load_archive(archive):
    from cffg.compiler import compile_template

    names = _check_members(archive)
    manifest = _json(archive.read("manifest.json"))
    version = manifest.get("schema_version")
    if version != BUNDLE_SCHEMA_VERSION:
        raise BundleError(f"unsupported bundle schema {version!r}; expected {BUNDLE_SCHEMA_VERSION!r}")
    document = _json(archive.read("model.json"))
    if manifest.get("structure_hash") != document_structure_hash(document):
        raise BundleError("structure hash mismatch")
    template = from_dict(document)
    try:
        expanded = expand_plates(template)
        validate(expanded)
    except (ValueError, KeyError) as error:
        raise BundleError(f"invalid bundled plate structure: {error}") from error
    paths = manifest.get("blob_paths")
    if paths is None:
        raise BundleError("schema-current bundle requires blob_paths")
    if not isinstance(paths, dict) or set(paths) != set(expanded.blobs):
        raise BundleError("blob inventory mismatch")
    if any(not isinstance(member, str) for member in paths.values()) or len(set(paths.values())) != len(paths):
        raise BundleError("invalid or colliding blob paths")
    if any(member != expanded.blobs[name].path for name, member in paths.items()):
        raise BundleError("blob paths disagree with model declarations")
    if paths != {name: _blob_member(name) for name in paths}:
        raise BundleError("noncanonical blob paths")
    hashes = manifest.get("blob_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(paths):
        raise BundleError("blob hash inventory mismatch")
    arrays = {}
    for name, member in paths.items():
        if member not in names:
            raise BundleError(f"missing blob file {member}")
        if hashlib.sha256(archive.read(member)).hexdigest() != hashes[name]:
            raise BundleError(f"blob {name!r} sha256 mismatch")
        arrays[name] = _read_array(archive, member)
    _point_arrays(expanded, arrays)
    if manifest.get("blob_arrays") != {name: _array_descriptor(value) for name, value in arrays.items()}:
        raise BundleError("point array shape, dtype or value digest mismatch")
    learning_members = set()
    inventory = manifest["learning"]
    for record in inventory["learnables"]:
        for row in record["arrays"]:
            member = row["name"]
            if member in learning_members or member in paths.values() or not member.startswith("learning/"):
                raise BundleError("duplicate or colliding learning array member")
            if member not in names:
                raise BundleError(f"missing learning array {member}")
            learning_members.add(member)
    expected_members = {"model.json", "manifest.json", *paths.values(), *learning_members}
    if names != expected_members:
        raise BundleError(f"unexpected ZIP members {sorted(names - expected_members)}")
    arena = compile_template(template, arrays=arrays)
    learning = _restore_learning(inventory, {
        member: _read_array(archive, member) for member in learning_members
    }, arena)
    return LoadedBundle(template, arrays, {**manifest, "resumable": True}, learning)
