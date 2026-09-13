"""Family-independent, resolved ownership and array schemas for learning state.

Descriptors contain logical names, never compiled arena addresses. Recompiling
the same model therefore reconstructs their ownership. Array bytes, shapes and
dtypes belong to this inventory; only semantic identity belongs to the IR.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from urllib.parse import quote
from types import SimpleNamespace

import numpy as np

from cffg.core.learning_identity import check_identity, check_scoped_name


PERSISTENT_MANIFEST_KEY = "__cffg_learning_manifest__"
PERSISTENT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class PersistentParameter:
    """A family-owned hyperparameter key, shape and admissible dtype kind."""

    key: str
    shape: tuple[int, ...]
    dtype_kind: str = "f"


@dataclass(frozen=True)
class PersistentArray:
    """One named prior or posterior array, bound to its exact representation."""

    name: str
    key: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str


@dataclass(frozen=True, order=True)
class PersistentTarget:
    blob: str
    role: str
    form: str
    layout: str
    transform: str
    factor: str


@dataclass(frozen=True)
class PersistentStateDescriptor:
    name: str
    family: str
    factor_type: str
    semantic_identity: str
    sources: tuple[str, ...]
    parameter_blobs: tuple[tuple[str, tuple[int, ...]], ...]
    targets: tuple[PersistentTarget, ...]
    arrays: tuple[PersistentArray, ...]

    def to_dict(self) -> dict:
        """Return JSON-native metadata, without references to mutable state."""
        return json.loads(json.dumps(asdict(self)))


def _array_name(name: str, role: str, key: str) -> str:
    return f"learning/{quote(name, safe='')}/{role}/{quote(key, safe='')}.npy"


def _digest(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()


def persistent_schema(plan, family) -> tuple[PersistentParameter, ...]:
    """Resolve and check the family's complete array and parameter-role contract."""
    label = f"learnable {plan.name!r}"
    try:
        check_scoped_name(plan.name)
        check_identity(plan.name, plan.semantic_identity)
        if plan.semantic_identity != family.semantic_identity:
            raise ValueError(
                f"unknown or incompatible semantic identity {plan.semantic_identity!r}; "
                f"supported identity is {family.semantic_identity!r}"
            )
        roles = {}
        role_shapes = {}
        for target in plan.targets:
            check_scoped_name(target.blob)
            check_scoped_name(target.factor)
            if target.form not in family.parameter_forms:
                raise ValueError(f"incompatible parameter form {target.form!r}")
            if target.blob not in plan.shapes:
                raise ValueError(f"missing target shape for {target.blob!r}")
            if target.blob in roles and roles[target.blob] != target.role:
                raise ValueError(f"blob {target.blob!r} has incompatible target roles")
            roles[target.blob] = target.role
            shape = plan.shapes[target.blob]
            if target.role in role_shapes and role_shapes[target.role] != shape:
                raise ValueError(f"targets for role {target.role!r} disagree on shape")
            role_shapes[target.role] = shape
        if not roles or set(roles) != set(plan.shapes):
            raise ValueError("incomplete parameter target schema")
        for source in plan.sources:
            check_scoped_name(source)
        if len(set(plan.sources)) != len(plan.sources):
            raise ValueError("duplicate statistic sources")
        family.check(plan.name, plan.ftype, dict(plan.shapes), roles)
        schema = tuple(family.persistent_parameters(role_shapes))
        if not schema or len({item.key for item in schema}) != len(schema):
            raise ValueError("missing or duplicate persistent keys")
        for item in schema:
            check_scoped_name(item.key)
            if item.dtype_kind != "f":
                raise ValueError(f"persistent key {item.key!r} requires a real floating dtype")
            if any(type(dim) is not int or dim <= 0 for dim in item.shape):
                raise ValueError(f"invalid persistent shape {item.shape!r}")
        return schema
    except (ValueError, KeyError) as error:
        raise ValueError(f"{label}: {error}") from error


def resolve_persistent_state(plan, family, prior, posterior) -> PersistentStateDescriptor:
    """Validate both complete states without casting; bind their immutable metadata."""
    schema = persistent_schema(plan, family)
    arrays = []
    for role, state in (("prior", prior), ("posterior", posterior)):
        expected = {item.key for item in schema}
        if set(state) != expected:
            raise ValueError(
                f"learnable {plan.name!r}: incomplete {role} persistent keys; "
                f"expected {sorted(expected)}, got {sorted(state)}"
            )
        for item in sorted(schema, key=lambda item: item.key):
            value = np.asarray(state[item.key])
            if value.shape != item.shape or value.dtype.kind != item.dtype_kind:
                raise ValueError(
                    f"learnable {plan.name!r}: {role} key {item.key!r} has "
                    f"shape/dtype {value.shape}/{value.dtype.str}; expected "
                    f"{item.shape}/real floating dtype"
                )
            arrays.append(PersistentArray(
                _array_name(plan.name, role, item.key), item.key, role,
                value.shape, value.dtype.str, _digest(value),
            ))
        try:
            family.validate_prior(plan.name, state, plan.shapes)
        except (ValueError, TypeError) as error:
            raise ValueError(f"learnable {plan.name!r}: invalid {role}: {error}") from error
    return PersistentStateDescriptor(
        plan.name, plan.family, plan.ftype, plan.semantic_identity,
        tuple(plan.sources), tuple(sorted(plan.shapes.items())),
        tuple(sorted({PersistentTarget(
            t.blob, t.role, t.form, t.layout, t.transform, t.factor
        ) for t in plan.targets})), tuple(arrays),
    )


def encode_persistent_manifest(descriptors) -> np.ndarray:
    """Encode validated descriptors for flat posterior checkpoints.

    Priors are described but not restored by this checkpoint API. Complete
    prior-plus-posterior archive storage is a separate bundle-layer operation.
    """
    descriptors = tuple(descriptors)
    if len({d.name for d in descriptors}) != len(descriptors):
        raise ValueError(f"duplicate persistent learnables {[d.name for d in descriptors]}")
    from cffg.learn import FAMILIES

    for descriptor in descriptors:
        family = FAMILIES.get(descriptor.family)
        if family is None:
            raise ValueError(f"learnable {descriptor.name!r}: unknown family {descriptor.family!r}")
        if len(dict(descriptor.parameter_blobs)) != len(descriptor.parameter_blobs):
            raise ValueError(f"learnable {descriptor.name!r}: duplicate parameter blobs")
        schema = persistent_schema(SimpleNamespace(
            name=descriptor.name, semantic_identity=descriptor.semantic_identity,
            shapes=dict(descriptor.parameter_blobs), ftype=descriptor.factor_type,
            targets=descriptor.targets, sources=descriptor.sources,
        ), family)
        required = {(role, item.key): item.shape for role in ("prior", "posterior") for item in schema}
        present = {(row.role, row.key): row.shape for row in descriptor.arrays}
        if present != required:
            raise ValueError(f"learnable {descriptor.name!r}: incomplete or invalid persistent array schema")
        for row in descriptor.arrays:
            if row.name != _array_name(descriptor.name, row.role, row.key):
                raise ValueError(f"learnable {descriptor.name!r}: unsafe persistent array name")
    encoded = np.asarray(json.dumps({
        "schema_version": PERSISTENT_SCHEMA_VERSION,
        "learnables": [d.to_dict() for d in descriptors],
    }, sort_keys=True, separators=(",", ":")))
    validate_persistent_manifest(encoded, descriptors)
    return encoded


def validate_persistent_manifest(value, descriptors, *, exact: bool = True) -> dict:
    """Check metadata against resolved owners before any posterior is changed.

    Destination priors and floating dtypes may differ. Each incoming array is
    validated against its recorded dtype/digest and the family's required shape.
    The descriptor's prior rows are inventory only in a posterior checkpoint.
    With ``exact=False``, complete descriptors may name a subset of owners.
    """
    encoded = np.asarray(value)
    if encoded.shape != () or encoded.dtype.kind != "U":
        raise ValueError("learning manifest must be a scalar Unicode string")
    try:
        document = json.loads(encoded.item())
    except (ValueError, TypeError) as error:
        raise ValueError("learning manifest must be valid JSON") from error
    if not isinstance(document, dict) or set(document) != {"schema_version", "learnables"}:
        raise ValueError("invalid learning manifest fields")
    if document["schema_version"] != PERSISTENT_SCHEMA_VERSION:
        raise ValueError(f"unknown learning manifest version {document['schema_version']!r}")
    expected = {d.name: d.to_dict() for d in descriptors}
    records = document["learnables"]
    if not isinstance(records, list):
        raise ValueError("learning manifest learnables must be a list")
    seen = set()
    payload = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError("invalid learning manifest learnable")
        name = record["name"]
        label = f"learnable {name!r}"
        check_identity(name, record.get("semantic_identity"))
        if name not in expected or name in seen:
            raise ValueError(f"{label}: unknown or duplicate persistent descriptor")
        seen.add(name)
        want = expected[name]
        ownership = {key: value for key, value in record.items() if key != "arrays"}
        expected_ownership = {key: value for key, value in want.items() if key != "arrays"}
        if set(record) != set(want) or json.dumps(ownership, sort_keys=True) != json.dumps(
            expected_ownership, sort_keys=True
        ):
            raise ValueError(f"{label}: semantic identity, ownership or shapes do not match this engine")
        rows = record["arrays"]
        required = {(row["role"], row["key"]): row for row in want["arrays"]}
        present = set()
        if not isinstance(rows, list):
            raise ValueError(f"{label}: persistent arrays must be a list")
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(want["arrays"][0]):
                raise ValueError(f"{label}: invalid persistent array fields")
            if not isinstance(row["key"], str) or not isinstance(row["role"], str):
                raise ValueError(f"{label}: invalid persistent key or role")
            pair = row["role"], row["key"]
            if pair not in required or pair in present:
                raise ValueError(f"{label}: unknown or duplicate persistent key {pair}")
            present.add(pair)
            target = required[pair]
            if not isinstance(row["shape"], list) or any(
                type(dim) is not int or dim <= 0 for dim in row["shape"]
            ):
                raise ValueError(f"{label}: invalid persistent array shape")
            if any(row[key] != target[key] for key in ("name", "shape")):
                raise ValueError(f"{label}: invalid persistent array name or shape")
            try:
                dtype = np.dtype(row["dtype"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"{label}: invalid dtype") from error
            if dtype.kind != "f" or dtype.str != row["dtype"]:
                raise ValueError(f"{label}: dtype must be canonical real floating")
            digest = row["sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"{label}: invalid array digest")
            if row["role"] == "posterior":
                payload[name, row["key"]] = (dtype, digest)
        if present != set(required):
            raise ValueError(f"{label}: incomplete multi-array persistent schema")
    if exact and seen != set(expected):
        raise ValueError(f"learning manifest missing learnables {sorted(set(expected) - seen)}")
    return payload


def restore_persistent_array(name, key, value, payload) -> np.ndarray:
    array = np.asarray(value)
    dtype, digest = payload[name, key]
    if array.dtype != dtype or _digest(array) != digest:
        raise ValueError(f"learnable {name!r}: key {key!r} dtype or value digest mismatch")
    return array.copy()


def validate_learning_declarations(template, arrays) -> None:
    """Check model-only bundle declarations before opening an archive for writing."""
    from cffg.learn import FAMILIES

    for name, spec in template.learnables.items():
        family = FAMILIES.get(spec.family)
        if family is None:
            raise ValueError(f"learnable {name!r}: unknown conjugate family {spec.family!r}")
        readers = [f for f in template.factors.values() if set(spec.blobs) & set(f.param_blobs.values())]
        ftypes = {f.type for f in readers}
        if len(ftypes) != 1:
            raise ValueError(f"learnable {name!r}: expected readers of one factor type, got {ftypes}")
        targets = tuple(
            PersistentTarget(blob, role, "expected", "as_is", "identity", f.name)
            for f in readers for role, blob in f.param_blobs.items() if blob in spec.blobs
        )
        shapes = {target.blob: tuple(np.shape(arrays[target.blob])) for target in targets}
        persistent_schema(SimpleNamespace(
            name=name, family=spec.family, ftype=next(iter(ftypes)), shapes=shapes,
            semantic_identity=spec.semantic_identity, targets=targets,
            sources=tuple(f.name for f in readers if spec.sources is None or f.name in spec.sources),
        ), family)
