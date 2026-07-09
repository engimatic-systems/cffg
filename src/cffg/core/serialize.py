"""Canonical serialization and the structure hash.

The dict form is canonical: keys sorted, constraints ordered deterministically,
enums as strings, no derived data. The structure hash is a sha256 over the
canonical form with blob *paths* erased (names kept): relocating a parameter
file changes nothing the compiler cares about, so JIT-cache keys and colony
batching groups survive blob moves. Any structural edit — a cardinality, a
wire, a bead — changes the hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cffg.core.ir import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)

SCHEMA_VERSION = "1"


def to_dict(t: Template) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "name": t.name,
        "variables": {
            v.name: (
                {"dim": v.dim}
                if v.dim
                else {"ef": v.ef} if v.ef else {"cardinality": v.cardinality}
            )
            for v in _sorted(t.variables)
        },
        "factors": {
            f.name: {
                "type": f.type,
                "edges": dict(sorted(f.edges.items())),
                "param_edges": dict(sorted(f.param_edges.items())),
                "param_blobs": dict(sorted(f.param_blobs.items())),
            }
            for f in _sorted(t.factors)
        },
        "constraints": sorted(
            (
                {"kind": c.kind.value, "target": c.target, "detail": dict(c.detail)}
                for c in t.constraints
            ),
            key=lambda d: (d["kind"], d["target"], sorted(d["detail"].items())),
        ),
        "ports": {
            p.name: {"variable": p.variable, "role": p.role.value} for p in _sorted(t.ports)
        },
        "blobs": {
            b.name: ({"path": b.path, "init": dict(b.init)} if b.init else {"path": b.path})
            for b in _sorted(t.blobs)
        },
    }


def from_dict(d: dict[str, Any]) -> Template:
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema version {d.get('schema_version')!r} != supported {SCHEMA_VERSION!r}"
        )
    return Template(
        name=d["name"],
        variables={
            name: VariableSpec(
                name, spec.get("cardinality", 0), spec.get("dim", 0), spec.get("ef", "")
            )
            for name, spec in d["variables"].items()
        },
        factors={
            name: FactorSpec(
                name,
                spec["type"],
                edges=dict(spec["edges"]),
                param_edges=dict(spec.get("param_edges", {})),
                param_blobs=dict(spec.get("param_blobs", {})),
            )
            for name, spec in d["factors"].items()
        },
        constraints=[
            ConstraintSpec(
                ConstraintKind(c["kind"]),
                c["target"],
                tuple(sorted(c.get("detail", {}).items())),
            )
            for c in d["constraints"]
        ],
        ports={
            name: PortSpec(name, spec["variable"], PortRole(spec["role"]))
            for name, spec in d["ports"].items()
        },
        blobs={
            name: BlobRef(
                name, spec["path"], tuple(sorted(spec.get("init", {}).items()))
            )
            for name, spec in d["blobs"].items()
        },
        schema_version=d["schema_version"],
    )


def to_json(t: Template) -> str:
    return json.dumps(to_dict(t), sort_keys=True, indent=2)


def from_json(text: str) -> Template:
    return from_dict(json.loads(text))


def save(t: Template, path: str | Path) -> None:
    Path(path).write_text(to_json(t) + "\n")


def load(path: str | Path) -> Template:
    return from_json(Path(path).read_text())


def structure_hash(t: Template) -> str:
    """Hex sha256 of the structure: blob paths AND init specs erased, names
    kept — both are parameter data, not structure (colony batching groups
    instances that differ only in parameters)."""
    d = to_dict(t)
    d["blobs"] = {name: {} for name in d["blobs"]}
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sorted(mapping: dict[str, Any]) -> list[Any]:
    return [mapping[k] for k in sorted(mapping)]
