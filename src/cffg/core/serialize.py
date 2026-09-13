"""Canonical serialization and the structure hash.

The dict form is canonical: keys sorted, constraints ordered deterministically,
enums as strings, no derived data. The structure hash is a sha256 over the
canonical form with blob *paths* erased (names kept): relocating a parameter
file changes nothing the compiler cares about, so JIT-cache keys and colony
batching groups survive blob moves. Any structural edit — a cardinality, a
wire, a bead — changes the hash.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from cffg.core.ir import (
    AgentSpec,
    BeliefBinding,
    BlobRef,
    BoundaryFactorization,
    Broadcast,
    Chain,
    ConstraintKind,
    ConstraintSpec,
    Expose,
    FactorSpec,
    LearnableSpec,
    PlateSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)

SCHEMA_VERSION = "1"


def _bond_to_dict(rule: Any) -> dict[str, Any]:
    if isinstance(rule, Chain):
        return {
            "kind": "chain",
            "port_a": rule.port_a,
            "port_b": rule.port_b,
            "axis": rule.axis,
            "offset": rule.offset,
        }
    if isinstance(rule, Broadcast):
        return {"kind": "broadcast", "port": rule.port, "hub": rule.hub}
    if isinstance(rule, Expose):
        return {"kind": "expose"}
    raise TypeError(f"unserializable wiring rule {rule!r}")


def _bond_from_dict(d: dict[str, Any]) -> Any:
    kind = d["kind"]
    if kind == "chain":
        return Chain(d["port_a"], d["port_b"], d.get("axis", 0), d.get("offset", 1))
    if kind == "broadcast":
        return Broadcast(d["port"], d["hub"])
    if kind == "expose":
        return Expose()
    raise ValueError(f"unknown wiring rule kind {kind!r}")


def _plate_to_dict(spec: PlateSpec) -> dict[str, Any]:
    return {
        "fragment": to_dict(spec.fragment),
        "dims": list(spec.dims),
        "bonds": [_bond_to_dict(b) for b in spec.bonds],
        "per_copy_blobs": list(spec.per_copy_blobs),
    }


def _plate_from_dict(d: dict[str, Any]) -> PlateSpec:
    return PlateSpec(
        fragment=from_dict(d["fragment"]),
        dims=tuple(d["dims"]),
        bonds=tuple(_bond_from_dict(b) for b in d.get("bonds", [])),
        per_copy_blobs=tuple(d.get("per_copy_blobs", [])),
    )


def to_dict(t: Template) -> dict[str, Any]:
    d: dict[str, Any] = {
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
    # Additive: templates without plates serialize byte-identically to before.
    if t.plates:
        d["plates"] = {name: _plate_to_dict(t.plates[name]) for name in sorted(t.plates)}
    # Additive (cff-x0sv): declared learnables. Structure, not blob data — a
    # model that learns its transitions is a different model from one that
    # does not, so this participates in the structure hash.
    if t.learnables:
        d["learning_schema_version"] = "1"
        d["learnables"] = {
            name: {
                "family": ln.family,
                "blobs": list(ln.blobs),
                "semantic_identity": ln.semantic_identity,
                **({"sources": list(ln.sources)} if ln.sources is not None else {}),
            }
            for name in sorted(t.learnables)
            for ln in [t.learnables[name]]
        }
    # Additive (cff-nhxf): the serialized model/runtime interface. These are
    # structural names and variational-family declarations, so changing any
    # binding changes the structure hash.
    if t.agent is not None:
        belief_names = [belief.name for belief in t.agent.beliefs]
        if len(set(belief_names)) != len(belief_names):
            raise ValueError(
                f"agent has duplicate belief names {sorted(belief_names)!r}"
            )
        agent: dict[str, Any] = {
            "action": t.agent.action,
            "beliefs": {
                belief.name: {
                    "blocks": list(belief.blocks),
                    "prior": belief.prior,
                    "posterior": belief.posterior,
                    "factorization": belief.factorization.value,
                    **(
                        {"constraints": list(belief.constraints)}
                        if belief.constraints
                        else {}
                    ),
                    **(
                        {"episode_learnable": belief.episode_learnable}
                        if belief.episode_learnable is not None
                        else {}
                    ),
                }
                for belief in sorted(t.agent.beliefs, key=lambda b: b.name)
            },
        }
        if t.agent.action_port is not None:
            agent["action_port"] = t.agent.action_port
        if t.agent.observation_port is not None:
            agent["observation_port"] = t.agent.observation_port
        d["agent"] = agent
    # Additive: record (impl, version) for factor types backed by *external*
    # FactorKernels (cff-ri8m), so loading without them fails loudly and
    # names what to install. Built-in (cffg.) kernels ship with the package
    # and are omitted — existing template JSON stays byte-identical.
    from cffg import kernels  # deferred: cffg.kernels imports cffg.core.ir

    required = kernels.manifest(f.type for f in t.factors.values())
    if required:
        d["kernels"] = required
    return d


def from_dict(d: dict[str, Any]) -> Template:
    from cffg.core.learning_identity import declaration_identity

    learning_version = d.get("learning_schema_version")
    if learning_version not in (None, "1"):
        raise ValueError(f"unknown learning schema version {learning_version!r}")
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema version {d.get('schema_version')!r} != supported {SCHEMA_VERSION!r}"
        )
    _check_kernel_requirements(d)
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
        plates={
            name: _plate_from_dict(pspec) for name, pspec in d.get("plates", {}).items()
        },
        learnables={
            name: LearnableSpec(
                name,
                spec["family"],
                tuple(spec["blobs"]),
                tuple(spec["sources"]) if "sources" in spec else None,
                declaration_identity(name, spec["family"], spec, learning_version),
            )
            for name, spec in d.get("learnables", {}).items()
        },
        agent=(
            AgentSpec(
                action=d["agent"]["action"],
                beliefs=tuple(
                    BeliefBinding(
                        name=name,
                        blocks=tuple(spec["blocks"]),
                        prior=spec["prior"],
                        posterior=spec["posterior"],
                        factorization=BoundaryFactorization(spec["factorization"]),
                        constraints=tuple(spec.get("constraints", ())),
                        episode_learnable=spec.get("episode_learnable"),
                    )
                    for name, spec in sorted(d["agent"]["beliefs"].items())
                ),
                action_port=d["agent"].get("action_port"),
                observation_port=d["agent"].get("observation_port"),
            )
            if "agent" in d
            else None
        ),
        schema_version=d["schema_version"],
    )


def _check_kernel_requirements(d: dict[str, Any]) -> None:
    """Loud-failure gate for kernel-backed factor types (cff-ri8m): a type
    that cannot execute must be refused at load, naming the missing
    implementation, instead of failing later and vaguer. Two distinct gaps
    are checked — an unknown FactorType, and (per the manifest) a known
    FactorType whose *executable kernel* is not registered: a structural
    registry entry is not evidence the implementation is installed. A
    registered kernel must match the recorded impl identity and MAJOR
    version — MINOR/PATCH drift is compatible by the policy in
    cffg.kernels."""
    from cffg.core.ir import FACTOR_TYPES  # local: avoid confusion with module import list
    from cffg import kernels

    required = d.get("kernels", {})
    used_types = {spec["type"] for spec in d.get("factors", {}).values()}
    for name, spec in d.get("factors", {}).items():
        ftype = spec["type"]
        if ftype in FACTOR_TYPES:
            continue
        need = required.get(ftype)
        hint = (
            f" (template requires implementation {need['impl']!r} "
            f"version {need['version']})"
            if need
            else ""
        )
        raise ValueError(
            f"factor {name!r}: unregistered factor type {ftype!r}{hint}; "
            "register its FactorKernel before loading"
        )
    for ftype, need in required.items():
        k = kernels.kernel_for(ftype)
        if k is None:
            if ftype in used_types:
                raise ValueError(
                    f"factor type {ftype!r} has a structural registration but no "
                    f"executable kernel; the template requires implementation "
                    f"{need.get('impl')!r} version {need.get('version')!r} — "
                    "register its FactorKernel before loading"
                )
            continue  # unused-by-any-factor requirement: nothing to run
        same_major = k.version.split(".")[0] == str(need.get("version", "")).split(".")[0]
        if k.impl_id != need.get("impl") or not same_major:
            raise ValueError(
                f"factor type {ftype!r}: registered kernel {k.impl_id!r} "
                f"v{k.version} is incompatible with the template requirement "
                f"{need.get('impl')!r} v{need.get('version')}"
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
    return document_structure_hash(to_dict(t))


def document_structure_hash(document: dict[str, Any]) -> str:
    """Hash the stored model before declaration migration, without changing it."""
    d = deepcopy(document)
    _erase_blob_data(d)
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _erase_blob_data(d: dict[str, Any]) -> None:
    """Blank blob paths/init here and inside every nested plate fragment, so a
    parameter-file move (at any depth) leaves the structure hash untouched."""
    d["blobs"] = {name: {} for name in d.get("blobs", {})}
    for pspec in d.get("plates", {}).values():
        _erase_blob_data(pspec["fragment"])


def legacy_learning_structure_hash(t: Template) -> str | None:
    """Pre-identity hash, only for families whose legacy semantics are known."""
    from cffg.core.learning_identity import LEGACY_IDENTITIES

    def erase_identity(document):
        for spec in document.get("learnables", {}).values():
            if spec["semantic_identity"] != LEGACY_IDENTITIES.get(spec["family"]):
                return False
            del spec["semantic_identity"]
        document.pop("learning_schema_version", None)
        return all(erase_identity(p["fragment"]) for p in document.get("plates", {}).values())

    document = to_dict(t)
    if not erase_identity(document):
        return None
    _erase_blob_data(document)
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sorted(mapping: dict[str, Any]) -> list[Any]:
    return [mapping[k] for k in sorted(mapping)]
