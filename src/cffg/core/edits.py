"""Transactional edit actions: the only sanctioned way to mutate a template.

An edit bundle is all-or-nothing: `apply_edits` works on a deep copy, applies
every action, validates the result, and only then returns it — the input
template is never touched. Bundles are therefore exactly the "edit actions
which result in a valid model" required for the GUI story (brief items 4/6):
things that must change together, change together, inside one bundle.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from cffg.core.ir import (
    BlobRef,
    ConstraintSpec,
    FactorSpec,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.validate import validate


class EditError(ValueError):
    """An action could not be applied (before validation even runs)."""


@dataclass(frozen=True)
class AddVariable:
    spec: VariableSpec

    def apply(self, t: Template) -> None:
        if self.spec.name in t.variables:
            raise EditError(f"variable {self.spec.name!r} already exists")
        t.variables[self.spec.name] = self.spec


@dataclass(frozen=True)
class AddFactor:
    spec: FactorSpec

    def apply(self, t: Template) -> None:
        if self.spec.name in t.factors:
            raise EditError(f"factor {self.spec.name!r} already exists")
        t.factors[self.spec.name] = self.spec


@dataclass(frozen=True)
class RemoveFactor:
    name: str

    def apply(self, t: Template) -> None:
        if self.name not in t.factors:
            raise EditError(f"factor {self.name!r} does not exist")
        del t.factors[self.name]


@dataclass(frozen=True)
class RemoveVariable:
    name: str

    def apply(self, t: Template) -> None:
        if self.name not in t.variables:
            raise EditError(f"variable {self.name!r} does not exist")
        del t.variables[self.name]
        t.ports = {n: p for n, p in t.ports.items() if p.variable != self.name}


@dataclass(frozen=True)
class AddConstraint:
    spec: ConstraintSpec

    def apply(self, t: Template) -> None:
        if self.spec in t.constraints:
            raise EditError(f"constraint {self.spec} already present")
        t.constraints.append(self.spec)


@dataclass(frozen=True)
class RemoveConstraint:
    spec: ConstraintSpec

    def apply(self, t: Template) -> None:
        try:
            t.constraints.remove(self.spec)
        except ValueError:
            raise EditError(f"constraint {self.spec} not present") from None


@dataclass(frozen=True)
class DeclarePort:
    spec: PortSpec

    def apply(self, t: Template) -> None:
        if self.spec.name in t.ports:
            raise EditError(f"port {self.spec.name!r} already exists")
        t.ports[self.spec.name] = self.spec


@dataclass(frozen=True)
class RemovePort:
    name: str

    def apply(self, t: Template) -> None:
        if self.name not in t.ports:
            raise EditError(f"port {self.name!r} does not exist")
        del t.ports[self.name]


@dataclass(frozen=True)
class AddBlob:
    ref: BlobRef

    def apply(self, t: Template) -> None:
        if self.ref.name in t.blobs:
            raise EditError(f"blob {self.ref.name!r} already exists")
        t.blobs[self.ref.name] = self.ref


@dataclass(frozen=True)
class RebindBlobPath:
    """Point an existing blob at a new file. Structure hash is unaffected."""

    name: str
    path: str

    def apply(self, t: Template) -> None:
        if self.name not in t.blobs:
            raise EditError(f"blob {self.name!r} does not exist")
        t.blobs[self.name] = BlobRef(self.name, self.path)


@dataclass(frozen=True)
class ReplaceFactor:
    """Swap an existing factor's definition (rewiring in one action)."""

    spec: FactorSpec

    def apply(self, t: Template) -> None:
        if self.spec.name not in t.factors:
            raise EditError(f"factor {self.spec.name!r} does not exist (use AddFactor)")
        t.factors[self.spec.name] = self.spec


EditAction = (
    AddVariable
    | AddFactor
    | RemoveFactor
    | RemoveVariable
    | ReplaceFactor
    | AddConstraint
    | RemoveConstraint
    | DeclarePort
    | RemovePort
    | AddBlob
    | RebindBlobPath
)


def apply_edits(t: Template, actions: list[EditAction]) -> Template:
    """Apply a bundle atomically; returns a new validated Template.

    Raises EditError or ModelError; `t` is unchanged in every failure mode.
    """
    working = copy.deepcopy(t)
    for action in actions:
        action.apply(working)
    validate(working)
    return working


def check_edits(t: Template, actions: list[EditAction]) -> list[str]:
    """Dry-run for editors: the problems a bundle would cause, without
    raising (empty list = the bundle is applicable). Live-validation hook
    for the GUI story."""
    from cffg.core.validate import ModelError

    try:
        apply_edits(t, actions)
    except ModelError as err:
        return list(err.problems)
    except EditError as err:
        return [str(err)]
    return []


# --- JSON codec (the GUI wire format) ----------------------------------------

_ACTIONS: dict[str, type] = {
    "add_variable": AddVariable,
    "add_factor": AddFactor,
    "remove_factor": RemoveFactor,
    "remove_variable": RemoveVariable,
    "replace_factor": ReplaceFactor,
    "add_constraint": AddConstraint,
    "remove_constraint": RemoveConstraint,
    "declare_port": DeclarePort,
    "remove_port": RemovePort,
    "add_blob": AddBlob,
    "rebind_blob_path": RebindBlobPath,
}
_ACTION_NAMES = {cls: name for name, cls in _ACTIONS.items()}


def _spec_to_dict(spec) -> dict:
    from cffg.core.ir import ConstraintSpec, PortSpec, VariableSpec as VS

    if isinstance(spec, VS):
        if spec.dim:
            return {"name": spec.name, "dim": spec.dim}
        return {"name": spec.name, "cardinality": spec.cardinality}
    if isinstance(spec, FactorSpec):
        return {
            "name": spec.name,
            "type": spec.type,
            "edges": dict(spec.edges),
            "param_edges": dict(spec.param_edges),
            "param_blobs": dict(spec.param_blobs),
        }
    if isinstance(spec, ConstraintSpec):
        return {"kind": spec.kind.value, "target": spec.target, "detail": dict(spec.detail)}
    if isinstance(spec, PortSpec):
        return {"name": spec.name, "variable": spec.variable, "role": spec.role.value}
    if isinstance(spec, BlobRef):
        return {"name": spec.name, "path": spec.path}
    raise TypeError(f"unserializable spec {type(spec).__name__}")


def action_to_dict(action: EditAction) -> dict:
    name = _ACTION_NAMES[type(action)]
    if hasattr(action, "spec"):
        return {"action": name, "spec": _spec_to_dict(action.spec)}
    if hasattr(action, "ref"):
        return {"action": name, "ref": _spec_to_dict(action.ref)}
    if isinstance(action, RebindBlobPath):
        return {"action": name, "name": action.name, "path": action.path}
    return {"action": name, "name": action.name}


def action_from_dict(d: dict) -> EditAction:
    from cffg.core.ir import ConstraintKind, ConstraintSpec, PortRole, PortSpec, VariableSpec as VS

    cls = _ACTIONS.get(d.get("action", ""))
    if cls is None:
        raise EditError(f"unknown edit action {d.get('action')!r}")
    if cls is AddVariable:
        return cls(VS(d["spec"]["name"], d["spec"].get("cardinality", 0), d["spec"].get("dim", 0)))
    if cls in (AddFactor, ReplaceFactor):
        s = d["spec"]
        return cls(
            FactorSpec(
                s["name"], s["type"],
                edges=dict(s.get("edges", {})),
                param_edges=dict(s.get("param_edges", {})),
                param_blobs=dict(s.get("param_blobs", {})),
            )
        )
    if cls in (AddConstraint, RemoveConstraint):
        s = d["spec"]
        return cls(
            ConstraintSpec(
                ConstraintKind(s["kind"]), s["target"], tuple(sorted(s.get("detail", {}).items()))
            )
        )
    if cls is DeclarePort:
        s = d["spec"]
        return cls(PortSpec(s["name"], s["variable"], PortRole(s["role"])))
    if cls is AddBlob:
        return cls(BlobRef(d["ref"]["name"], d["ref"]["path"]))
    if cls is RebindBlobPath:
        return cls(d["name"], d["path"])
    return cls(d["name"])  # Remove{Factor,Variable,Port}


def bundle_to_json(actions: list[EditAction]) -> str:
    import json

    return json.dumps([action_to_dict(a) for a in actions], indent=2)


def bundle_from_json(text: str) -> list[EditAction]:
    import json

    return [action_from_dict(d) for d in json.loads(text)]
