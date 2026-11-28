"""Transactional edit actions: the only sanctioned way to mutate a template.

An edit bundle is all-or-nothing: `apply_edits` works on a deep copy, applies
every action, validates the result, and only then returns it — the input
template is never touched. Bundles are therefore exactly the "edit actions
which result in a valid model" required for the GUI story (brief items 4/6):
things that must change together, change together, inside one bundle.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass

from cffg.core.compose import ComposeError
from cffg.core.equality_tree import plan_equality_tree
from cffg.core.ir import (
    BlobRef,
    ConstraintSpec,
    FactorSpec,
    PortRole,
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
class RemoveBlob:
    """Drop a parameter declaration. Final template validation checks surviving readers."""

    name: str

    def apply(self, t: Template) -> None:
        if self.name not in t.blobs:
            raise EditError(f"blob {self.name!r} does not exist")
        del t.blobs[self.name]


@dataclass(frozen=True)
class RebindBlobPath:
    """Point an existing blob at a new file. Structure hash is unaffected."""

    name: str
    path: str

    def apply(self, t: Template) -> None:
        if self.name not in t.blobs:
            raise EditError(f"blob {self.name!r} does not exist")
        # Keep the init spec: rebinding the file location must not erase the
        # warm start (the file still wins when it exists).
        t.blobs[self.name] = BlobRef(self.name, self.path, t.blobs[self.name].init)


@dataclass(frozen=True)
class ReplaceFactor:
    """Swap an existing factor's definition (rewiring in one action)."""

    spec: FactorSpec

    def apply(self, t: Template) -> None:
        if self.spec.name not in t.factors:
            raise EditError(f"factor {self.spec.name!r} does not exist (use AddFactor)")
        t.factors[self.spec.name] = self.spec


@dataclass(frozen=True)
class SetPlateDims:
    """Resize a plate (cff-cpxj). Resizing IS a structural change: the copy
    count is part of the structure hash, so the model recompiles."""

    name: str
    dims: tuple[int, ...]

    def apply(self, t: Template) -> None:
        if self.name not in t.plates:
            raise EditError(f"plate {self.name!r} does not exist")
        t.plates[self.name] = dataclasses.replace(t.plates[self.name], dims=tuple(self.dims))


def _attach_branches(t: Template, target: str, names: tuple[str, ...], role: PortRole, prefix: str) -> None:
    operation = "observe_interior" if role is PortRole.DATA else "fan_out"
    where = f"{operation}({target!r})"
    if not isinstance(target, str) or target not in t.variables:
        raise EditError(f"{where}: expected a known variable name")
    original = t.variables[target]
    if role is PortRole.DATA and original.ef:
        raise EditError(f"{where}: unsupported DATA domain {original.ef!r}")
    endpoints = []
    for factor in t.factors.values():
        if target in factor.param_edges.values():
            raise EditError(f"{where}: cannot branch a parameter edge")
        endpoints.extend((factor.name, key) for key, variable in factor.edges.items() if variable == target)
    ports = t.ports_of(target)
    if len(endpoints) + len(ports) != 2 or len(ports) > 1:
        raise EditError(f"{where}: expected two endpoints including any boundary port")
    if role is PortRole.DATA and ports:
        raise EditError(f"{where}: expected an interior variable, already exposed by {ports[0].name!r}")

    peer = f"{prefix}/peer"
    for name in (peer, *names):
        if name in t.variables:
            raise EditError(f"{where}: variable {name!r} already exists")
    for name in names:
        if name in t.ports:
            raise EditError(f"{where}: port {name!r} already exists")
    tree = plan_equality_tree(target, (peer, *names), prefix)
    try:
        tree.install(t.variables, t.factors, original)
    except ComposeError as exc:
        raise EditError(f"{where}: {exc}") from exc

    if ports:
        port = ports[0]
        t.ports[port.name] = dataclasses.replace(port, variable=peer)
    else:
        # Retain one endpoint on the original, including any constraint on it.
        factor_name, key = max(endpoints)
        factor = t.factors[factor_name]
        t.factors[factor_name] = dataclasses.replace(factor, edges={**factor.edges, key: peer})
    for name in (peer, *names):
        t.variables[name] = dataclasses.replace(original, name=name)
    for name in names:
        t.ports[name] = PortSpec(name, name, role)


@dataclass(frozen=True)
class FanOut:
    target: str
    branch_count: int

    @property
    def ports(self) -> tuple[str, ...]:
        if type(self.branch_count) is not int or self.branch_count < 1:
            raise EditError(f"fan_out({self.target!r}): branch_count must be a positive integer")
        return tuple(f"{self.target}/_fanout/branch{i}" for i in range(self.branch_count))

    def apply(self, t: Template) -> None:
        _attach_branches(t, self.target, self.ports, PortRole.GLUE, f"{self.target}/_fanout")


@dataclass(frozen=True)
class ObserveInterior:
    target: str

    @property
    def port(self) -> str:
        return f"{self.target}/_observe/data"

    def apply(self, t: Template) -> None:
        _attach_branches(t, self.target, (self.port,), PortRole.DATA, f"{self.target}/_observe")


def fan_out(t: Template, target: str, branch_count: int) -> tuple[Template, tuple[str, ...]]:
    """Branch an existing structural edge and return the copy and its ordered GLUE ports.

    The two existing endpoints (including a boundary port when present) keep
    their roles. Each additional port carries the same variable domain.
    """
    action = FanOut(target, branch_count)
    return apply_edits(t, [action]), action.ports


def observe_interior(t: Template, target: str) -> tuple[Template, str]:
    """Return a copy with one equality tap and a DATA port on the degree-two latent."""
    action = ObserveInterior(target)
    return apply_edits(t, [action]), action.port


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
    | RemoveBlob
    | RebindBlobPath
    | SetPlateDims
    | FanOut
    | ObserveInterior
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
    "remove_blob": RemoveBlob,
    "rebind_blob_path": RebindBlobPath,
    "set_plate_dims": SetPlateDims,
    "fan_out": FanOut,
    "observe_interior": ObserveInterior,
}
_ACTION_NAMES = {cls: name for name, cls in _ACTIONS.items()}


def _spec_to_dict(spec) -> dict:
    from cffg.core.ir import ConstraintSpec, PortSpec, VariableSpec as VS

    if isinstance(spec, VS):
        if spec.dim:
            return {"name": spec.name, "dim": spec.dim}
        if spec.ef:
            return {"name": spec.name, "ef": spec.ef}
        return {"name": spec.name, "cardinality": spec.cardinality}
    if isinstance(spec, FactorSpec):
        return {
            "name": spec.name,
            "type": spec.type,
            "edges": dict(spec.edges),
            "param_edges": dict(spec.param_edges),
            "param_blobs": dict(spec.param_blobs),
            **({"function": spec.function.to_dict()} if spec.function is not None else {}),
        }
    if isinstance(spec, ConstraintSpec):
        return {"kind": spec.kind.value, "target": spec.target, "detail": dict(spec.detail)}
    if isinstance(spec, PortSpec):
        return {"name": spec.name, "variable": spec.variable, "role": spec.role.value}
    if isinstance(spec, BlobRef):
        if spec.init:
            return {"name": spec.name, "path": spec.path, "init": dict(spec.init)}
        return {"name": spec.name, "path": spec.path}
    raise TypeError(f"unserializable spec {type(spec).__name__}")


def action_to_dict(action: EditAction) -> dict:
    name = _ACTION_NAMES[type(action)]
    if isinstance(action, (FanOut, ObserveInterior)):
        return {"action": name, **dataclasses.asdict(action)}
    if hasattr(action, "spec"):
        return {"action": name, "spec": _spec_to_dict(action.spec)}
    if hasattr(action, "ref"):
        return {"action": name, "ref": _spec_to_dict(action.ref)}
    if isinstance(action, RebindBlobPath):
        return {"action": name, "name": action.name, "path": action.path}
    if isinstance(action, SetPlateDims):
        return {"action": name, "name": action.name, "dims": list(action.dims)}
    return {"action": name, "name": action.name}


def action_from_dict(d: dict) -> EditAction:
    from cffg.core.ir import ConstraintKind, ConstraintSpec, PortRole, PortSpec, VariableSpec as VS

    cls = _ACTIONS.get(d.get("action", ""))
    if cls is None:
        raise EditError(f"unknown edit action {d.get('action')!r}")
    if cls in (FanOut, ObserveInterior):
        expected = {"action", "target", "branch_count"} if cls is FanOut else {"action", "target"}
        if set(d) != expected or not isinstance(d["target"], str):
            raise EditError(f"{d['action']!r} requires fields {sorted(expected)} and a string target")
        if cls is FanOut:
            if type(d["branch_count"]) is not int or d["branch_count"] < 1:
                raise EditError(f"fan_out({d['target']!r}): branch_count must be a positive integer")
            return FanOut(d["target"], d["branch_count"])
        return ObserveInterior(d["target"])
    if cls is AddVariable:
        s = d["spec"]
        return cls(VS(s["name"], s.get("cardinality", 0), s.get("dim", 0), s.get("ef", "")))
    if cls in (AddFactor, ReplaceFactor):
        from cffg.core.functions import FunctionSpec

        s = d["spec"]
        return cls(
            FactorSpec(
                s["name"], s["type"],
                edges=dict(s.get("edges", {})),
                param_edges=dict(s.get("param_edges", {})),
                param_blobs=dict(s.get("param_blobs", {})),
                function=FunctionSpec.from_dict(s["function"]) if "function" in s else None,
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
        return cls(
            BlobRef(
                d["ref"]["name"],
                d["ref"]["path"],
                tuple(sorted(d["ref"].get("init", {}).items())),
            )
        )
    if cls is RebindBlobPath:
        return cls(d["name"], d["path"])
    if cls is SetPlateDims:
        return cls(d["name"], tuple(d["dims"]))
    return cls(d["name"])  # Remove{Factor,Variable,Port}


def bundle_to_json(actions: list[EditAction]) -> str:
    import json

    return json.dumps([action_to_dict(a) for a in actions], indent=2)


def bundle_from_json(text: str) -> list[EditAction]:
    import json

    return [action_from_dict(d) for d in json.loads(text)]
