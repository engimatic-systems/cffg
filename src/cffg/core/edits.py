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


EditAction = (
    AddVariable
    | AddFactor
    | RemoveFactor
    | RemoveVariable
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
