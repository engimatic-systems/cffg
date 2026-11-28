"""Serialized interfaces for a joint Gaussian rolling boundary."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields, replace
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cffg.core.ir import Template


@dataclass(frozen=True)
class GaussianPlanSpec:
    """Ordered state and command coordinates include their model-local units."""

    coordinates: tuple[str, ...]
    prior: str
    commands: tuple[str, ...]
    action: str
    indices: tuple[int, ...]
    preview_ports: tuple[str, ...]


@dataclass(frozen=True)
class GaussianFilterSpec:
    """The known control contributes B @ applied to the transition offset."""

    coordinates: tuple[str, ...]
    prior: str
    commands: tuple[str, ...]
    posterior: str
    transition: str
    control_matrix: str
    observation_port: str


def to_document(spec):
    kind = "plan" if isinstance(spec, GaussianPlanSpec) else "filter"
    record = {"version": 2, "window": kind}
    for field in fields(spec):
        value = getattr(spec, field.name)
        record[field.name] = list(value) if isinstance(value, tuple) else value
    return record


def from_document(record):
    if type(record["version"]) is not int or record["version"] != 2:
        raise ValueError("unsupported agent version")
    cls = {"plan": GaussianPlanSpec, "filter": GaussianFilterSpec}.get(record.get("window"))
    if cls is None or set(record) != {"version", "window", *(f.name for f in fields(cls))}:
        raise ValueError("invalid Gaussian agent declaration")
    values = {}
    for field in fields(cls):
        value = record[field.name]
        if field.name in {"coordinates", "commands", "indices", "preview_ports"}:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"Gaussian agent {field.name} must be a sequence")
            value = tuple(value)
        elif not isinstance(value, str):
            raise ValueError(f"Gaussian agent {field.name} must be a name")
        values[field.name] = value
    return cls(**values)


def scoped(spec, name):
    if isinstance(spec, GaussianPlanSpec):
        return replace(spec, prior=name(spec.prior), action=name(spec.action),
                       preview_ports=tuple(map(name, spec.preview_ports)))
    return replace(spec, prior=name(spec.prior), posterior=name(spec.posterior),
                   transition=name(spec.transition), control_matrix=name(spec.control_matrix),
                   observation_port=name(spec.observation_port))


def _same_equality_class(t: Template, left: str, right: str) -> bool:
    """A bound Gaussian tap fixes every variable connected through equalities."""
    neighbors = defaultdict(list)
    for factor in t.factors.values():
        if factor.type == "equality":
            for a, b in pairwise(factor.edges.values()):
                neighbors[a].append(b)
                neighbors[b].append(a)

    pending = [left]
    seen = {left}
    while pending:
        edge = pending.pop()
        if edge == right:
            return True
        for neighbor in neighbors.get(edge, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return False


def validate_window(t: Template, spec) -> list[str]:
    from cffg.core.ir import PortRole

    errors = []

    def names(items):
        return (isinstance(items, tuple) and bool(items)
                and all(isinstance(item, str) and item for item in items)
                and len(set(items)) == len(items))

    if not names(spec.coordinates) or not names(spec.commands):
        return ["Gaussian agent coordinates and commands require unique, ordered names"]
    dim = len(spec.coordinates)
    prior = t.factors.get(spec.prior)
    if (prior is None or prior.type != "gaussian"
            or set(prior.param_blobs) != {"mean", "cov"}
            or getattr(t.variables.get(prior.edges.get("out")), "dim", None) != dim):
        return ["Gaussian agent prior requires one vector Gaussian matching its coordinates"]

    dynamic = list(prior.param_blobs.values())
    if isinstance(spec, GaussianPlanSpec):
        edge = t.variables.get(spec.action)
        if (not getattr(edge, "dim", None) or not isinstance(spec.indices, tuple)
                or len(spec.indices) != len(spec.commands)
                or any(type(i) is not int or not 0 <= i < edge.dim for i in spec.indices)
                or len(set(spec.indices)) != len(spec.indices)):
            errors.append("Gaussian agent action indices must select distinct Gaussian coordinates")
        if not names(spec.preview_ports):
            return ["Gaussian agent requires distinct named preview ports"]
        ports = set(spec.preview_ports)
    else:
        edge = t.variables.get(spec.posterior)
        if getattr(edge, "dim", None) != dim:
            errors.append("Gaussian agent posterior must be a vector matching the full boundary")
        transition = t.factors.get(spec.transition)
        if (transition is None or transition.type != "gaussian_linear"
                or "offset" not in transition.param_blobs or spec.control_matrix not in t.blobs):
            errors.append("Gaussian filter requires an affine transition offset and control matrix blob")
        else:
            dynamic.append(transition.param_blobs["offset"])
            if (transition.edges.get("out") != spec.posterior
                    or getattr(t.variables.get(transition.edges.get("in")), "dim", None) != dim):
                errors.append("the controlled state transition must produce the declared posterior edge")
        ports = {spec.observation_port}

    data_ports = {name for name, port in t.ports.items() if port.role is PortRole.DATA}
    if ports != data_ports or any(
        not getattr(t.variables.get(t.ports[p].variable), "dim", None) for p in ports & data_ports
    ):
        errors.append("Gaussian agent ports must cover precisely the Gaussian DATA ports of its window")
    readout = spec.action if isinstance(spec, GaussianPlanSpec) else spec.posterior
    if any(t.ports[p].variable == readout for p in data_ports):
        errors.append("Gaussian agent readout cannot be a DATA edge with a cavity predictive")
    elif isinstance(spec, GaussianFilterSpec) and spec.observation_port in data_ports:
        observed = t.ports[spec.observation_port].variable
        if _same_equality_class(t, observed, spec.posterior):
            errors.append(f"Gaussian filter observation {spec.observation_port!r} clamps posterior "
                          f"{spec.posterior!r} through equality and gives it zero carry covariance")
    readers = [blob for factor in t.factors.values() for blob in factor.param_blobs.values()]
    if len(set(dynamic)) != len(dynamic) or any(readers.count(blob) != 1 for blob in dynamic):
        errors.append("Gaussian agent effective offset and boundary prior need dedicated blobs")
    if isinstance(spec, GaussianFilterSpec) and spec.control_matrix in dynamic:
        errors.append("control matrix must be independent of dynamic boundary and offset blobs")
    if any(set(dynamic).intersection(spec.blobs) for spec in t.learnables.values()):
        errors.append("Gaussian agent runtime inputs overlap a learnable")
    return errors
