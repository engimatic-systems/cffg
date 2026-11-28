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


@dataclass(frozen=True)
class HybridPlanSpec:
    """Read a categorical action on a window with one carried Gaussian belief."""

    coordinates: tuple[str, ...]
    prior: str
    commands: tuple[str, ...]
    action: str
    preview_ports: tuple[str, ...]


@dataclass(frozen=True)
class HybridFilterSpec:
    """Condition one switching Gaussian transition on the applied selector."""

    coordinates: tuple[str, ...]
    prior: str
    commands: tuple[str, ...]
    posterior: str
    transition: str
    selector_port: str
    observation_port: str


PLAN_SPECS = (GaussianPlanSpec, HybridPlanSpec)
FILTER_SPECS = (GaussianFilterSpec, HybridFilterSpec)
GAUSSIAN_WINDOW_SPECS = (*PLAN_SPECS, *FILTER_SPECS)
_WINDOW_TYPES = {
    "plan": GaussianPlanSpec,
    "filter": GaussianFilterSpec,
    "hybrid_plan": HybridPlanSpec,
    "hybrid_filter": HybridFilterSpec,
}


def to_document(spec):
    kind = next(name for name, cls in _WINDOW_TYPES.items() if isinstance(spec, cls))
    record = {"version": 2, "window": kind}
    for field in fields(spec):
        value = getattr(spec, field.name)
        record[field.name] = list(value) if isinstance(value, tuple) else value
    return record


def from_document(record):
    if type(record["version"]) is not int or record["version"] != 2:
        raise ValueError("unsupported agent version")
    cls = _WINDOW_TYPES.get(record.get("window"))
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
    if isinstance(spec, PLAN_SPECS):
        return replace(spec, prior=name(spec.prior), action=name(spec.action),
                       preview_ports=tuple(map(name, spec.preview_ports)))
    control = (
        {"control_matrix": name(spec.control_matrix)} if isinstance(spec, GaussianFilterSpec)
        else {"selector_port": name(spec.selector_port)}
    )
    return replace(spec, prior=name(spec.prior), posterior=name(spec.posterior),
                   transition=name(spec.transition), observation_port=name(spec.observation_port),
                   **control)


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
    bindings = [getattr(spec, field.name) for field in fields(spec)
                if field.name not in ("coordinates", "commands", "indices", "preview_ports")]
    if any(not isinstance(value, str) or not value for value in bindings):
        return ["Gaussian agent bindings must be nonempty model-local names"]
    dim = len(spec.coordinates)
    prior = t.factors.get(spec.prior)
    if (prior is None or prior.type != "gaussian"
            or set(prior.param_blobs) != {"mean", "cov"}
            or getattr(t.variables.get(prior.edges.get("out")), "dim", None) != dim):
        return ["Gaussian agent prior requires one vector Gaussian matching its coordinates"]

    dynamic = list(prior.param_blobs.values())
    planning = isinstance(spec, PLAN_SPECS)
    categorical_ports = set()
    if planning:
        edge = t.variables.get(spec.action)
        if isinstance(spec, GaussianPlanSpec):
            if (not getattr(edge, "dim", None) or not isinstance(spec.indices, tuple)
                    or len(spec.indices) != len(spec.commands)
                    or any(type(i) is not int or not 0 <= i < edge.dim for i in spec.indices)
                    or len(set(spec.indices)) != len(spec.indices)):
                errors.append("Gaussian agent action indices must select distinct Gaussian coordinates")
        else:
            selectors = [f.edges["selector"] for f in t.factors.values()
                         if f.type == "gaussian_mixture_linear" and "selector" in f.edges]
            if (getattr(edge, "dim", None) != 0 or edge.cardinality != len(spec.commands)
                    or not any(_same_equality_class(t, spec.action, slot) for slot in selectors)):
                errors.append("hybrid plan action must be a categorical mixture selector matching the commands")
        if not names(spec.preview_ports):
            return ["Gaussian agent requires distinct named preview ports"]
        ports = set(spec.preview_ports)
    else:
        edge = t.variables.get(spec.posterior)
        if getattr(edge, "dim", None) != dim:
            errors.append("Gaussian agent posterior must be a vector matching the full boundary")
        transition = t.factors.get(spec.transition)
        if isinstance(spec, GaussianFilterSpec):
            if (transition is None or transition.type != "gaussian_linear"
                    or "offset" not in transition.param_blobs or spec.control_matrix not in t.blobs):
                errors.append("Gaussian filter requires an affine transition offset and control matrix blob")
            else:
                dynamic.append(transition.param_blobs["offset"])
                if (transition.edges.get("out") != spec.posterior
                        or getattr(t.variables.get(transition.edges.get("in")), "dim", None) != dim):
                    errors.append("the controlled state transition must produce the declared posterior edge")
        else:
            categorical_ports.add(spec.selector_port)
            errors.extend(_selector_filter_errors(t, spec, prior, transition))
        ports = {spec.observation_port, *categorical_ports}

    data_ports = {name for name, port in t.ports.items() if port.role is PortRole.DATA}
    if ports != data_ports or any(
        not getattr(t.variables.get(t.ports[p].variable), "dim", None)
        for p in (ports & data_ports) - categorical_ports
    ):
        errors.append(
            "hybrid agent ports must cover precisely its declared DATA inputs"
            if isinstance(spec, (HybridPlanSpec, HybridFilterSpec))
            else "Gaussian agent ports must cover precisely the Gaussian DATA ports of its window"
        )
    readout = spec.action if planning else spec.posterior
    if any(t.ports[p].variable == readout for p in data_ports):
        errors.append("Gaussian agent readout cannot be a DATA edge with a cavity predictive")
    elif not planning and spec.observation_port in data_ports:
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


def _selector_filter_errors(t, spec, prior, transition):
    if transition is None or transition.type != "gaussian_mixture_linear":
        return ["hybrid filter requires a declared switching Gaussian transition"]
    selector = transition.edges.get("selector")
    source = transition.edges.get("in")
    observation = t.ports.get(spec.selector_port)
    choice = t.variables.get(selector)
    if (observation is None or observation.variable != selector or choice is None
            or choice.dim or choice.cardinality != len(spec.commands)
            or spec.selector_port == spec.observation_port):
        return ["hybrid filter must observe the executed selector as categorical DATA"]
    if (transition.edges.get("out") != spec.posterior or source is None
            or not _same_equality_class(t, prior.edges["out"], source)):
        return ["the commanded transition must connect the Gaussian prior to the posterior"]
    # An unobserved categorical state requires a joint regime/state carry.
    # This filter has exactly one switching node and one observed selector.
    if (any(var.dim == 0 for name, var in t.variables.items() if name != selector)
            or sum(f.type == "gaussian_mixture_linear" for f in t.factors.values()) != 1):
        return ["a hybrid filter cannot carry a hidden regime with a Gaussian marginal"]
    return []
