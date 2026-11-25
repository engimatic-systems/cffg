"""Versioned boundary declaration for the exact two-clock categorical schedule."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import jax.numpy as jnp
import numpy as np

from cffg.compiler import compile_template
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.serialize import structure_hash


@dataclass(frozen=True)
class HierarchySpec:
    """The two directed messages share one Forney edge, named at both endpoints."""

    parent_template: str
    child_template: str
    parent_port: str
    child_port: str
    child_transition: str
    thickness: int
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported hierarchy declaration version")
        if type(self.thickness) is not int or self.thickness < 1:
            raise ValueError("hierarchy temporal thickness must be a positive integer")
        for name in ("parent_template", "child_template", "parent_port", "child_port", "child_transition"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"hierarchy {name} must be a nonempty name")

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text):
        fields = json.loads(text)
        if "version" not in fields:
            raise ValueError("unversioned hierarchy declarations are unsupported")
        return cls(**fields)

    def structure_hash(self, parent, child):
        fields = [self.to_json(), structure_hash(parent), structure_hash(child)]
        return sha256(json.dumps(fields).encode()).hexdigest()


def canonical_model(parent_prior, parent_transition, child_prior, child_transition, emission, thickness):
    transition = np.asarray(child_transition)
    if transition.ndim != 3 or min(transition.shape) == 0 or transition.shape[1] != transition.shape[2]:
        raise ValueError("child transition must have shape [regime, destination, source]")
    r, x, _ = transition.shape
    if np.ndim(emission) != 2:
        raise ValueError("child emission must have shape [observation, state]")
    p_arrays = dict(d=np.full(r, 1/r, dtype=np.result_type(transition.dtype, np.float32)) if parent_prior is None else parent_prior, T=parent_transition)
    c_arrays = dict(d=child_prior, B=transition, A=emission)
    def variables(**cards):
        return {k: VariableSpec(k, int(v)) for k, v in cards.items()}
    def blobs(arrays):
        return {k: BlobRef(k, k + ".npy") for k in arrays}
    parent = Template("slow", variables=variables(old=r, regime=r), factors={
        "initial": FactorSpec("initial", "categorical", {"out": "old"}, param_blobs={"statistic": "d"}),
        "advance": FactorSpec("advance", "matrix_cat", {"out": "regime", "in": "old"}, param_blobs={"matrix": "T"}),
    }, blobs=blobs(p_arrays), ports={"regime": PortSpec("regime", "regime", PortRole.GLUE)})
    child = Template("fast", variables=variables(old=x, new=x, regime=r, observed=np.shape(emission)[0]), factors={
        "initial": FactorSpec("initial", "categorical", {"out": "old"}, param_blobs={"statistic": "d"}),
        "move": FactorSpec("move", "transition_mixture", {"out": "new", "in": "old", "selector": "regime"},
                           param_blobs={"matrices": "B"}),
        "emit": FactorSpec("emit", "matrix_cat", {"out": "observed", "in": "new"}, param_blobs={"matrix": "A"}),
    }, blobs=blobs(c_arrays), ports={"regime": PortSpec("regime", "regime", PortRole.GLUE),
                                     "reading": PortSpec("reading", "observed", PortRole.DATA)})
    spec = HierarchySpec(parent.name, child.name, "regime", "regime", "move", thickness)
    return spec, parent, p_arrays, child, c_arrays


def _factors(level, template, array_values, kinds):
    if (template.constraints or template.learnables or template.plates or template.agent
            or sorted(f.type for f in template.factors.values()) != sorted(kinds)):
        raise ValueError(f"{level}: unsupported factors, constraints, learning, plates or agent declaration")
    if set(array_values) != set(template.blobs):
        raise ValueError(f"{level}: supply all and only the declared probability arrays")
    if any(f.param_edges for f in template.factors.values()):
        raise ValueError(f"{level}: hierarchy needs fixed categorical factor parameters")
    for name, value in array_values.items():
        a = np.asarray(value)
        if a.dtype.kind not in "fiu" or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f"{level} array {name!r} must contain finite nonnegative probabilities")
    return {f.type: f for f in template.factors.values()}


def _port(template, name, role):
    port = template.ports.get(name)
    if port is None or port.role is not role or port.variable not in template.variables:
        raise ValueError(f"{template.name}: missing {role.value} boundary port {name!r}")
    return port.variable


def compile_hierarchy(spec, parent, parent_arrays, child, child_arrays):
    if (parent.name, child.name) != (spec.parent_template, spec.child_template):
        raise ValueError("parent or child template name differs from the hierarchy declaration")
    p = _factors("parent", parent, parent_arrays, ("categorical", "matrix_cat"))
    c = _factors("child", child, child_arrays, ("categorical", "transition_mixture", "matrix_cat"))
    pp, cp = _port(parent, spec.parent_port, PortRole.GLUE), _port(child, spec.child_port, PortRole.GLUE)
    r = parent.variables[pp].cardinality
    if r <= 0 or r != child.variables[cp].cardinality:
        raise ValueError("parent-to-child and child-to-parent boundary cardinalities differ")
    pd, pt = p["categorical"], p["matrix_cat"]
    cd, ct, ce = c["categorical"], c["transition_mixture"], c["matrix_cat"]
    if pt.edges != {"in": pd.edges["out"], "out": pp} or parent.variables[pd.edges["out"]].cardinality != r:
        raise ValueError("parent boundary must expose the destination of the same-cardinality slow transition")
    if (ct.name != spec.child_transition or ct.edges["selector"] != cp or cd.edges["out"] != ct.edges["in"]
            or ce.edges["in"] != ct.edges["out"] or
            child.variables[ct.edges["in"]].cardinality != child.variables[ct.edges["out"]].cardinality):
        raise ValueError("child boundary must select the declared fast transition with a persistent child state")
    if not any(port.variable == ce.edges["out"] and port.role is PortRole.DATA for port in child.ports.values()):
        raise ValueError("child emission requires a terminating observation DATA port")

    # Compile the actual named factors. Reuse the registered packer's parameter
    # address, role order and compiler checks, restoring literal log zero where
    # the generic single-arena encoding uses its finite structural sentinel.
    def packed(level, template, arrays, factors):
        try:
            arena = compile_template(template, arrays=arrays)
        except ValueError as exc:
            raise ValueError(f"{level} hierarchy template: {exc}") from exc
        result = []
        for f in factors:
            addr, = arena.member_of[f.name].parameters
            raw = np.asarray(arrays[addr.blob])
            axis = addr.probability_axis
            dtype = np.dtype(jnp.result_type(raw, jnp.float32))
            eps = np.finfo(dtype).eps
            if not np.allclose(raw.sum(axis=axis, dtype=np.float64), 1, rtol=0, atol=8*eps):
                raise ValueError(f"{level} factor {f.name!r} has a non-stochastic probability column")
            if np.any((raw > 0) & (raw < np.finfo(dtype).tiny)):
                raise ValueError(f"{level} factor {f.name!r}: positive subnormal probability in runtime dtype {dtype}")
            logs = np.asarray(arena.log_params[addr.key][addr.index])
            if addr.permutation:
                logs = logs.transpose(np.argsort(addr.permutation))
            result.append(jnp.asarray(np.where(raw > 0, logs, -np.inf), dtype=dtype))
        return result
    return tuple(packed("parent", parent, parent_arrays, (pd, pt)) + packed("child", child, child_arrays, (cd, ct, ce)))
