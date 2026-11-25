"""Validate the policy semantics assumed when decoding a restored tree."""

from collections import defaultdict
from math import prod

import jax
import numpy as np

from cffg.core import PortRole
from cffg.models.contingent import _dimensions, _histories, _key, _probabilities


def inspect_policy(t, arrays, max_entries, max_histories):
    """Check the causal prefix topology and probability tables before compilation."""
    tied = "commit/t0" in t.ports
    histories = {}
    for label in t.ports:
        if label.startswith("response/"):
            key = label.removeprefix("response/")
            h = () if key == "root" else tuple(map(int, key.split("-")))
            if _key(h) != key or len(h) > 2 or any(i < 0 for i in h):
                raise ValueError(f"Invalid policy history {key!r}")
            histories[h] = (label, f"t{len(h)}" if tied else key)
    if () not in histories or len(histories) > max_histories:
        raise ValueError("Missing root or policy exceeds the history limit")
    depth = max(map(len, histories))
    obs = [1 + max(h[j] for h in histories if len(h) > j) for j in range(depth)]
    acts = []
    for j in range(depth + 1):
        layer = [h for h in histories if len(h) == j]
        if not layer or len(layer) != prod(obs[:j]):
            raise ValueError("Policy histories must cover every observation at each stage")
        sizes = {t.variables[histories[h][0]].cardinality - 1 for h in layer}
        if len(sizes) != 1:
            raise ValueError("Action cardinalities must agree within a stage")
        acts.append(sizes.pop())
    shapes = [(*obs[:j], *acts[:j+1]) for j in range(depth+1)]
    _dimensions([s + (o,) for s, o in zip(shapes, obs)], shapes, tied, max_entries, max_histories)

    # Contract equality wires so each prior, transition, terminal and DATA tap
    # can be checked against its logical prefix regardless of fan-out shape.
    adjacent = defaultdict(set)
    ordinary = set()
    for name, f in t.factors.items():
        if f.type == "equality":
            wires = tuple(f.edges.values())
            for v in wires:
                adjacent[v].update(wires)
        else:
            ordinary.add(name)
    owner = {}
    groups = {g: acts[len(h)] for h, (_, g) in histories.items()}
    for group in groups:
        stack = ["prefix/" + group]
        while stack:
            v = stack.pop()
            if v in owner:
                if owner[v] != group:
                    raise ValueError("Distinct policy decisions share an equality wire")
                continue
            owner[v] = group
            stack.extend(adjacent[v])

    def table(name, kind, roles, parameter, shape):
        f = t.factors[name]
        if (f.type != kind or set(f.edges) != set(roles)
                or any(owner.get(f.edges[k], f.edges[k]) != v for k, v in roles.items())):
            raise ValueError(f"Incorrect policy factor binding at {name}")
        ordinary.remove(name)
        x = np.asarray(arrays[f.param_blobs[parameter]])
        if x.shape != shape or x.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"Incorrect policy table shape or dtype at {name}")
        effective = jax.dtypes.canonicalize_dtype(x.dtype)
        _probabilities(x.T, effective, True)
        return x

    root = histories[()][1]
    prior = table("prior", "categorical", {"out": root}, "statistic", (acts[0],))
    if not np.all(prior == prior.dtype.type(1 / acts[0])):
        raise ValueError("Policy prior must be uniform")
    reaches, goals = {}, {}
    extends = set()
    ports = set()
    for j, a in enumerate(acts):
        size = prod(acts[:j+1])
        idx = np.arange(size)
        for h in _histories(obs[:j]):
            label, group = histories[h]
            tap = "commit/" + group
            if t.ports[tap].role != PortRole.DATA or owner.get(t.ports[tap].variable) != group:
                raise ValueError(f"Incorrect policy clamp at {tap}")
            ports.add(tap)
            if h and group not in extends:
                parent = histories[h[:-1]][1]
                x = table("extend/" + group, "matrix_cat", {"in": parent, "out": group},
                          "matrix", (size, size // a))
                if np.count_nonzero(x) != size or np.any(x[idx, idx // a] != x.dtype.type(1/a)):
                    raise ValueError("Policy extension must copy its prefix and append a uniform action")
                extends.add(group)
            goal = "goal/" + _key(h)
            for name, role in ((label, PortRole.GLUE), (goal, PortRole.DATA)):
                if t.ports[name].role != role or t.ports[name].variable != name:
                    raise ValueError(f"Incorrect policy terminal at {name}")
                ports.add(name)
            read = table(label, "matrix_cat", {"in": group, "out": label}, "matrix", (a+1, size))
            reach = read[idx % a, idx]
            if (np.count_nonzero(read[:-1]) != np.count_nonzero(reach)
                    or not np.array_equal(read[-1], 1 - reach)
                    or np.any(reach.reshape(-1, a) != reach[::a, None])):
                raise ValueError("A response must report physical reach independent of the new action")
            goals[h] = table(goal, "matrix_cat", {"in": group, "out": goal}, "matrix", (2, size))[1].copy()
            if np.any(goals[h][reach == 0] != 1):
                raise ValueError("An unreachable history must have neutral preference")
            reaches[h] = reach
    if ordinary or ports != set(t.ports) or t.constraints or t.plates or t.learnables or t.agent:
        raise ValueError("Unexpected factor, constraint or declaration in policy tree")
    if not np.all(reaches[()] == 1):
        raise ValueError("The root must have unit reach")
    for h, r in reaches.items():
        if len(h) < depth:
            children = [reaches[h + (o,)][::acts[len(h)+1]] for o in range(obs[len(h)])]
            # Relative error protects nonzero rare branches as well as unit mass.
            if not np.allclose(np.sum(children, axis=0, dtype=np.float64), r,
                               rtol=16*np.finfo(r.dtype).eps, atol=0):
                raise ValueError("Child history masses must sum to their parent's reach")
    return histories, groups, reaches, goals
