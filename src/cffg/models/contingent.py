"""Exact bounded precommitment with one action per available observation history.

At stage t the source table axes are (o0,...,o[t-1], a0,...,a[t]).
A preference is the probability of a preferred event at that entry. Each
nonterminal table has a matching causal observation kernel with a final o[t]
axis. For a fixed policy let r_h be the product of its observation probabilities
along history h. The expanded tree observes one event of likelihood pref_h**r_h
at each history. Thus its fixed-policy BFE, less the uniform action-prior cost,
is sum_h r_h * (-log pref_h). An unreachable branch has unit likelihood even
when its preference is zero. No future observation informs an earlier action.

Each named prefix state holds only the actions on the path to that node.
Normalized copy-and-append transitions share earlier actions among descendants.
A passive response leaf reports the selected action with probability r_h and
an extra absent category with probability 1-r_h. It integrates to one.
"""

from collections import defaultdict
from dataclasses import replace
from itertools import product
from math import prod

import numpy as np

from cffg.core import FactorSpec, PortRole, PortSpec, Template, VariableSpec, fan_out, observe_interior, validate
from cffg.core.ir import BlobRef


def _key(history):
    return "-".join(map(str, history)) or "root"


def _histories(cards):
    return product(*(range(size) for size in cards))


def _dimensions(observation_shapes, preference_shapes, open_loop, max_entries, max_histories):
    depth = len(observation_shapes)
    if depth not in (1, 2) or len(preference_shapes) != depth + 1:
        raise ValueError("Use one or two observation layers and one more preference table; "
                         "deeper policy trees grow combinatorially")
    if type(open_loop) is not bool or any(type(n) is not int or n < 1 for n in (max_entries, max_histories)):
        raise ValueError("open_loop must be boolean and resource limits must be positive integers")
    obs, acts = [], []
    entries = histories = 0
    for i, shape in enumerate(preference_shapes):
        if len(shape) != len(obs) + len(acts) + 1 or shape[:-1] != (*obs, *acts) or shape[-1] < 1:
            raise ValueError(f"preference[{i}] must have observation history then action prefix axes")
        a = shape[-1]
        p, h = prod(acts), prod(obs)
        acts.append(a)
        histories += h
        if histories > max_histories:
            raise ValueError(f"policy expansion requires at least {histories} histories; limit {max_histories}")
        # Count each factor's table even when multiple transitions share one blob on disk.
        transitions = 1 if open_loop else h
        entries += h * p * a * (a + 3) + (transitions * p * p * a if i else a)
        if entries > max_entries:
            raise ValueError(f"policy expansion requires at least {entries} CPT entries; limit {max_entries}")
        if i < depth:
            shape_o = observation_shapes[i]
            if len(shape_o) != len(shape) + 1 or shape_o[:-1] != shape or shape_o[-1] < 1:
                raise ValueError(f"observation[{i}] must append its categorical outcome axis")
            obs.append(shape_o[-1])
    return obs, acts


def _probabilities(value, dtype, kernel):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
        raise ValueError("Preferences and observation probabilities must be finite in [0, 1]")
    x = raw.astype(dtype, copy=False)
    if np.any((raw > 0) & (x < np.finfo(dtype).tiny)):
        raise ValueError("Positive probability is unresolved at the requested working precision")
    if np.any((raw < 1) & (x == 1)):
        raise ValueError("A probability below one rounds to certainty at working precision")
    if kernel:
        error = abs(x.sum(axis=-1, dtype=np.float64) - 1)
        if np.any(error > 8 * np.finfo(dtype).eps):
            raise ValueError("An observation row must sum to one")
    return x


def build_contingent(name, *, observations, preferences, open_loop=False, dtype=np.float32,
                     max_entries=1_000_000, max_histories=256):
    """Build a depth-one/two policy template and arrays in the chosen precision.

    With open_loop=True all histories at one stage share the same action.
    Admission bounds history count and the total CPT entries across factor instances.
    """
    obs, acts = _dimensions(list(map(np.shape, observations)), list(map(np.shape, preferences)),
                            open_loop, max_entries, max_histories)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Contingent policies support float32 and float64")
    preferences = [_probabilities(x, dtype, False) for x in preferences]
    observations = [_probabilities(x, dtype, True) for x in observations]
    t, arrays = Template(name=name), {}

    def blob(key, x):
        t.blobs[key] = BlobRef(key, key + ".npy")
        arrays[key] = x
        return key

    def matrix(label, src, dst, parameter):
        t.factors[label] = FactorSpec(label, "matrix_cat", {"in": src, "out": dst},
                                     param_blobs={"matrix": parameter})

    def terminal(label, src, values, role):
        t.variables[label] = VariableSpec(label, values.shape[0])
        matrix(label, src, label, blob(label, values))
        t.ports[label] = PortSpec(label, label, role)

    mass = {(): np.ones(1, dtype=dtype)}
    incoming = {}
    for level, a in enumerate(acts):
        size = prod(acts[:level + 1])
        histories = list(_histories(obs[:level]))
        groups = defaultdict(list)
        for h in histories:
            groups[f"t{level}" if open_loop else _key(h)].append(h)
        if level:
            parent_size = size // a
            append = np.zeros((size, parent_size), dtype=dtype)
            append[np.arange(size), np.arange(size) // a] = 1 / a
            blob(f"append/{level}", append)
        for group, members in groups.items():
            prefix = "prefix/" + group
            t.variables[prefix] = VariableSpec(prefix, size)
            if not level:
                t.factors["prior"] = FactorSpec("prior", "categorical", {"out": prefix},
                                                param_blobs={"statistic": blob("prior", np.full(a, 1/a, dtype=dtype))})
            else:
                source = t.ports.pop(incoming[group]).variable
                matrix("extend/" + group, source, prefix, f"append/{level}")

            def event(h, edge):
                reach = np.repeat(mass[h], a)
                pref = preferences[level][h].reshape(-1)
                chance = np.power(pref.astype(np.float64), reach).astype(dtype)
                if np.any((reach > 0) & (pref < 1) & (chance == 1)):
                    raise ValueError("Reach-weighted preference rounds to neutral at working precision")
                terminal("goal/" + _key(h), edge, np.stack((1 - chance, chance)), PortRole.DATA)

            event(members[0], prefix)
            t, clamp = observe_interior(t, prefix)
            label = "commit/" + group
            t.ports[label] = replace(t.ports.pop(clamp), name=label)
            children = []
            if level < len(obs):
                children = ([f"t{level+1}"] if open_loop else
                            [_key(members[0] + (o,)) for o in range(obs[level])])
            t, ports = fan_out(t, prefix, 2 * len(members) - 1 + len(children))
            free = iter(ports)
            for h in members[1:]:
                event(h, t.ports.pop(next(free)).variable)
            for h in members:
                reach = np.repeat(mass.pop(h), a)
                read = np.zeros((a + 1, size), dtype=dtype)
                ids = np.arange(size)
                read[ids % a, ids] = reach
                read[-1] = 1 - reach
                terminal("response/" + _key(h), t.ports.pop(next(free)).variable, read, PortRole.GLUE)
                if level < len(obs):
                    kernel = observations[level][h].reshape(size, obs[level])
                    for o in range(obs[level]):
                        r = reach * kernel[:, o]
                        if np.any((reach > 0) & (kernel[:, o] > 0) & (r < np.finfo(dtype).tiny)):
                            raise ValueError("Positive branch reach is unresolved at working precision")
                        mass[h + (o,)] = r
            incoming.update(zip(children, free, strict=True))
    validate(t)
    return t, arrays


def build_contingent_rollout(name, *, depth, d, A, Bs, c, open_loop=False,
                             dtype=np.float32, max_entries=1_000_000, max_histories=256):
    """Expand a categorical controlled hidden-state model for depth+1 actions.

    d is the initial state prior, Bs[action,next,state] the transition,
    A[reading,state] the sensor, and c[reading] a preference probability.
    Forward filtering uses only the interventions and the readings in each
    prefix. The local cost is the physical expectation of -log c[reading].
    Policy scoring and choice occur in the returned CFFG.
    """
    if type(depth) is not int or depth not in (1, 2):
        raise ValueError("Only branch depths one and two are supported; deeper trees grow combinatorially")
    states, obs, act = np.shape(d), np.shape(A), np.shape(Bs)
    if (len(states) != 1 or not states[0] or len(obs) != 2 or not obs[0] or obs[1:] != states
            or len(act) != 3 or act[1:] != states * 2 or not act[0] or np.shape(c) != obs[:1]):
        raise ValueError("Expected d[S], A[O,S], Bs[U,S,S], c[O] with positive cardinalities")
    o, u = obs[0], act[0]
    shapes = [(o,) * t + (u,) * (t + 1) for t in range(depth + 1)]
    _dimensions([s + (o,) for s in shapes[:-1]], shapes, open_loop, max_entries, max_histories)
    if prod(shapes[-1]) * (o + states[0]) > max_entries:
        raise ValueError("Causal state/observation expansion exceeds max_entries")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Contingent policies support float32 and float64")

    def distribution(x):
        # Normalize admitted producer roundoff in double before propagating mass.
        a = _probabilities(x, dtype, True).astype(np.float64)
        return a / a.sum(axis=-1, keepdims=True)

    initial = distribution(d)
    sensor = distribution(np.asarray(A).T).T
    transitions = distribution(np.swapaxes(Bs, 1, 2)).transpose(0, 2, 1)
    goals = distribution(c)
    transition_support = transitions > 0
    sensor_support = sensor > 0
    kernels, preferences = [], []
    current = {(): initial[None, :]}
    for t, shape in enumerate(shapes):
        pref = np.empty(shape, dtype=dtype)
        next_beliefs = {}
        if t < depth:
            kernel = np.empty(shape + (o,), dtype=dtype)
        for h, beliefs in current.items():
            predicted = np.einsum("usv,nv->nus", transitions, beliefs).reshape(-1, states[0])
            predicted_support = np.einsum("usv,nv->nus", transition_support, beliefs > 0).reshape(predicted.shape)
            if np.any(predicted_support & (predicted == 0)):
                raise ValueError("A possible hidden state is unresolved in the causal forward pass")
            observations = predicted @ sensor.T
            if np.any((predicted_support @ sensor_support.T) & (observations == 0)):
                raise ValueError("A possible sensor outcome is unresolved in the causal forward pass")
            # A reachable exact-zero preference makes this action impossible.
            good = goals > 0
            log_like = observations[:, good] @ np.log(goals[good])
            like = np.exp(log_like)
            like[np.any(observations[:, ~good] > 0, axis=1)] = 0
            if np.any(np.any(observations[:, goals < 1] > 0, axis=1) & (like.astype(dtype) == 1)):
                raise ValueError("Local preference rounds to neutral at working precision")
            if np.any((like > 0) & (like < np.finfo(dtype).tiny)):
                raise ValueError("Positive local preference is unresolved at working precision")
            pref[h] = like.reshape((u,) * (t + 1))
            if t < depth:
                kernel[h] = observations.reshape((u,) * (t + 1) + (o,))
                if np.any((observations > 0) & (observations < np.finfo(dtype).tiny)):
                    raise ValueError("Positive observation probability is unresolved at working precision")
                for reading in range(o):
                    weight = observations[:, reading, None]
                    # A zero-probability prefix never contributes to the score.
                    # Its conditional kernel still needs a proper placeholder.
                    posterior = np.divide(predicted * sensor[reading], weight,
                                          out=np.broadcast_to(initial, predicted.shape).copy(), where=weight > 0)
                    if np.any(predicted_support & sensor_support[reading] & (posterior == 0)):
                        raise ValueError("A possible posterior state is unresolved in the causal forward pass")
                    next_beliefs[h + (reading,)] = posterior
        preferences.append(pref)
        if t < depth:
            kernels.append(kernel)
        current = next_beliefs
    return build_contingent(name, observations=kernels, preferences=preferences,
                            dtype=dtype, open_loop=open_loop, max_entries=max_entries, max_histories=max_histories)
