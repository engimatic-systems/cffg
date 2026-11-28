"""Predictive information from the unique path between two categorical edges."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

import numpy as np

from cffg.compiled import NEG
from cffg.core.ir import PortRole
from cffg.kernel_runtime import discrete_potential, parameters
from cffg.readouts.discrete import _card
from cffg.readouts.views import SensingView
from cffg.runtime_types import State


def _paths_from(arena, query):
    # Tag vertices because a factor and an edge may have the same public name.
    neighbors = defaultdict(list)
    for edge, endpoints in arena.endpoints.items():
        for factor, role in endpoints:
            neighbors[False, edge].append(((True, factor), role))
            neighbors[True, factor].append(((False, edge), role))
    root = (False, query)
    parent = {root: None}
    pending = deque([root])
    degree_sum = 0
    while pending:
        vertex = pending.popleft()
        degree_sum += len(neighbors[vertex])
        for adjacent, role in neighbors[vertex]:
            if adjacent not in parent:
                parent[adjacent] = (vertex, role)
                pending.append(adjacent)
    if degree_sum != 2 * (len(parent) - 1):
        raise ValueError(f"information gain requires an acyclic component containing {query!r}")
    return parent


def _factor_path(parent, query, target):
    vertex = (False, target)
    if vertex not in parent:
        raise ValueError(f"query {query!r} and candidate edge {target!r} are disconnected")
    reverse = []
    while parent[vertex] is not None:
        factor, outgoing = parent[vertex]
        vertex, incoming = parent[factor]
        reverse.append((factor[1], incoming, outgoing))
    return reversed(reverse)


def _log_values(values):
    # Detect the sentinel before promotion: its float32 and float64 words differ.
    return np.where(values == np.asarray(NEG, dtype=values.dtype), -np.inf,
                    values.astype(np.float64))


class _PathBeliefs:
    def __init__(self, arena, state):
        self.params = state.params
        self.messages = np.asarray(state.messages)
        self.starts = np.asarray(arena.message_starts)
        self.members = {name: (group, row) for group in arena.groups
                        for row, name in enumerate(group.members)}
        self.unsupported = {}
        for kind, groups in (("GFE composite", (*arena.gfe_groups, *arena.gfe_tm_groups,
                                                 *arena.hybrid_gfe_groups)),
                             ("constraint", arena.constraint_groups),
                             ("VMP", arena.dirichlet_groups)):
            for group in groups:
                for name in (*group.members, *getattr(group, "goal_members", ())):
                    self.unsupported[name] = kind
        self.potentials = {}
        self.gathers = {}
        self.delta_slots = set(np.asarray(arena.delta_pairs).ravel().tolist())

    def message(self, slot):
        return self.messages[self.starts[slot]:self.starts[slot + 1]]

    def potential(self, group, row):
        if group.name not in self.potentials:
            table = discrete_potential(group, parameters(group, self.params))
            self.potentials[group.name] = np.asarray(table)
        return self.potentials[group.name][row]

    def check_support(self, arena, parent):
        """Carry literal support toward the query once, including side observations."""
        children = defaultdict(dict)
        for vertex, previous in reversed(parent.items()):
            is_factor, name = vertex
            if not is_factor:
                edge = arena.edges[name]
                if not _card(edge):
                    continue
                a, b = self.message(edge.slot0), self.message(edge.slot1)
                if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                    raise ValueError(f"nonfinite message on edge {name!r}")
                zero = np.asarray(NEG, dtype=a.dtype)
                live = (a != zero) & (b != zero)
                for constraint in children[vertex].values():
                    live &= constraint
            elif name in self.members and self.members[name][0].mode == "bp":
                group, row = self.members[name]
                parent_role = previous[1]
                if group.type == "equality":
                    live = np.ones(group.cards[0], dtype=bool)
                    for constraint in children[vertex].values():
                        live &= constraint
                else:
                    table = self.potential(group, row)
                    live = (table != np.asarray(NEG, dtype=table.dtype)) & np.isfinite(table)
                    for role, constraint in children[vertex].items():
                        axis = group.roles.index(role)
                        shape = [1] * live.ndim
                        shape[axis] = len(constraint)
                        live &= constraint.reshape(shape)
                    axes = tuple(i for i, role in enumerate(group.roles) if role != parent_role)
                    live = np.any(live, axis=axes)
            else:
                # Specialized factors supply their own messages to an ordinary path.
                continue
            if not live.any():
                raise ValueError(f"no joint support at {name!r}")
            if previous is not None:
                predecessor, role = previous
                # On the edge side, two adjacent factors can use the same role name.
                key = role if predecessor[0] else name
                children[predecessor][key] = live

    def joint(self, path):
        joint = None
        for factor, incoming_role, outgoing_role in path:
            if factor not in self.members:
                kind = self.unsupported.get(factor, "unsupported factor")
                raise ValueError(f"path crosses {kind} {factor!r}")
            group, row = self.members[factor]
            if group.mode != "bp":
                raise ValueError(f"path crosses VMP factor {factor!r}")
            left, right = group.roles.index(incoming_role), group.roles.index(outgoing_role)
            if group.name not in self.gathers:
                self.gathers[group.name] = np.asarray(group.gather)
            slots = self.gathers[group.name][row]
            if slots[left] in self.delta_slots or slots[right] in self.delta_slots:
                raise ValueError(f"path crosses a point-mass constraint at {factor!r}")
            incoming = []
            for slot in slots:
                msg = self.message(slot)
                if not np.all(np.isfinite(msg)):
                    raise ValueError(f"nonfinite incoming message at {factor!r}")
                msg = _log_values(msg)
                if not np.isfinite(msg).any():
                    raise ValueError(f"no incoming support at {factor!r}")
                incoming.append(msg)

            if group.type == "equality":
                weight = np.zeros_like(incoming[left])
                for i, message in enumerate(incoming):
                    if i not in (left, right):
                        weight += message
                if joint is None:
                    joint = np.full((len(weight), len(weight)), -np.inf)
                    np.fill_diagonal(joint, incoming[left] + weight)
                else:
                    joint += weight
            else:
                pair = _log_values(self.potential(group, row))
                if np.any(np.isnan(pair) | np.isposinf(pair)):
                    raise ValueError(f"invalid potential at {factor!r}")
                for i, message in enumerate(incoming):
                    if i not in (left, right):
                        shape = [1] * pair.ndim
                        shape[i] = len(message)
                        pair += message.reshape(shape)
                pair = np.moveaxis(pair, (left, right), (0, 1))
                if pair.ndim > 2:
                    pair = np.logaddexp.reduce(pair, axis=tuple(range(2, pair.ndim)))
                if joint is None:
                    joint = incoming[left][:, None] + pair
                else:
                    # Accumulate one separator state at a time to bound scratch space.
                    combined = np.full((len(joint), pair.shape[1]), -np.inf)
                    for k in range(pair.shape[0]):
                        np.logaddexp(combined, joint[:, k, None] + pair[k], out=combined)
                    joint = combined
            right_cavity = incoming[right]
        return joint + right_cavity


def _information(log_joint):
    peak = np.max(log_joint)
    if not np.isfinite(peak):
        raise ValueError("no joint support")
    joint = log_joint - peak
    joint -= np.logaddexp.reduce(joint, axis=None)
    independent = np.logaddexp.reduce(joint, axis=1)[:, None] + np.logaddexp.reduce(joint, axis=0)
    live = np.isfinite(joint)
    value = np.sum(np.exp(joint[live]) * (joint[live] - independent[live]))
    # MI is nonnegative; discard only negative roundoff in the host contraction.
    if value < -128 * np.finfo(joint.dtype).eps * joint.size:
        raise ValueError("negative mutual information exceeds rounding tolerance")
    return float(max(0., value))


def information_gain(arena: SensingView, state: State, query_edge: str,
                     ports: Iterable[str] | str) -> dict[str, float]:
    if query_edge not in arena.edges:
        raise KeyError(f"unknown query edge {query_edge!r}")
    if not _card(arena.edges[query_edge]):
        raise ValueError(f"query {query_edge!r} must be a categorical edge")
    if arena.semiring != "sum":
        raise ValueError("predictive information requires sum-product inference")
    parent = _paths_from(arena, query_edge)
    beliefs = _PathBeliefs(arena, state)
    results = {}
    for port in dict.fromkeys([ports] if isinstance(ports, str) else ports):
        if port not in arena.ports:
            raise KeyError(f"unknown candidate port {port!r}")
        spec = arena.ports[port]
        if spec.role != PortRole.DATA or not _card(arena.edges[spec.variable]):
            raise ValueError(f"candidate {port!r} must be a categorical DATA port")
        if spec.variable == query_edge:
            raise ValueError(f"candidate {port!r} names the query edge {query_edge!r}")
        message = beliefs.message(arena.data_slots[port])
        if len(message) == 1:
            # The lifecycle writes -log(1) for missing and +0 for an observed zero.
            missing = message[0] == 0 and np.signbit(message[0])
        else:
            # Damping can shift a uniform boundary by a common rounding offset.
            missing = (np.isfinite(message[0]) and message[0] != message.dtype.type(NEG)
                       and np.all(message == message[0]))
        if not missing:
            raise ValueError(f"query {query_edge!r}, candidate {port!r} needs a missing observation")
        try:
            path = _factor_path(parent, query_edge, spec.variable)
            log_joint = beliefs.joint(path)
            if not results:
                beliefs.check_support(arena, parent)
            results[port] = _information(log_joint)
        except ValueError as error:
            raise ValueError(f"query {query_edge!r}, candidate {port!r}: {error}") from error
    return results
