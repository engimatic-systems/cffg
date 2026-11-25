"""Native inference and branch lookup for bounded CFFG commitments."""
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
from math import fsum, log

import numpy as np

from cffg.compiler import compile_template
from cffg.core import PortRole, PortSpec
from cffg.engine import Engine
from cffg.models.contingent_validation import inspect_policy


@dataclass(frozen=True)
class Branch:
    """A physically reachable reading and its contingent action."""
    action: int
    reach: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class ContingentPolicy:
    """Joint MAP choices, their conditional Gibbs marginals, and exact tree scores."""
    branches: Mapping[tuple[int, ...], Branch]
    free_energy: float
    committed_free_energy: float
    expected_loss: float

    def action(self, history=()) -> int:
        """Execute an already committed action without an Engine call."""
        return self.branches[tuple(history)].action


class ContingentPlanner:
    """Compile a build_contingent template, including one restored from CFFGZ.

    The sum-product tree scores and reads out the complete policy distribution.
    For joint MAP, integrate passive stochastic report leaves first. Each is a
    normalized conditional with no evidence, so its integral is one for every
    prefix. The reduced max tree therefore has an open port at the former input.
    Maximizing the original report categories instead could favor policies with
    less observation uncertainty.
    """

    def __init__(self, template, arrays, *, max_entries=1_000_000, max_histories=256):
        self.histories, self.groups, self._reach, self._likelihood = inspect_policy(
            template, arrays, max_entries, max_histories)
        reduced = deepcopy(template)
        for label, _ in self.histories.values():
            source = reduced.factors.pop(label).edges["in"]
            del reduced.variables[label]
            del reduced.ports[label]
            port = label.replace("response/", "marginalized/", 1)
            reduced.ports[port] = PortSpec(port, source, PortRole.GLUE)
        self._goals = {label: 1 for label in template.ports if label.startswith("goal/")}
        self.engine = Engine(compile_template(template, arrays=arrays), schedule="two_pass")
        self._map = Engine(compile_template(reduced, arrays=arrays, semiring="max"), schedule="two_pass")
        self._base = self.engine.bind_data(self.engine.init_state(), self._goals)
        self._map_base = self._map.bind_data(self._map.init_state(), self._goals)

    def solve(self) -> ContingentPolicy:
        """Select a joint MAP commitment and score it by conditioning the native ledger.

        At a given history, the previously chosen prefix is known. Conditioning
        all earlier layers before reading each native response yields action
        probabilities for that prefix. Mutually compatible MAP ties use the
        first available action in model order. The uniform action-prior cost is
        removed only in expected_loss; both BFE fields are literal Engine scores.
        """
        max_state = self._map.run(self._map_base, 1).require_converged()
        max_marginals = self._map.marginals(max_state, edges=["prefix/" + g for g in self.groups])
        chosen = {}
        branches = {}
        first_energy = None
        state = self._base
        for depth in range(max(map(len, self.histories)) + 1):
            result = self.engine.run(state, 1)
            state = result.require_converged()
            if first_energy is None:
                first_energy = result.free_energies[-1]
            layer = [(h, row) for h, row in self.histories.items() if len(h) == depth]
            marginals = self.engine.marginals(state, edges=[row[0] for _, row in layer])
            clamps = {}
            for h, (response, group) in layer:
                parent = self.histories[h[:-1]][1] if h else None
                start = chosen[parent] * self.groups[group] if h else 0
                values = max_marginals["prefix/" + group][start:start + self.groups[group]]
                if not np.any(values > 0):
                    raise FloatingPointError(f"Unresolved MAP prefix at observation history {h}")
                action = int(np.argmax(values))
                chosen[group] = start + action
                clamps["commit/" + group] = start + action
                probs = np.asarray(marginals[response][:-1], dtype=np.float64)
                mass = float(probs.sum())
                source_reach = self._reach[h][start + action]
                supported = source_reach > 0
                if supported != (mass > 0):
                    raise FloatingPointError(f"Unresolved branch support at observation history {h}")
                if supported:
                    reach = mass / (mass + float(marginals[response][-1]))
                    # A positive branch can lose relative mass well before it disappears.
                    tolerance = 256 * np.finfo(state.messages.dtype).eps * len(self.histories)
                    if not np.isclose(reach, source_reach, rtol=tolerance, atol=0):
                        raise FloatingPointError(f"Native reach disagrees with its source at {h}")
                    branches[h] = Branch(action, reach, tuple(map(float, probs / mass)))
            state = self.engine.bind_data(state, clamps)
        fixed = self.engine.run(state, 1)
        fixed.require_converged()
        cost = float(fixed.free_energies[-1])
        # Read the same observed factor terms at the commitment. Subtracting a
        # large uniform-prior offset from the runtime scalar can erase a small loss.
        likelihoods = [row[chosen[self.histories[h][1]]] for h, row in self._likelihood.items()]
        if any(p <= 0 for p in likelihoods):
            raise FloatingPointError("The native commitment selected an impossible preference")
        loss = fsum(-log(float(p)) for p in likelihoods)
        source_energy = loss + fsum(log(a) for a in self.groups.values())
        # The native ledger sums node and edge terms in arena precision.
        eps = np.finfo(state.messages.dtype).eps
        bound = 256 * eps * (len(self.histories) + abs(source_energy))
        if not abs(source_energy - cost) <= bound:
            raise FloatingPointError("Committed Bethe energy disagrees with its source factor terms")
        return ContingentPolicy(MappingProxyType(branches), float(first_energy), cost, loss)
