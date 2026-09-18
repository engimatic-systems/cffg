"""Belief free energy and decoded assignment scoring, without run control."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from cffg import efrules
from cffg import ledger_terms as terms
from cffg.compiled import Arena, ConstraintPlan
from cffg.execution import parameters as _parameters
from cffg.execution.transport import _disc_both, _disc_rows
from cffg.kernel_runtime import discrete_energy as _disc_energy
from cffg.kernel_runtime import discrete_potential as _disc_potential
from cffg.kernel_runtime import parameters as _kernel_params
from cffg.runtime_types import State
from cffg.schedule_contracts import FreeEnergy, Parameters


@dataclass(frozen=True)
class ObjectiveLedger:
    """Two distinct objectives sharing current compiled parameter ownership.

    The raw belief kernel serves fused schedules and direct JAX transforms.
    The host operation retains semiring, readiness and exact-support gates.
    Decoding is the caller's responsibility; assignment scoring never updates
    messages or chooses a public run status.
    """

    arena: Arena
    belief_kernel: FreeEnergy

    @property
    def contract(self) -> dict[str, str | bool | None]:
        adaptive = any(
            g.kernel is not None and g.kernel.binding.iterative
            for g in self.arena.gauss_groups
        )
        return {
            "objective": "energy_with_adaptive_gaussian_surrogates" if adaptive else "declared_bethe_or_generalised_energy",
            "approximate": True if adaptive else None,
            "evidence_bound": None,
            "stationarity": "frozen Gaussian coefficients only" if adaptive else "declared factor contract",
        }

    @property
    def metadata(self) -> dict[str, dict[str, str | bool | None]]:
        """Gaussian node objective meaning; no bound is inferred for a surrogate."""
        result = {}
        for group in self.arena.gauss_groups:
            binding = group.kernel.binding if group.kernel is not None else None
            adaptive = binding is not None and binding.iterative
            for name in group.members:
                result[name] = {
                    "objective": binding.objective if binding is not None else "analytic",
                    "approximate": adaptive,
                    "normalization": "normalized Gaussian conditional" if adaptive else "analytic factor",
                    "stationarity": "frozen coefficients only" if adaptive else "Bethe stationary equations",
                    "evidence_bound": None,
                }
        return result

    def free_energy(
        self,
        state: State,
        *,
        evaluate: FreeEnergy,
        zero_support: Callable[[State], bool],
        require_ready: Callable[[State], None],
    ) -> float:
        if self.arena.semiring == "max":
            raise RuntimeError(
                "min-sum reports an objective value, not a free energy — call Engine.objective"
            )
        require_ready(state)
        # NEG is a finite representation of log zero, so the local ledger can
        # otherwise turn log Z = -inf into a huge finite sentinel score.  Empty
        # exact edge support proves Z=0 and therefore F=+inf.
        if zero_support(state):
            return float("inf")
        return float(
            evaluate(
                state.messages,
                state.g_eta,
                state.g_lam,
                state.g_obs,
                state.g_mask,
                state.ef_eta,
                state.ef_obs,
                state.ef_mask,
                state.params,
            )
        )

    def assignment_energy(
        self, assignment: Mapping[str, int], params: Parameters
    ) -> float:
        """Score the supplied discrete assignment with current parameter values."""
        xhat = assignment
        slot2var = {}
        for e in self.arena.edges.values():
            if e.dim or e.ef or e.param_shape:
                continue
            slot2var[e.slot0] = e.name
            slot2var[e.slot1] = e.name
        total = 0.0
        for g in self.arena.groups:
            vals = [
                [xhat[slot2var[int(g.scatter[i, r])]] for r in range(len(g.roles))]
                for i in range(len(g.members))
            ]
            if g.type == "equality":
                for v in vals:
                    total += 0.0 if v[0] == v[1] == v[2] else np.inf
                continue
            logf = np.asarray(_disc_potential(g, _kernel_params(g, params)))
            for i, v in enumerate(vals):
                total += -float(logf[(i, *v)])
        for c in self.arena.constraint_groups:
            v = [xhat[slot2var[int(c.scatter[j])]] for j in range(len(c.cards))]
            parameters = _parameters.constraint_parameters(c, params)
            total += 0.0 if constraint_satisfied(c, v, parameters) else np.inf
        return total


def constraint_satisfied(
    c: ConstraintPlan, v: list[int], parameters: _parameters.ConstraintParameters
) -> bool:
    if c.type == "alldifferent":
        return len(set(v)) == len(v)
    if c.type == "exactly_k":
        return sum(v) == int(np.asarray(parameters.threshold))
    if c.type == "at_most_k":
        return sum(v) <= int(np.asarray(parameters.threshold))
    # table: the assignment must be one of the allowed tuples
    return any(
        np.array_equal(np.asarray(v), row) for row in np.asarray(parameters.table)
    )


def make_belief_energy(arena: Arena) -> FreeEnergy:
    """Build the existing ordered BFE/GFE sum for raw JIT and schedule use."""
    groups = arena.groups

    def bfe(
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        ef_eta: jax.Array,
        ef_obs: jax.Array,
        ef_mask: jax.Array,
        params: dict[str, jax.Array],
    ) -> jax.Array:
        total = jnp.asarray(0.0, dtype=messages.dtype)
        # prec_groups included so the interior-edge entropy tables run even for
        # a prec-only model: terms.gaussian_energy owns gauss_interior (the out edges),
        # terms.ef_energy owns ef_interior (the prec edges), both EdgeInfo-driven.
        if arena.gauss_groups or arena.prec_groups or arena.recurrence_groups:
            total = total + terms.gaussian_energy(
                arena, g_eta, g_lam, g_obs, g_mask, params
            )
        if arena.mix_groups:
            total = total + terms.mixture_energy(arena, messages, g_eta, g_lam, params)
        if arena.recurrence_groups:
            total = total + terms.recurrence_energy(
                arena, messages, g_eta, g_lam, params
            )
        if arena.hybrid_gfe_groups:
            total = total + terms.hybrid_gfe_energy(
                arena, messages, g_eta, g_lam, params
            )
        if arena.ef_groups or arena.prec_groups:
            total = total + terms.ef_energy(arena, ef_eta, ef_obs, ef_mask, params)
        if arena.prec_groups:
            total = total + terms.precision_energy(arena, g_eta, g_lam, ef_eta, params)
        total = terms.add_dirichlet_terms(total, arena, messages, params)
        for g in groups:
            if g.mode == "vmp":
                inc = _disc_both(arena, messages, g.gather_both, g.cards)
            else:
                inc = _disc_rows(arena, messages, g.gather, g.cards)
            total = total + _disc_energy(g, inc, _kernel_params(g, params)).sum()
        total = terms.add_constraint_terms(total, arena, messages, params)
        total = terms.add_gfe_terms(total, arena, messages, params)
        total = terms.add_transition_gfe_terms(total, arena, messages, params)
        for card, pairs in arena.interior_slot_groups:
            lq = efrules.CATEGORICAL.product(
                messages[arena.message_indices(pairs[:, 0], card)],
                messages[arena.message_indices(pairs[:, 1], card)],
            )
            total = total + efrules.CATEGORICAL.entropy_total(lq)
        return total

    return bfe
