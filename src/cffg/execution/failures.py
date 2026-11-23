"""Attribute local execution facts before scattering loses factor identity."""

import jax
import jax.numpy as jnp
import numpy as np

from cffg.execution.contracts import (
    DirichletMessages,
    GfeMessages,
    HybridMessages,
    PrecisionMessages,
    RecurrenceMessages,
    TransitionMessages,
)
from cffg.failure_facts import (
    FailureFact,
    FailureReason,
    empty_failure,
    local_failure,
    nonfinite_rows,
)


def simultaneous_failure(left: FailureFact, right: FailureFact) -> FailureFact:
    """Prefer a known lowest factor rank; unknown pooled ownership comes last."""
    choose_left = (left.reason != 0) & (
        (right.reason == 0) | (right.factor < 0)
        | ((left.factor >= 0) & (left.factor <= right.factor))
    )
    return FailureFact(*(jnp.where(choose_left, a, b) for a, b in zip(left, right, strict=True)))


class FailureIndex:
    """Static model-local name ranks; no observations or mutable run state."""

    def __init__(self, arena):
        collections = (
            arena.groups, arena.dirichlet_groups, arena.constraint_groups,
            arena.gauss_groups, arena.mix_groups, arena.recurrence_groups,
            arena.prec_groups, arena.ef_groups, arena.gfe_groups,
            arena.gfe_tm_groups, arena.hybrid_gfe_groups, arena.mu3_groups,
        )
        names = set(arena.member_of)
        for groups in collections:
            for group in groups:
                for field in ("members", "priors", "goal_members", "goal_priors"):
                    names.update(getattr(group, field, ()))
        self.names = tuple(sorted(names))
        self._ranks = {name: rank for rank, name in enumerate(self.names)}

    def ranks(self, members):
        return np.asarray([self._ranks[name] for name in members], dtype=np.int32)

    def ids(self, plan, rows=None):
        ids = jnp.asarray(self.ranks(plan.members))
        return ids if rows is None else ids[rows]

    def delta(self, plan, delta, kind, *, rows=None, ids=None,
              discrete_dtype=None, gaussian_dtypes=None, ef_dtype=None):
        """Check only written coordinates, separating log support from parameters."""
        ids = self.ids(plan, rows) if ids is None else ids
        out = delta.outgoing
        blocks = []
        prior_blocks = []

        def add(values, log=False, dtype=None):
            blocks.extend((value if dtype is None else value.astype(dtype), log)
                          for value in jax.tree.leaves(values))

        def gaussian(values):
            eta_dtype, precision_dtype = gaussian_dtypes or (None, None)
            for value in values:
                add(value.eta, dtype=eta_dtype)
                add(value.precision, dtype=precision_dtype)

        def categorical(values, cards):
            if values.ndim == 2:
                values = values[None, ...]
            for role, card in enumerate(cards):
                add(values[:, role, :card], True, discrete_dtype)

        if isinstance(out, DirichletMessages):
            categorical(out.structural, plan.cards)
            add(out.counts, dtype=discrete_dtype)
            prior_blocks.append((plan.priors, out.prior))
        elif isinstance(out, GfeMessages):
            add(out.latent[:, :plan.cards[1]], True, discrete_dtype)
            add(out.log_prediction[:, :plan.cards[0]], True, discrete_dtype)
            add(out.goal_prediction, dtype=discrete_dtype)
            if out.goal_prior is not None:
                prior_blocks.append((plan.goal_priors, out.goal_prior))
        elif isinstance(out, TransitionMessages):
            for value, card in zip(out, plan.cards, strict=True):
                add(value[:, :card], True, discrete_dtype)
        elif isinstance(out, HybridMessages):
            add(out.selector[:, :plan.k_sel], True, discrete_dtype)
            gaussian(out.gaussian)
        elif isinstance(out, RecurrenceMessages):
            add(out.destination[:, :plan.cards[0]], True, discrete_dtype)
            add(out.source[:, :plan.cards[1]], True, discrete_dtype)
            gaussian((out.drive,))
        elif isinstance(out, PrecisionMessages):
            gaussian((out.gaussian,))
            add(out.precision_parameter, dtype=ef_dtype)
        elif kind == "disc":
            categorical(out, plan.cards)
        elif kind == "gauss":
            gaussian(out)
        else:
            add(out, dtype=ef_dtype)

        invalid = jnp.zeros(ids.shape, dtype=bool)
        for value, log in blocks:
            invalid |= nonfinite_rows(value, log_support=log)
        fact = local_failure(ids, status=delta.status, reasons=delta.failure_reason, nonfinite=invalid)
        for members, value in prior_blocks:
            if discrete_dtype is not None:
                value = value.astype(discrete_dtype)
            prior_ids = jnp.asarray(self.ranks(members))
            if rows is not None:
                prior_ids = prior_ids[rows]
            fact = simultaneous_failure(fact, local_failure(prior_ids, nonfinite=nonfinite_rows(value)))
        return fact


def pooled_goal_failure(arena, messages):
    """A pooled count has multiple contributors, not an invented single owner."""
    invalid = jnp.asarray(False)
    for plan in arena.goal_parameter_plans:
        indices = arena.message_indices(plan.count_scatter[0], plan.cardinality)
        invalid |= jnp.any(~jnp.isfinite(messages[indices]))
    return empty_failure()._replace(
        reason=jnp.where(invalid, FailureReason.NONFINITE_SHARED_GOAL, FailureReason.NONE).astype(jnp.int32)
    )
