"""parameters readouts over explicit structure and current numerical inputs."""

from __future__ import annotations

import numpy as np

from cffg import ef, efrules
from cffg.readouts.types import (
    DirichletMarginal,
    ParameterBeliefs,
)
from cffg.readouts.views import (
    DiscreteView,
    GoalView,
)
from cffg.runtime_types import State


def dirichlet_marginals(
    view: DiscreteView, state: State
) -> dict[str, DirichletMarginal]:
    """Parameter-edge name -> proper Dirichlet posterior readout.

    Concentrations are reconstructed from the two packed edge messages:
    the prior's positive ``alpha0`` plus the likelihood's nonnegative
    fractional counts.  Improper pre-sweep state is reported with
    ``proper=False`` and NaN moments rather than resembling a posterior.
    """
    out: dict[str, DirichletMarginal] = {}
    messages = np.asarray(state.messages)

    for e in view.edges.values():
        if not e.param_shape:
            continue
        packed = ef.DIRICHLET.posterior(
            messages[view.message_indices(e.slot0, e.card)],
            messages[view.message_indices(e.slot1, e.card)],
        )
        alpha = packed.reshape(e.param_shape)
        proper = bool(
            ef.DIRICHLET.is_proper_concentration(alpha, axis=tuple(range(alpha.ndim)))
        )
        if proper:
            mean = ef.DIRICHLET.mean_from_concentration(alpha, axis=e.param_axis)
            expected_log = np.asarray(
                efrules.DIRICHLET.mean_stats_from_concentration(
                    alpha, axis=e.param_axis
                )
            )
        else:
            mean = np.full_like(alpha, np.nan)
            expected_log = np.full_like(alpha, np.nan)
        out[e.name] = DirichletMarginal(
            alpha=alpha.copy(),
            mean=mean,
            expected_log=expected_log,
            proper=proper,
        )
    return out


def goal_marginals(
    view: GoalView, state: State, *, parameter_beliefs: ParameterBeliefs
) -> dict[str, DirichletMarginal]:
    """Learned goal-factor name -> its Dirichlet posterior."""
    parameter = parameter_beliefs(state)
    out: dict[str, DirichletMarginal] = {}
    for group in view.gfe_groups:
        for goal, edge in zip(group.goal_members, group.goal_param_edges):
            out[goal] = parameter[edge]
    return out


def goal_prior_updates(
    view: GoalView, state: State, *, parameter_beliefs: ParameterBeliefs
) -> dict[str, np.ndarray]:
    """Return converged goal posteriors keyed by serialized prior blob.

    This dictionary is the explicit trial-rollover/checkpoint boundary.
    Reading it does not mutate this state or a future trial's prior.
    """
    parameter = parameter_beliefs(state)
    out: dict[str, np.ndarray] = {}
    for plan in view.goal_parameter_plans:
        blob = plan.prior_blob
        alpha = parameter[plan.parameter_edges[0]].alpha
        for edge in plan.parameter_edges:
            marginal = parameter[edge]
            if not marginal.proper:
                raise ValueError(
                    f"goal prior {blob!r}: posterior is improper; "
                    "rollover requires a completed proper coordinate"
                )
            if not np.array_equal(alpha, marginal.alpha):
                raise ValueError(
                    f"goal prior blob {blob!r} has inconsistent posteriors "
                    "across compiled goal parameter edges"
                )
        out[blob] = alpha.copy()
    return out
