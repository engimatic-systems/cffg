"""Shared learned-goal ownership, count pooling, and parameter free energy."""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from cffg.compiled import Arena, GfePlan, GoalParameterPlan
from cffg.rules import goal_dirichlet_energy


def build_goal_parameter_plans(
    groups: Sequence[GfePlan],
) -> tuple[GoalParameterPlan, ...]:
    """Derive one owner per serialized prior, without changing group addresses."""
    by_blob: dict[str, list[tuple[GfePlan, tuple[int, ...]]]] = {}
    for group in sorted(groups, key=lambda value: value.name):
        rows_by_blob: dict[str, list[int]] = {}
        for row, blob in enumerate(group.goal_prior_blobs):
            rows_by_blob.setdefault(blob, []).append(row)
        for blob, rows in sorted(rows_by_blob.items()):
            by_blob.setdefault(blob, []).append((group, tuple(rows)))
    plans = []
    for blob, contributors in sorted(by_blob.items()):
        cardinalities = {group.cards[0] for group, _ in contributors}
        if len(cardinalities) != 1:
            raise ValueError(
                f"goal prior {blob!r}: incompatible outcome cardinalities {sorted(cardinalities)}"
            )
        plans.append(
            GoalParameterPlan(
                prior_blob=blob,
                cardinality=cardinalities.pop(),
                group_rows=tuple((group.name, rows) for group, rows in contributors),
                parameter_edges=tuple(
                    group.goal_param_edges[row]
                    for group, rows in contributors
                    for row in rows
                ),
                prior_scatter=np.array(
                    [
                        group.goal_prior_scatter[row]
                        for group, rows in contributors
                        for row in rows
                    ],
                    dtype=np.int64,
                ),
                count_scatter=np.array(
                    [
                        group.goal_count_scatter[row]
                        for group, rows in contributors
                        for row in rows
                    ],
                    dtype=np.int64,
                ),
            )
        )
    return tuple(plans)


def pooled_prediction(
    plan: GoalParameterPlan, predictions: Mapping[str, jax.Array]
) -> jax.Array:
    """Sum original member predictions, never replicated pooled messages."""
    parts = [predictions[name][jnp.asarray(rows)] for name, rows in plan.group_rows]
    return jnp.concatenate(parts, axis=0).sum(axis=0)


def write_goal_counts(
    arena: Arena,
    messages: jax.Array,
    predictions: Mapping[str, jax.Array],
) -> jax.Array:
    """Publish one new count coordinate to all aliases, before sweep damping."""
    for plan in arena.goal_parameter_plans:
        indices = arena.message_indices(plan.count_scatter, plan.cardinality)
        messages = messages.at[indices].set(pooled_prediction(plan, predictions))
    return messages


def shared_goal_energy(
    arena: Arena,
    messages: jax.Array,
    params: Mapping[str, jax.Array],
    predictions: Mapping[str, jax.Array],
) -> jax.Array:
    """Charge D6c.5 once per owner, including its off-coordinate correction.

    Predictions must be recomputed by the ledger at the evaluated state; a
    stored count message need not equal their sum away from a fixed point.
    """
    total = jnp.asarray(0.0, dtype=messages.dtype)
    for plan in arena.goal_parameter_plans:
        name, rows = plan.group_rows[0]
        alpha0 = params[name + ":c_alpha0"][rows[0]]
        indices = arena.message_indices(plan.count_scatter[0], plan.cardinality)
        alpha = alpha0 + messages[indices]
        total = total + goal_dirichlet_energy(
            alpha, alpha0, pooled_prediction(plan, predictions)
        )
    return total
