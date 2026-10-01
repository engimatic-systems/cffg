"""Explicit local owners preserve numerical blocks and solver facts under vmap."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp
from test_b_epistemics import build_curiosity_engine
from test_compiler_contracts import compiled
from test_gauss_vmp import gamma_gauss_prec
from test_hybrid_gfe import _prior_template
from test_shared_goals import mixed_goals

from cffg.compiled import NEG
from cffg.compiler import compile_template
from cffg.execution import ExecutionDelta, composites, cross_domain, functional
from cffg.execution.constraints import execute_constraint
from cffg.execution.parameters import (
    ConstraintParameters,
    dirichlet_prior,
    gfe_parameters,
    hybrid_parameters,
    mixture_parameters,
    precision_mean,
    recurrence_parameters,
    transition_parameters,
)


def values(arena, dtype):
    return {key: jnp.asarray(value, dtype) for key, value in arena.log_params.items()}


def gaussian_inputs(members, roles, dim, dtype):
    shape = (members, roles, dim)
    return jnp.full(shape, 0.25, dtype), jnp.broadcast_to(
        jnp.eye(dim, dtype=dtype), (*shape, dim)
    )


def local_case(kind, dtype):
    if kind in ("mixture", "mixture_prior", "hybrid_gfe", "hybrid_prior"):
        hybrid = kind.startswith("hybrid")
        if kind.endswith("prior"):
            template, arrays = _prior_template()
            if not hybrid:
                template.constraints = []
            arena = compile_template(template, arrays=arrays)
        else:
            arena = compiled("hybrid_gfe" if hybrid else "mixture")
        plan = (arena.hybrid_gfe_groups if hybrid else arena.mix_groups)[0]
        count = len(plan.members)
        selector = jnp.zeros((count, plan.k_sel), dtype)
        eta, precision = gaussian_inputs(count, len(plan.dims), max(plan.dims), dtype)
        params = (hybrid_parameters if hybrid else mixture_parameters)(
            plan, values(arena, dtype)
        )
        execute = (
            composites.execute_hybrid_gfe if hybrid else cross_domain.execute_mixture
        )
        return partial(execute, plan), (selector, eta, precision, params)

    if kind == "recurrence":
        arena = compiled("recurrence")
        plan = arena.recurrence_groups[0]
        eta, precision = gaussian_inputs(len(plan.members), 2, plan.dim, dtype)
        incoming = jnp.zeros((len(plan.members), plan.cards[0]), dtype)
        return partial(cross_domain.execute_recurrence, plan), (
            incoming,
            incoming,
            eta,
            precision,
            recurrence_parameters(plan, values(arena, dtype)),
        )

    if kind == "precision":
        template, arrays = gamma_gauss_prec(n=1)
        arena = compile_template(template, arrays=arrays)
        plan = arena.prec_groups[0]
        eta, precision = gaussian_inputs(1, 2, 1, dtype)
        tau = jnp.broadcast_to(jnp.asarray([1.0, -2.0], dtype), (1, 2, 2))
        return partial(cross_domain.execute_precision, plan), (
            tau,
            eta,
            precision,
            precision_mean(plan, values(arena, dtype)),
        )

    if kind == "dirichlet":
        arena = compiled("parameter_edge")
        plan = arena.dirichlet_groups[0]
        incoming = jnp.zeros(
            (len(plan.members), len(plan.roles), 2, max(plan.cards)), dtype
        )
        for role, cardinality in enumerate(plan.cards):
            incoming = incoming.at[:, role, :, cardinality:].set(NEG)
        return partial(composites.execute_dirichlet, plan), (
            incoming,
            dirichlet_prior(plan, values(arena, dtype)),
        )

    if kind in ("gfe", "gfe_learned", "gfe_goal", "gfe_functional"):
        if kind == "gfe_goal":
            template, arrays = mixed_goals()
            arena = compile_template(template, arrays=arrays)
        else:
            arena = compiled(
                "mu3" if kind in ("gfe_learned", "gfe_functional") else "gfe"
            )
        plan = arena.gfe_groups[0]
        if kind == "gfe_learned":
            plan = replace(plan, mu3=False)
        params_values = values(arena, dtype)
        if plan.mu3:
            mean, entropy = particle_statistics(plan.cards, len(plan.members), dtype)[
                1:3
            ]
            params_values[plan.name + ":Abar"] = mean
            params_values[plan.name + ":hbar"] = entropy
        params = gfe_parameters(plan, params_values)
        incoming = jnp.zeros((len(plan.members), plan.cards[1]), dtype)
        goal_counts = jnp.zeros_like(params.goal_prior) if plan.goal_learned else None
        return partial(composites.execute_gfe, plan), (incoming, params, goal_counts)

    if kind in ("transition", "transition_fixed"):
        arena = build_curiosity_engine(True).arena
        plan = arena.gfe_tm_groups[0]
        channel = transition_parameters(plan, values(arena, dtype))
        if kind == "transition_fixed":
            plan = replace(plan, learned=False)
            channel = jnp.log(channel / channel.sum(axis=2, keepdims=True))
        incoming = tuple(
            jnp.zeros((len(plan.members), card), dtype) for card in plan.cards
        )
        return partial(composites.execute_transition_gfe, plan), (*incoming, channel)

    plan = compiled("mu3").mu3_groups[0]
    particles, mean, entropy, log_goal = particle_statistics(
        plan.cards, len(plan.rows), dtype
    )
    incoming = jnp.zeros((len(plan.rows), plan.cards[1]), dtype)
    if kind == "stationary":
        return partial(functional.execute_functional_stationary, plan), (
            incoming,
            mean,
            entropy,
            log_goal,
        )
    assert kind == "importance"
    beliefs = jnp.full_like(incoming, 1 / plan.cards[1])
    return partial(functional.execute_functional_importance, plan), (
        particles,
        beliefs,
        log_goal,
        mean[0],
    )


def particle_statistics(cards, rows, dtype):
    size = 4 * cards[0] * cards[1]
    particles = jnp.arange(1, size + 1, dtype=dtype).reshape(4, *cards)
    particles = particles / particles.sum(axis=1, keepdims=True)
    mean = jnp.broadcast_to(particles.mean(axis=0), (rows, *cards))
    entropy = jnp.broadcast_to(
        -(particles * jnp.log(particles)).sum(axis=1).mean(axis=0), (rows, cards[1])
    )
    return (
        particles,
        mean,
        entropy,
        jnp.full((rows, cards[0]), -jnp.log(cards[0]), dtype),
    )


def assert_vmap(execute, inputs):
    direct = execute(*inputs)
    batched = jax.jit(jax.vmap(execute))(
        *jax.tree_util.tree_map(lambda value: value[None], inputs)
    )
    actual = jax.tree_util.tree_map(lambda value: value[0], batched)
    assert isinstance(actual, ExecutionDelta)
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(direct)
    for left, right in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(direct),
        strict=True,
    ):
        assert isinstance(left, jax.Array)
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        if jnp.issubdtype(left.dtype, jnp.inexact):
            tolerance = 128 * np.finfo(left.dtype).eps
            np.testing.assert_allclose(left, right, rtol=tolerance, atol=tolerance)
        else:
            np.testing.assert_array_equal(left, right)
    return actual


@pytest.mark.parametrize(
    "kind,dtype",
    [pytest.param(kind, dtype, id=f"{kind}-{np.dtype(dtype).name}",
                  marks=pytest.mark.slow if kind == "mixture" and dtype is np.float64 else ())
     for kind in (
        "mixture",
        "mixture_prior",
        "hybrid_gfe",
        "hybrid_prior",
        "recurrence",
        "precision",
        "dirichlet",
        "gfe",
        "gfe_learned",
        "gfe_goal",
        "gfe_functional",
        "transition",
        "transition_fixed",
        "stationary",
        "importance",
    ) for dtype in (np.float32, np.float64)],
)
def test_specialized_blocks_and_existing_facts_survive_jitted_vmap(kind, dtype):
    execute, inputs = local_case(kind, dtype)
    delta = assert_vmap(execute, inputs)
    if kind in ("mixture", "mixture_prior", "precision", "dirichlet"):
        assert delta.status is delta.residual is None
    else:
        np.testing.assert_array_equal(delta.status, 0)
        if kind in ("recurrence", "importance"):
            assert delta.residual is None
        else:
            assert np.all(np.isfinite(delta.residual))


@pytest.mark.parametrize("kind", ("gfe", "stationary", "importance"))
def test_local_divergence_remains_a_numerical_fact(kind):
    execute, inputs = local_case(kind, np.float32)
    inputs = list(inputs)
    field = 2 if kind == "importance" else 0
    inputs[field] = inputs[field].at[0].set(jnp.nan)
    delta = assert_vmap(execute, tuple(inputs))
    assert np.max(delta.status) == 2
    if kind != "importance":
        assert not np.isfinite(np.max(delta.residual))


@pytest.mark.parametrize("kind", ("exactly_k", "at_most_k", "table"))
@pytest.mark.parametrize("reduction", (logsumexp, jnp.max))
def test_constraint_values_are_dynamic_local_arguments(kind, reduction):
    plan = replace(compiled("constraint").constraint_groups[0], type=kind)
    incoming = jnp.asarray([[0.0, -1.0], [-2.0, 0.0], [-0.5, 0.0]], jnp.float32)
    if kind == "table":
        examples = (
            ConstraintParameters(table=jnp.asarray([[0, 0, 0], [1, 1, 1]])),
            ConstraintParameters(table=jnp.asarray([[0, 1, 0], [1, 0, 1]])),
        )
    else:
        examples = tuple(ConstraintParameters(threshold=jnp.asarray(k)) for k in (0, 1))
    execute = partial(execute_constraint, plan, reduction=reduction)
    params = jax.tree_util.tree_map(lambda *arrays: jnp.stack(arrays), *examples)
    batch = jax.jit(jax.vmap(execute, in_axes=(None, 0)))(incoming, params)
    for index, example in enumerate(examples):
        direct = execute(incoming, example)
        np.testing.assert_allclose(batch.outgoing[index], direct.outgoing)
    assert not np.array_equal(batch.outgoing[0], batch.outgoing[1])
    assert batch.status is batch.residual is None
