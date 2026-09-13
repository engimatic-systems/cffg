"""Local executor contracts; family tests remain the mathematical authorities."""

import ast
import inspect
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp
from test_compiler_contracts import compiled
from test_engine_contracts import ordinary_case

from cffg.compiled import NEG
from cffg.compiler import compile_template
from cffg.execution import ExecutionDelta
from cffg.execution.registered import (
    ef_parameters,
    execute_discrete,
    execute_ef,
    execute_gaussian,
)
from cffg.kernel_runtime import parameters


def test_synchronous_transport_has_one_owner_outside_engine():
    from cffg import engine
    from cffg.execution import synchronous

    assert engine.make_sweep is synchronous.make_sweep
    assert "_make_step" not in vars(engine)
    tree = ast.parse(inspect.getsource(synchronous))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {
        "cffg.engine",
        "cffg.rules",
        "cffg.csp",
        "cffg.gauss",
        "cffg.efrules",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & {"State", "RunStatus", "RunResult"}


def registered_case(kind, dtype):
    if kind in ("bp", "vmp"):
        template, arrays, _ = ordinary_case(0)
        if kind == "bp":
            template.constraints = []
        arena = compile_template(template, arrays=arrays)
        plan = next(group for group in arena.groups if group.type == "matrix_cat")
        shape = (len(plan.members), len(plan.roles))
        if kind == "vmp":
            shape += (2,)
        incoming = jnp.zeros((*shape, max(plan.cards)), dtype)
        for role, card in enumerate(plan.cards):
            incoming = incoming.at[:, role, ..., card:].set(NEG)
        values = {
            key: jnp.asarray(value, dtype) for key, value in arena.log_params.items()
        }
        params = parameters(plan, values)
        changed = {key: jnp.flip(value, axis=1) for key, value in params.items()}
        return partial(execute_discrete, plan, reduction=logsumexp), (
            (incoming, params),
            (incoming, changed),
        )

    arena = compiled("gaussian" if kind == "gaussian" else "ef")
    values = {key: jnp.asarray(value, dtype) for key, value in arena.log_params.items()}
    if kind == "gaussian":
        plan = next(
            group for group in arena.gauss_groups if group.type == "gaussian_linear"
        )
        shape = (len(plan.members), len(plan.roles), max(plan.dims))
        eta = jnp.full(shape, 0.25, dtype)
        precision = jnp.broadcast_to(
            jnp.eye(shape[-1], dtype=dtype), (*shape, shape[-1])
        )
        params = parameters(plan, values)
        return partial(execute_gaussian, plan), (
            (eta, precision, params),
            (eta * 3, precision, params),
        )

    factor_type = {
        "ef_prior": "gamma",
        "ef_obs": "poisson_obs",
        "ef_equality": "equality",
    }
    plan = next(group for group in arena.ef_groups if group.type == factor_type[kind])
    incoming = jnp.broadcast_to(
        jnp.asarray([1.0, -2.0], dtype), (len(plan.members), len(plan.roles), 2)
    )
    params = ef_parameters(plan, values)
    if kind == "ef_obs":
        observations = jnp.asarray([2.0, 5.0], dtype)
        mask = jnp.asarray([True, False])
        examples = (
            (incoming, params, observations, mask),
            (incoming, params, observations + 1, ~mask),
        )
    elif kind == "ef_prior":
        examples = (
            (incoming, params),
            (incoming, {key: value * 2 for key, value in params.items()}),
        )
    else:
        examples = ((incoming, params), (incoming * 2, params))
    return partial(execute_ef, plan), examples


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    "kind", ("bp", "vmp", "gaussian", "ef_prior", "ef_obs", "ef_equality")
)
def test_registered_outputs_are_local_jax_pytrees_with_dynamic_inputs(kind, dtype):
    execute, examples = registered_case(kind, dtype)
    direct = [execute(*example) for example in examples]
    inputs = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *examples)
    batched = jax.jit(jax.vmap(execute))(*inputs)
    tolerance = 64 * np.finfo(dtype).eps

    assert isinstance(batched, ExecutionDelta)
    assert batched.residual is batched.status is None
    for index, reference in enumerate(direct):
        actual = jax.tree_util.tree_map(
            lambda value, index=index: value[index], batched
        )
        assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
            reference
        )
        for left, right in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(reference),
            strict=True,
        ):
            assert isinstance(left, jax.Array)
            assert left.dtype == np.dtype(dtype)
            assert left.shape == right.shape
            np.testing.assert_allclose(left, right, rtol=tolerance, atol=tolerance)
    assert any(
        not np.allclose(left, right, rtol=tolerance, atol=tolerance)
        for left, right in zip(
            jax.tree_util.tree_leaves(direct[0]),
            jax.tree_util.tree_leaves(direct[1]),
            strict=True,
        )
    )
