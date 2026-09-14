"""Bounded numerical policy checks; complete registered solves have saved reports."""

import json
from pathlib import Path
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import hybrid_joint as joint, hybrid_joint_solver as solver
from cffg.hybrid_joint_storage import arrays
from hybrid_joint_candidate import context as make_context, held_coordinate

context = lru_cache(maxsize=None)(make_context)


@lru_cache(maxsize=1)
def held_programs():
    c, p, s, _ = context('coupled_t2')
    return (jax.jit(lambda x: joint.evaluate(c, p, s, x)),
            jax.jit(lambda x, e, d: solver.line_search(c, p, s, x, e, d, solver.Budgets(trials=1))),
            jax.jit(lambda x, e: solver.gauss_newton(c, p, s, x, e, solver.Budgets())))


@pytest.mark.parametrize('budget', [0, 1, 8])
def test_cgls_has_rectangular_residual_and_fixed_vector_storage(budget):
    # A small standalone least-squares problem tests the generic linear loop;
    # this is not a dense Jacobian of a hybrid candidate.
    matrix = jnp.asarray([[2., 1.], [-1., 3.], [4., -2.]])
    solution = jnp.asarray([-.75, 1.25])
    residual = -(matrix @ solution)
    at = jax.jit(lambda residual: solver.cgls(residual, lambda v: matrix @ v,
                                            lambda w: matrix.T @ w, 2, budget))
    result = at(residual)
    assert result.step.shape == result.normal_residual.shape == result.direction.shape == (2,)
    assert result.residual.shape == (3,)
    assert result.iteration.dtype == jnp.int32
    assert int(result.iteration) <= budget
    assert bool(result.finite)
    if budget == 8:
        np.testing.assert_allclose(result.step, solution, atol=64 * np.finfo(float).eps)
    elif budget == 0:
        np.testing.assert_array_equal(result.step, 0)
        assert not bool(result.done)
    else:
        assert np.linalg.norm(matrix @ result.step + residual) < np.linalg.norm(residual)
    inventory = arrays(result)
    assert sum(r['elements'] for r in inventory if r['dtype'] == 'float64') == 9


def test_linear_nonfinite_and_stationary_residual_are_distinct():
    residual = jnp.asarray([1., -2., 3.])
    zero = solver.cgls(residual, lambda v: jnp.zeros(3), lambda w: jnp.zeros(2), 2, 8)
    assert bool(zero.done) and bool(zero.finite)
    assert float(jnp.linalg.norm(zero.residual)) > 0
    bad = solver.cgls(residual, lambda v: jnp.full(3, jnp.nan), lambda w: jnp.full(2, jnp.nan), 2, 8)
    assert not bool(bad.finite)
    assert int(bad.iteration) == 0
    huge = jnp.full(2, jnp.finfo(residual.dtype).max / 2)
    overflow = solver.cgls(residual, lambda v: jnp.zeros(3), lambda w: huge, 2, 8)
    assert not bool(overflow.finite)
    assert int(overflow.iteration) == 0


def test_budget_inputs_are_bounded_and_explicit():
    for options in ({'outer': -1}, {'linear': 2.5}, {'trials': None}, {'backtrack': 1}, {'armijo': 0}):
        with pytest.raises(ValueError):
            solver.Budgets(**options)


def test_compiled_memory_interface_keeps_arguments_dynamic():
    from cffg.hybrid_joint_storage import compiled_memory
    value = jnp.ones((3,))
    result = compiled_memory(lambda a, b: a + b, value, value)
    if result['available']:
        assert result['argument_size_in_bytes'] == 2 * value.nbytes
        assert result['output_size_in_bytes'] == value.nbytes
        assert result['temp_size_in_bytes'] >= 0
    else:
        assert result['substitute']


@pytest.mark.parametrize('kind', ['improper', 'xi_boundary'])
def test_held_invalid_trial_keeps_the_accepted_coordinate(kind):
    c, p, s, model = context('coupled_t2')
    x = held_coordinate(c, model, start=0., perturb=False)
    evaluate, line_search, _ = held_programs()
    value = evaluate(x)
    direction = jnp.zeros_like(x)
    if kind == 'improper':
        first_precision = c.constants.size + c.active.size * c.shape[-1]
        direction = direction.at[first_precision].set(-1e6)
        expected = solver.Rejection.NUMERICAL
    else:
        direction = direction.at[c.message_size:].set(-1)
        expected = solver.Rejection.XI_BOUNDARY
    result = line_search(x, value, direction)
    assert not bool(result.accepted)
    assert int(result.evaluations) == 1
    assert int(result.rejection_counts[expected]) == 1
    np.testing.assert_array_equal(result.coordinate, x)
    for a, b in zip(jax.tree.leaves(result.evaluation), jax.tree.leaves(value)):
        np.testing.assert_array_equal(a, b)


def test_held_complete_gauss_newton_step_and_storage_interfaces():
    from cffg import hybrid_joint_storage as storage
    c, p, s, model = context('coupled_t2')
    x = held_coordinate(c, model, perturb=True)
    evaluate, _, gauss_newton = held_programs()
    value = evaluate(x)
    direction, linear = gauss_newton(x, value)
    assert bool(linear.finite)
    assert direction.shape == (c.size,)
    assert linear.residual.shape == (c.residual_size,)
    assert int(linear.iteration) <= 128
    assert set(storage.programs(c)) == {'residual', 'normal_product', 'linear_step', 'iteration', 'certification', 'solve'}
