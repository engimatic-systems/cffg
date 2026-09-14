"""Bounded numerical policy checks; complete registered solves have saved reports."""

import json
import hashlib
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


def test_complete_loop_has_fixed_int32_status_and_counter_shapes():
    c, p, s, _ = context('coupled_t2')
    state = jax.eval_shape(lambda p, s: solver.solve(c, p, s), p, s)
    for name in ('iterations', 'evaluations', 'rejected_trials', 'linear_iterations',
                 'gauss_newton_steps', 'last_rejection', 'status', 'reason'):
        value = getattr(state, name)
        assert value.shape == () and value.dtype == jnp.int32
    assert state.rejection_counts.shape == (len(solver.Rejection),)
    assert state.rejection_counts.dtype == jnp.int32


@pytest.fixture(scope='module')
def checkpoint_report():
    return json.loads((Path(__file__).parent / 'fixtures/hybrid-joint-candidate-results-002.json').read_text())


def test_frozen_checkpoint_sources_and_complete_case_inventory(checkpoint_report):
    root = Path(__file__).resolve().parents[1]
    spec_path = root / 'tests/fixtures/hybrid-joint-candidate-spec.json'
    spec = json.loads(spec_path.read_text())
    assert hashlib.sha256(spec_path.read_bytes()).hexdigest() == checkpoint_report['spec_sha256']
    for path, expected in checkpoint_report['source_sha256'].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
    reference = spec['independent_reference']
    assert hashlib.sha256((root / reference['path']).read_bytes()).hexdigest() == reference['sha256']
    expected = {(name, start, perturb) for name in reference['cases']
                for start in spec['initializations']['xi']
                for perturb in ([False] if name in spec['initializations']['expected_invalid'] else [False, True])}
    singles = [r for r in checkpoint_report['records'] if r['execution'] == 'jit']
    assert {(r['case'], r['initial_xi'], r['perturbation']) for r in singles} == expected
    assert len(singles) == len(expected)
    assert all(r['expected_status_passed'] and r['dtype'] == 'float64' for r in singles)
    assert all(r['budgets'] == vars(solver.Budgets()) for r in singles)
    assert len(checkpoint_report['records']) == len(expected) + 6
    groups = checkpoint_report['held_buffer_inventory']['groups']
    assert all(groups[name] for name in ('derivative_captures', 'linear_state', 'trial_state', 'outer_state'))


def test_saved_terminal_certificates_outputs_and_parity(checkpoint_report):
    from hybrid_joint_reference import load_case
    fixtures = Path(__file__).parent / 'fixtures'
    references = {(r['case'], r['initial_xi']): r for r in
                  json.loads((fixtures / 'hybrid-joint-reference-results.json').read_text())['records']}
    tau, eps = solver.tolerance(np.float64), np.finfo(np.float64).eps
    converged = []
    for record in checkpoint_report['records']:
        if record['execution'] == 'jit_vmap':
            assert record['status'] == 'completed'
            for start, member in zip(record['initial_xi'], record['members'], strict=True):
                assert member['single_vmap_parity']['passed']
                converged.append((record['case'], start, member))
        elif record['status'] == 'converged':
            if record['execution'] == 'host':
                assert record['host_jit_parity']['passed']
            converged.append((record['case'], record['initial_xi'], record))
        else:
            assert record['expected_status_passed']
            if record['execution'] == 'outer_zero':
                assert record['status'] == 'exhausted' and record['iterations'] == 0
            else:
                assert record['status'] == 'diverged'
                assert record['reason'] == 'invalid_initialization' and record['numerical_status'] != 0
    assert len(converged) == 40
    for name, start, record in converged:
        assert record['all_gates_pass'] and record['proper_active_beliefs']
        assert record['numerical_status'] == 0 and record['reason'] == 'certified'
        assert max(record['certificates'].values()) <= tau
        assert record['relative_energy_change'] <= tau
        assert np.all(np.isfinite(record['coordinate'])) and np.all(np.asarray(record['xi']) >= 0)
        assert np.all(np.asarray(record['inactive_xi']) == 0)
        model = load_case(name)
        for group, values in record['outputs'].items():
            assert np.all(np.isfinite(values)), (name, group)
            arithmetic = record['held_coordinate_arithmetic'][group]
            assert arithmetic['passed'] and arithmetic['max_allowance_ratio'] <= 1
            comparison = record['output_comparison_by_group'][group]
            allowance = references[name, start]['stopping_propagation']['groups'][group]['scaled_allowance']
            atol = 512 * max(4, 5 * model.shape[0]) * eps if group == 'energy' else 512 * 4 * eps
            assert comparison['stopping_allowance'] == allowance
            assert comparison['arithmetic_allowance'] == atol + 512 * 4 * eps
            assert comparison['passed'] and comparison['scaled_error'] <= allowance + atol + 512 * 4 * eps
        assert record['outputs']['energy'] == [record['free_energy'], record['dual_energy']]
        t, k, d = model.shape
        for region, shape, active in (
                ('separators', (t, k, d), model.support),
                ('regions', (t - 1, k, k, 2 * d), model.support[:-1, :, None] & model.support[1:, None, :])):
            covariance = np.reshape(record['outputs'][region + '_covariance'], (*shape, shape[-1]))
            probability = np.reshape(record['outputs'][region + '_probability'], shape[:-1])
            assert np.all(np.linalg.eigvalsh(covariance[active]) > 0)
            assert np.all(probability[active] > 0) and np.all(probability[~active] == 0)
