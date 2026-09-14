"""Bounded numerical policy checks; complete registered solves have saved reports."""

import json
import hashlib
from copy import deepcopy
from pathlib import Path
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import hybrid_joint as joint, hybrid_joint_solver as solver
from cffg.hybrid_joint_storage import arrays
from hybrid_joint_candidate import context as make_context, held_coordinate
from hybrid_joint_report import (
    OUTPUT_GROUPS, REPORT_SCHEMA, FAILED_COORDINATE_POLICY, arithmetic_tolerance,
    compare_outputs, validate_inventory, validate_report_policy, validate_saved_report, validate_terminal_variant,
    validate_initial_work,
)

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


def test_frozen_checkpoint_sources(checkpoint_report):
    root = Path(__file__).resolve().parents[1]
    spec_path = root / 'tests/fixtures/hybrid-joint-candidate-spec.json'
    spec = json.loads(spec_path.read_text())
    assert hashlib.sha256(spec_path.read_bytes()).hexdigest() == checkpoint_report['spec_sha256']
    from run_hybrid_joint_candidate import SOURCE
    assert set(checkpoint_report['source_sha256']) == set(SOURCE)
    for path, expected in checkpoint_report['source_sha256'].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
    reference = spec['independent_reference']
    assert hashlib.sha256((root / reference['path']).read_bytes()).hexdigest() == reference['sha256']
    groups = checkpoint_report['held_buffer_inventory']['groups']
    assert all(groups[name] for name in ('derivative_captures', 'linear_state', 'trial_state', 'outer_state'))


@pytest.fixture(scope='module')
def checkpoint_context():
    fixtures = Path(__file__).parent / 'fixtures'
    spec = json.loads((fixtures / 'hybrid-joint-candidate-spec.json').read_text())
    references = {(r['case'], r['initial_xi']): r for r in
                  json.loads((fixtures / 'hybrid-joint-reference-results.json').read_text())['records']}
    return spec, references


def test_saved_terminal_certificates_outputs_and_parity(checkpoint_report, checkpoint_context):
    assert len(validate_saved_report(checkpoint_report, *checkpoint_context)) == 40


@pytest.mark.parametrize('mutation', ['learning_value', 'missing_learning', 'shape', 'extra_group', 'coordinate'])
def test_saved_outputs_require_independent_recomputation(checkpoint_report, checkpoint_context, mutation):
    report = deepcopy(checkpoint_report)
    record = report['records'][0]
    if mutation == 'learning_value':
        record['outputs']['learning_linear'][0] = 1e9
    elif mutation == 'missing_learning':
        for name in OUTPUT_GROUPS:
            if name.startswith('learning_'):
                del record['outputs'][name]
    elif mutation == 'shape':
        record['outputs']['learning_linear'] = [record['outputs']['learning_linear']]
    elif mutation == 'extra_group':
        record['outputs']['unregistered'] = [0.]
    else:
        record['coordinate'][0] += .1
    with pytest.raises(ValueError, match='independent|output groups|shape'):
        validate_saved_report(report, *checkpoint_context)


@pytest.mark.parametrize('mutation', ['duplicate_host', 'vmap_starts', 'duplicate_vmap_member', 'status_variant', 'budget'])
def test_saved_execution_inventory_requires_exact_starts(checkpoint_report, checkpoint_context, mutation):
    report = deepcopy(checkpoint_report)
    records = report['records']
    batch = next(r for r in records if r['execution'] == 'jit_vmap')
    if mutation == 'duplicate_host':
        indices = [i for i, r in enumerate(records) if r['execution'] == 'host']
        records[indices[1]] = deepcopy(records[indices[0]])
    elif mutation == 'vmap_starts':
        batch['initial_xi'] = [0., 0.]
    elif mutation == 'duplicate_vmap_member':
        batch['members'][1] = deepcopy(batch['members'][0])
    elif mutation == 'status_variant':
        next(r for r in records if r['execution'] == 'negative_xi')['initial_xi'] = 0.
    else:
        next(r for r in records if r['execution'] == 'outer_zero')['budgets']['outer'] = 200
    with pytest.raises(ValueError, match='inventory|starts|parity|budgets'):
        if mutation == 'duplicate_vmap_member':
            validate_saved_report(report, *checkpoint_context)
        else:
            validate_inventory(report, checkpoint_context[0])


def test_energy_arithmetic_is_absolute_even_for_large_objectives():
    held = {name: np.zeros(2) for name in OUTPUT_GROUPS}
    held['energy'][:] = 1e6
    actual = deepcopy(held)
    actual['energy'][0] += 1e-8
    reference = {'stopping_propagation': {'groups': {name: {'scaled_allowance': 0.} for name in OUTPUT_GROUPS}}}
    arithmetic, comparison = compare_outputs(actual, held, held, reference, 2, np.float64)
    atol, rtol = arithmetic_tolerance('energy', 2, np.float64)
    assert rtol == 0 and atol == 512 * 10 * np.finfo(np.float64).eps
    assert not arithmetic['energy']['passed']
    assert comparison['energy']['arithmetic_allowance'] == atol


@pytest.mark.parametrize('mutation', [
    'outer_nan_coordinate', 'outer_energy', 'outer_xi', 'outer_flag', 'outer_certificates', 'outer_sentinel',
    'outer_fabricated_convergence', 'swapped_failures', 'failed_finite_message', 'failed_energy',
    'failed_status', 'failed_auxiliary', 'missing_field',
])
def test_saved_terminal_variants_require_consistent_payloads(checkpoint_report, checkpoint_context, mutation):
    report = deepcopy(checkpoint_report)
    variants = {r['execution']: r for r in report['records'] if r['execution'] != 'jit'}
    outer, negative, nonfinite = (variants[name] for name in ('outer_zero', 'negative_xi', 'nonfinite_site'))
    if mutation == 'outer_nan_coordinate':
        outer['coordinate'][0] = 'nan'
    elif mutation == 'outer_energy':
        outer['free_energy'] = 'nan'
    elif mutation == 'outer_xi':
        outer['xi'][0][0][0] = -1.
    elif mutation == 'outer_flag':
        outer['proper_active_beliefs'] = False
    elif mutation == 'outer_certificates':
        outer['certificates']['auxiliary'] = 0.
    elif mutation == 'outer_sentinel':
        outer['relative_energy_change'] = 0.
    elif mutation == 'outer_fabricated_convergence':
        baseline = report['records'][0]
        identity = {'case', 'dtype', 'initial_xi', 'perturbation', 'execution', 'budgets', 'elapsed_seconds'}
        outer.update({key: deepcopy(value) for key, value in baseline.items() if key not in identity})
        outer.update(iterations=0, evaluations=2)
    elif mutation == 'swapped_failures':
        for key in ('coordinate', 'xi', 'certificates', 'relative_energy_change', 'free_energy', 'dual_energy',
                    'proper_active_beliefs', 'inactive_xi', 'N', 'M', 'numerical_status'):
            negative[key], nonfinite[key] = nonfinite[key], negative[key]
    elif mutation == 'failed_finite_message':
        negative['coordinate'][0] = 0.
    elif mutation == 'failed_energy':
        nonfinite['dual_energy'] = 0.
    elif mutation == 'failed_status':
        next(r for r in report['records'] if r['case'] == 'underflow_source_t2')['numerical_status'] = 39
    elif mutation == 'failed_auxiliary':
        negative['certificates']['auxiliary'] = 0.
    else:
        del nonfinite['free_energy']
    with pytest.raises(ValueError, match='zero-budget|zero-update|failed|proper-belief|saved xi|missing required'):
        validate_saved_report(report, *checkpoint_context)


@pytest.mark.parametrize('mutation', [
    'outer_cap', 'boolean', 'negative', 'fractional', 'evaluation_accounting', 'linear_cap', 'fallback_cap',
    'trial_cap', 'histogram_negative', 'histogram_accounting', 'zero_step', 'vmap_cap',
])
def test_saved_counters_obey_execution_budgets(checkpoint_report, checkpoint_context, mutation):
    report = deepcopy(checkpoint_report)
    record = next(r for r in report['records'] if r['case'] == 'coupled_t3')
    if mutation == 'outer_cap':
        record['iterations'] = 201
    elif mutation == 'boolean':
        record['iterations'] = True
    elif mutation == 'negative':
        record['evaluations'] = -1
    elif mutation == 'fractional':
        record['rejected_trials'] = .5
    elif mutation == 'evaluation_accounting':
        record['evaluations'] += 1
    elif mutation == 'linear_cap':
        record.update(gauss_newton_steps=1, linear_iterations=129, rejected_trials=24,
                      evaluations=2 + record['iterations'] + 24)
        record['rejection_counts']['numerical'] = 24
    elif mutation == 'fallback_cap':
        record['gauss_newton_steps'] = record['iterations'] + 1
    elif mutation == 'trial_cap':
        rejected = record['iterations'] * (record['budgets']['trials'] - 1) + 1
        record.update(rejected_trials=rejected, evaluations=2 + record['iterations'] + rejected)
        record['rejection_counts']['numerical'] = rejected
    elif mutation == 'histogram_negative':
        record['rejection_counts']['numerical'] = -1
    elif mutation == 'histogram_accounting':
        record['rejection_counts']['numerical'] = 1
    elif mutation == 'zero_step':
        record = next(r for r in report['records'] if r['execution'] == 'negative_xi')
        record.update(iterations=1, evaluations=3)
    else:
        batch = next(r for r in report['records'] if r['execution'] == 'jit_vmap')
        batch['members'][0]['iterations'] = 201
    with pytest.raises(ValueError, match='counter|budget|histogram'):
        validate_saved_report(report, *checkpoint_context)


def test_saved_failure_policy_is_explicit_and_bound_to_packing_source(checkpoint_report):
    from run_hybrid_joint_candidate import hashes
    metadata = dict(report_schema=REPORT_SCHEMA, failed_coordinate_policy=FAILED_COORDINATE_POLICY,
                    source_sha256=hashes())
    assert validate_report_policy(metadata) == REPORT_SCHEMA
    assert validate_report_policy(checkpoint_report) == 1
    for key, value in (('report_schema', 3), ('failed_coordinate_policy', 'unspecified')):
        changed = deepcopy(metadata)
        changed[key] = value
        with pytest.raises(ValueError, match='schema|policy'):
            validate_report_policy(changed)
    changed = deepcopy(checkpoint_report)
    changed.update(report_schema=REPORT_SCHEMA, failed_coordinate_policy=FAILED_COORDINATE_POLICY)
    with pytest.raises(ValueError, match='packing source'):
        validate_report_policy(changed)


@pytest.mark.parametrize('name,execution', [
    ('coupled_t2', 'negative_xi'), ('coupled_t2', 'nonfinite_site'),
    ('empty_source_t2', 'jit'), ('underflow_source_t2', 'jit'),
])
def test_saved_current_failure_policy_accepts_only_the_declared_sentinel(checkpoint_report, name, execution):
    from hybrid_joint_reference import load_case
    model = load_case(name)
    record = deepcopy(next(r for r in checkpoint_report['records'] if r['case'] == name and r['execution'] == execution))
    # Synthetic schema input, not a new solve or a rewritten historical report.
    record['coordinate'] = np.full(record['N'], np.nan).tolist()
    xi = np.zeros_like(model.offsets)
    xi[model.auxiliary_support] = np.nan
    record['xi'] = xi.tolist()
    validate_terminal_variant(record, model, REPORT_SCHEMA)
    record['coordinate'][0] = 0.
    with pytest.raises(ValueError, match='failed message coordinate'):
        validate_terminal_variant(record, model, REPORT_SCHEMA)


@pytest.mark.parametrize('execution,start,perturb', [
    ('jit', 0., False), ('jit', 0., True), ('jit', 1., False), ('jit', 1., True),
    ('host', 0., False), ('host', 1., False), ('jit_vmap', 0., False), ('jit_vmap', 1., False),
])
def test_saved_zero_update_claims_require_the_declared_initializer(checkpoint_report, checkpoint_context,
                                                                  execution, start, perturb):
    report = deepcopy(checkpoint_report)
    if execution == 'jit_vmap':
        record = next(r for r in report['records'] if r['execution'] == execution)['members'][int(start)]
    else:
        name = 'coupled_t3' if execution == 'jit' else 'coupled_t2'
        record = next(r for r in report['records'] if r['execution'] == execution and r['case'] == name
                      and r['initial_xi'] == start and r['perturbation'] == perturb)
    record.update(iterations=0, evaluations=2, rejected_trials=0, linear_iterations=0,
                  gauss_newton_steps=0, last_rejection='none')
    record['rejection_counts'] = dict.fromkeys(record['rejection_counts'], 0)
    with pytest.raises(ValueError, match='zero-update coordinate'):
        validate_saved_report(report, *checkpoint_context)


@pytest.mark.parametrize('start,perturb', [(0., False), (0., True), (1., False), (1., True)])
def test_saved_zero_update_initializer_preserves_start_and_perturbation(start, perturb):
    c, _, _, model = context('source_masked_t2')
    record = dict(iterations=0, coordinate=np.asarray(held_coordinate(c, model, start, perturb)))
    validate_initial_work(record, model, start, perturb)
    with pytest.raises(ValueError, match='zero-update coordinate'):
        validate_initial_work(record, model, 1 - start, perturb)
    with pytest.raises(ValueError, match='zero-update coordinate'):
        validate_initial_work(record, model, start, not perturb)


@pytest.mark.parametrize('fallback,linear', [(1, 0), (2, 1)])
def test_saved_accepted_fallbacks_require_linear_work(checkpoint_report, checkpoint_context, fallback, linear):
    report = deepcopy(checkpoint_report)
    record = next(r for r in report['records'] if r['case'] == 'coupled_t3')
    rejected = fallback * record['budgets']['trials']
    record.update(gauss_newton_steps=fallback, linear_iterations=linear, rejected_trials=rejected,
                  evaluations=2 + record['iterations'] + rejected)
    record['rejection_counts']['numerical'] = rejected
    with pytest.raises(ValueError, match='fallback or linear counters'):
        validate_saved_report(report, *checkpoint_context)


@pytest.mark.parametrize('name', ['mean_forward', 'mean_reverse', 'variance_forward', 'variance_reverse'])
def test_saved_certified_initializer_cannot_claim_updates(checkpoint_report, checkpoint_context, name):
    report = deepcopy(checkpoint_report)
    record = next(r for r in report['records'] if r['case'] == name and r['initial_xi'] == 0.
                  and r['perturbation'] is False)
    assert record['iterations'] == 0
    record.update(iterations=1, evaluations=3)
    with pytest.raises(ValueError, match='certified initializer cannot have accepted updates'):
        validate_saved_report(report, *checkpoint_context)


@pytest.mark.parametrize('fallback,rejected,last,valid', [
    (1, 24, 'numerical', False), (0, 115, 'none', False),
    (1, 24, 'none', True), (1, 25, 'numerical', True),
    (0, 92, 'none', True), (0, 93, 'none', False), (0, 115, 'numerical', True),
    (1, 116, 'none', True), (1, 117, 'none', False),
])
def test_saved_last_rejection_obeys_successful_search_endpoints(checkpoint_report, checkpoint_context,
                                                               fallback, rejected, last, valid):
    report = deepcopy(checkpoint_report)
    record = next(r for r in report['records'] if r['case'] == 'coupled_t3')
    # Synthetic aggregate counters; no optimization trajectory is asserted.
    record.update(iterations=5, gauss_newton_steps=fallback, linear_iterations=fallback,
                  rejected_trials=rejected, evaluations=2 + 5 + rejected, last_rejection=last)
    record['rejection_counts']['numerical'] = rejected
    if valid:
        assert len(validate_saved_report(report, *checkpoint_context)) == 40
    else:
        with pytest.raises(ValueError, match='last rejection contradicts successful search totals'):
            validate_saved_report(report, *checkpoint_context)
