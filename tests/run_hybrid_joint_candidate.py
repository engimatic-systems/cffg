"""Frozen tiny candidate checkpoint; each invocation writes a new attempt file.

Run only after test_hybrid_joint_coordinate.py passes. No moderate solve runs
here. Partial files retain started/failed attempts if compilation is interrupted.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from cffg import hybrid_joint as joint
from cffg import hybrid_joint_solver as solver
from cffg import hybrid_joint_storage as storage
from hybrid_joint_candidate import context, held_coordinate, output_arrays, to_reference
from hybrid_joint_reference import evaluate as independent_evaluate
from hybrid_joint_solver_reference import Coordinate as ReferenceCoordinate, output_arrays as independent_arrays
from hybrid_joint_report import REPORT_SCHEMA, FAILED_COORDINATE_POLICY, compare_outputs

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'tests/fixtures/hybrid-joint-candidate-spec.json'
REFERENCE = ROOT / 'tests/fixtures/hybrid-joint-reference-results.json'
SOURCE = ['src/cffg/hybrid_joint.py', 'src/cffg/hybrid_joint_solver.py',
          'src/cffg/hybrid_joint_storage.py', 'tests/hybrid_joint_candidate.py',
          'tests/run_hybrid_joint_candidate.py', 'tests/hybrid_joint_report.py']
SOURCE += [str(path.relative_to(ROOT)) for path in sorted((ROOT / 'src/cffg').glob('hybrid_*.py'))
           if str(path.relative_to(ROOT)) not in SOURCE]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE}


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(x) for x in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return serial(np.asarray(value).tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, np.generic):
        return serial(value.item())
    return value


def describe(coordinate, model, state, reference):
    status = solver.Outcome(int(state.status)).name.lower()
    result = dict(status=status, reason=solver.Reason(int(state.reason)).name.lower(),
                  numerical_status=int(state.evaluation.status), coordinate=np.asarray(state.coordinate),
                  xi=np.asarray(coordinate.unpack(state.coordinate)[1]),
                  certificates=dict(zip(state.evaluation.certificates._fields, state.evaluation.certificates)),
                  relative_energy_change=float(state.relative_energy_change),
                  free_energy=float(state.evaluation.local.free_energy), dual_energy=float(state.evaluation.local.dual_energy),
                  proper_active_beliefs=int(state.evaluation.status) == 0,
                  inactive_xi=np.asarray(coordinate.unpack(state.coordinate)[1])[~model.auxiliary_support],
                  N=coordinate.size, M=coordinate.residual_size)
    for name in ('iterations', 'evaluations', 'rejected_trials', 'linear_iterations', 'gauss_newton_steps'):
        result[name] = int(getattr(state, name))
    result['rejection_counts'] = {reason.name.lower(): int(state.rejection_counts[reason]) for reason in solver.Rejection}
    result['last_rejection'] = solver.Rejection(int(state.last_rejection)).name.lower()
    if status != 'converged':
        return result
    outputs = output_arrays(coordinate, state.coordinate, state.evaluation)
    result['outputs'] = outputs
    held = independent_arrays(independent_evaluate(model, *to_reference(coordinate, state.coordinate)))
    reference_value = independent_evaluate(model, *ReferenceCoordinate(model).unpack(np.asarray(reference['message_coordinate'])))
    expected = independent_arrays(reference_value)
    arithmetic, comparison = compare_outputs(outputs, held, expected, reference, model.shape[0], state.coordinate.dtype)
    result['held_coordinate_arithmetic'] = arithmetic
    result['output_comparison_by_group'] = comparison
    result['all_gates_pass'] = (max(float(x) for x in state.evaluation.certificates) <= solver.tolerance(state.coordinate.dtype)
                               and float(state.relative_energy_change) <= solver.tolerance(state.coordinate.dtype)
                               and all(x['passed'] for x in arithmetic.values())
                               and all(x['passed'] for x in comparison.values()))
    return result


def parity(state, expected):
    """Compare the entire returned state, including statuses and finite masks."""
    actual_leaves, actual_tree = jax.tree.flatten(state)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    if actual_tree != expected_tree:
        return dict(passed=False, reason='different state structure')
    maximum = 0.
    exact = True
    for actual, wanted in zip(actual_leaves, expected_leaves):
        actual, wanted = np.asarray(actual), np.asarray(wanted)
        if actual.shape != wanted.shape or actual.dtype != wanted.dtype:
            return dict(passed=False, reason='different shape or dtype')
        if not np.issubdtype(actual.dtype, np.floating):
            exact &= np.array_equal(actual, wanted)
            continue
        finite = np.isfinite(wanted)
        exact &= np.array_equal(np.isfinite(actual), finite)
        exact &= np.array_equal(actual[~finite], wanted[~finite], equal_nan=True)
        allowance = 512 * 4 * np.finfo(actual.dtype).eps
        error = abs(actual[finite] - wanted[finite]) / (allowance * (1 + abs(wanted[finite])))
        maximum = max(maximum, float(np.max(error, initial=0)))
    return dict(passed=bool(exact and maximum <= 1), discrete_and_nonfinite_equal=bool(exact),
                max_allowance_ratio=maximum)


def held_buffers(coordinate, parameters, sites, model):
    """Inspect shapes by tracing one held point; execute no optimizer."""
    inventory = {}

    def inspect(p, s, x):
        inventory.update(storage.buffers(coordinate, p, s, x))
        return x

    jax.eval_shape(inspect, parameters, sites, held_coordinate(coordinate, model))
    return inventory


def run(output):
    if output.exists():
        raise ValueError('attempt files are immutable: choose a new output path')
    spec = json.loads(SPEC.read_text())
    references = {(r['case'], r['initial_xi']): r for r in json.loads(REFERENCE.read_text())['records']}
    report = dict(spec_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(), source_sha256=hashes(),
                  report_schema=REPORT_SCHEMA, failed_coordinate_policy=FAILED_COORDINATE_POLICY,
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  versions=dict(jax=jax.__version__, numpy=np.__version__),
                  compilation_cache=os.environ.get('JAX_COMPILATION_CACHE_DIR'),
                  boundary='Tiny float64 candidate only; no moderate/float32 solve or actual iteration memory claim.',
                  approximation_context=json.loads(REFERENCE.read_text())['fixed_auxiliary_approximation_context'], records=[])

    def save():
        output.write_text(json.dumps(serial(report), indent=2, allow_nan=False) + '\n')

    compiled = {}
    baseline = {}
    c, p, s, model = context('coupled_t2')
    report['held_buffer_inventory'] = dict(case='coupled_t2', **held_buffers(c, p, s, model))
    save()

    def runner(c, budgets=solver.Budgets()):
        key = (c.shape, np.asarray(c.support).tobytes(), np.asarray(c.origin).tobytes(), budgets)
        if key not in compiled:
            compiled[key] = jax.jit(lambda p, s, start, perturb: solver.solve(c, p, s, start, perturb=perturb, budgets=budgets))
        return compiled[key]

    def attempt(name, start, perturb, kind='jit', budgets=solver.Budgets(), transform=None):
        c, p, s, model = context(name)
        if transform:
            s = transform(s)
        record = dict(case=name, dtype=str(s.linear.dtype), initial_xi=start, perturbation=perturb, execution=kind,
                      budgets=vars(budgets),
                      status='started')
        report['records'].append(record)
        save()
        began = time.monotonic()
        try:
            if kind == 'host':
                state = solver.solve(c, p, s, start, perturb=perturb, budgets=budgets)
            else:
                state = runner(c, budgets)(p, s, jnp.asarray(start), jnp.asarray(perturb))
            jax.block_until_ready(state)
            record['elapsed_seconds'] = time.monotonic() - began
            record.update(describe(c, model, state, references[(name, start if start >= 0 else 0.)]))
            if name in spec['initializations']['expected_invalid'] or kind in ('negative_xi', 'nonfinite_site'):
                record['expected_status_passed'] = record['status'] == 'diverged'
            elif kind == 'outer_zero':
                record['expected_status_passed'] = record['status'] in ('exhausted', 'converged')
            else:
                record['expected_status_passed'] = record['status'] == 'converged'
            if kind == 'jit' and budgets == solver.Budgets():
                baseline[(name, start, perturb)] = state
            save()
            print(name, start, perturb, kind, record['status'], flush=True)
            if not record['expected_status_passed'] or (record['status'] == 'converged' and not record['all_gates_pass']):
                raise AssertionError(f'checkpoint failed: {name}, {start}, {perturb}, {kind}')
            return state
        except Exception as error:
            if record['status'] == 'started':
                record['status'] = 'execution_error'
            record.update(execution_error=repr(error), elapsed_seconds=time.monotonic() - began)
            save()
            raise

    invalid = set(spec['initializations']['expected_invalid'])
    for name in spec['independent_reference']['cases']:
        for start in spec['initializations']['xi']:
            for perturb in ([False] if name in invalid else [False, True]):
                attempt(name, start, perturb)
    for start in (0., 1.):
        state = attempt('coupled_t2', start, False, 'host')
        expected = baseline[('coupled_t2', start, False)]
        comparison = parity(state, expected)
        report['records'][-1]['host_jit_parity'] = comparison
        save()
        assert comparison['passed'], 'host/JIT state mismatch'
    c, p, s, model = context('coupled_t2')
    report['records'].append(dict(case='coupled_t2', dtype='float64', execution='jit_vmap',
                                 initial_xi=[0., 1.], status='started'))
    save()
    began = time.monotonic()
    batch = jax.jit(jax.vmap(lambda start: solver.solve(c, p, s, start)))(jnp.asarray([0., 1.]))
    jax.block_until_ready(batch)
    record = report['records'][-1]
    record.update(status='completed', elapsed_seconds=time.monotonic() - began, members=[])
    for index, start in enumerate((0., 1.)):
        state = jax.tree.map(lambda x: x[index], batch)
        member = describe(c, model, state, references[('coupled_t2', start)])
        expected = baseline[('coupled_t2', start, False)]
        member['single_vmap_parity'] = parity(state, expected)
        record['members'].append(member)
    save()
    assert all(m['all_gates_pass'] and m['single_vmap_parity']['passed'] for m in record['members'])
    attempt('coupled_t2', 0., False, 'outer_zero', solver.Budgets(outer=0))
    attempt('coupled_t2', -1., False, 'negative_xi')
    attempt('coupled_t2', 0., False, 'nonfinite_site', transform=lambda s: s._replace(linear=s.linear.at[0, 0, 0].set(jnp.nan)))
    save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
