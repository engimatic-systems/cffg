"""Independent saved-coordinate qualification; this module runs no optimizer.

Stored comparison summaries are descriptive only. Qualification reconstructs
the native coordinate order with NumPy and evaluates the independent reference.
Vmap members are positional: members[i] must match the single start initial_xi[i].
"""

from collections import Counter

import numpy as np

from hybrid_joint_reference import evaluate, load_case
from hybrid_joint_solver_reference import Coordinate, output_arrays


OUTPUT_GROUPS = ('messages', 'xi', 'energy') + tuple(
    f'{region}_{field}' for region in ('regions', 'separators')
    for field in ('probability', 'displacement', 'mean', 'covariance', 'second')) + tuple(
    f'learning_{field}' for field in
    ('target', 'exposure', 'centered_linear', 'centered_precision', 'linear', 'precision'))
CERTIFICATES = ('natural', 'label_mass', 'conditional_mean', 'conditional_covariance', 'auxiliary', 'primal_dual')
COUNTERS = ('iterations', 'evaluations', 'rejected_trials', 'linear_iterations', 'gauss_newton_steps')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arithmetic_tolerance(group, length, dtype):
    eps = np.finfo(dtype).eps
    if group == 'energy':
        return 512 * max(4, 5 * length) * eps, 0.
    return 512 * 4 * eps, 512 * 4 * eps


def compare_outputs(actual, held, stopped, reference, length, dtype):
    """Recompute exact-group arithmetic and frozen propagated comparisons."""
    for label, groups in (('outputs', actual), ('held outputs', held), ('reference outputs', stopped),
                          ('stopping allowances', reference['stopping_propagation']['groups'])):
        require(set(groups) == set(OUTPUT_GROUPS), f'{label}: exact output groups required')
    arithmetic, comparison = {}, {}
    for group in OUTPUT_GROUPS:
        value, same, expected = (np.asarray(groups[group]) for groups in (actual, held, stopped))
        require(value.shape == same.shape == expected.shape and value.ndim == 1,
                f'{group}: output shape disagrees with independent evaluator')
        require(all(np.all(np.isfinite(x)) for x in (value, same, expected)), f'{group}: nonfinite output')
        atol, rtol = arithmetic_tolerance(group, length, dtype)
        ratio = float(np.max(abs(value - same) / (atol + rtol * abs(same)), initial=0))
        allowance = reference['stopping_propagation']['groups'][group]['scaled_allowance']
        require(np.isfinite(allowance) and allowance >= 0, f'{group}: invalid stopping allowance')
        error = float(np.max(abs(value - expected) / (1 + abs(expected)), initial=0))
        arithmetic[group] = dict(max_allowance_ratio=ratio, passed=ratio <= 1)
        comparison[group] = dict(scaled_error=error, stopping_allowance=allowance,
                                 arithmetic_allowance=atol + rtol, passed=error <= allowance + atol + rtol)
    return arithmetic, comparison


def evaluate_native_coordinate(model, vector):
    """Permute the frozen native order into independent reference storage."""
    coordinate = Coordinate(model)
    vector = np.asarray(vector, dtype=np.float64)
    require(vector.shape == (coordinate.message_size + len(coordinate.auxiliary_indices),),
            'saved coordinate shape disagrees with independent inventory')
    require(np.all(np.isfinite(vector)), 'nonfinite saved coordinate')
    order = sorted(range(coordinate.message_size), key=lambda i:
                   (coordinate.entries[i][2], coordinate.entries[i][1], coordinate.entries[i][0],
                    *coordinate.entries[i][3]))
    independent = vector.copy()
    independent[order] = vector[:coordinate.message_size]
    return evaluate(model, *coordinate.unpack(independent))


def frozen_budgets(spec):
    policy = spec['strategy']
    return dict(outer=policy['max_outer_iterations'], trials=policy['line_search_trials_per_direction'],
                linear=policy['max_linear_iterations'], backtrack=policy['backtrack_factor'], armijo=policy['armijo'])


def validate_inventory(report, spec):
    """Require every declared execution once, with its starts and budgets."""
    invalid = set(spec['initializations']['expected_invalid'])
    expected = [('jit', name, start, perturb) for name in spec['independent_reference']['cases']
                for start in spec['initializations']['xi']
                for perturb in ([False] if name in invalid else [False, True])]
    expected += [('host', 'coupled_t2', start, False) for start in (0., 1.)]
    expected += [('outer_zero', 'coupled_t2', 0., False), ('negative_xi', 'coupled_t2', -1., False),
                 ('nonfinite_site', 'coupled_t2', 0., False)]
    records = report['records']
    singles = [r for r in records if r['execution'] != 'jit_vmap']
    identities = [(r['execution'], r['case'], r['initial_xi'], r['perturbation']) for r in singles]
    require(Counter(identities) == Counter(expected), 'exact execution inventory required')
    budgets = frozen_budgets(spec)
    for record in singles:
        require(record['dtype'] == 'float64', 'checkpoint requires float64')
        wanted = dict(budgets, outer=0) if record['execution'] == 'outer_zero' else budgets
        require(record['budgets'] == wanted, 'execution budgets disagree with frozen policy')
        require(record['expected_status_passed'] is True, 'failed expected execution status')
        if record['case'] in invalid or record['execution'] in ('negative_xi', 'nonfinite_site'):
            require(record['status'] == 'diverged' and record['reason'] == 'invalid_initialization'
                    and record['numerical_status'] != 0 and record['iterations'] == 0,
                    'invalid initialization must diverge before iteration')
        elif record['execution'] == 'outer_zero':
            require(record['status'] in ('exhausted', 'converged') and record['iterations'] == 0,
                    'zero budget must exhaust or certify without an iteration')
            if record['status'] == 'exhausted':
                require(record['reason'] == 'outer_budget' and record['numerical_status'] == 0,
                        'zero-budget exhaustion status disagrees')
        else:
            require(record['status'] == 'converged', 'valid declared execution must converge')
    batches = [r for r in records if r['execution'] == 'jit_vmap']
    require(len(batches) == 1, 'exact vmap execution inventory required')
    batch = batches[0]
    require(batch['case'] == 'coupled_t2' and batch['dtype'] == 'float64'
            and batch['status'] == 'completed' and batch['initial_xi'] == [0., 1.]
            and len(batch['members']) == 2, 'exact vmap member starts required')
    for member in batch['members']:
        require(member['status'] == 'converged', 'valid vmap member must converge')


def validate_saved_parity(actual, expected):
    """Recheck all persisted state fields instead of trusting the parity flag."""
    for name in (*COUNTERS, 'status', 'reason', 'numerical_status', 'last_rejection', 'rejection_counts', 'N', 'M'):
        require(actual[name] == expected[name], f'saved parity disagrees: {name}')
    atol = 512 * 4 * np.finfo(np.float64).eps
    groups = {name: (actual[name], expected[name]) for name in
              ('coordinate', 'xi', 'inactive_xi', 'relative_energy_change', 'free_energy', 'dual_energy')}
    groups.update({name: (actual['certificates'][name], expected['certificates'][name]) for name in CERTIFICATES})
    groups.update({name: (actual['outputs'][name], expected['outputs'][name]) for name in OUTPUT_GROUPS})
    for name, (value, wanted) in groups.items():
        value, wanted = np.asarray(value), np.asarray(wanted)
        require(value.shape == wanted.shape and np.all(np.isfinite(value)) and np.all(np.isfinite(wanted)),
                f'saved parity shape or finiteness disagrees: {name}')
        require(np.all(abs(value - wanted) <= atol * (1 + abs(wanted))), f'saved parity disagrees: {name}')


def validate_saved_report(report, spec, references):
    """Qualify saved outputs with independent mathematics and exact identities.

    Source/spec hashes are a separate gate. Historical arithmetic summaries
    are not used here, so immutable earlier attempts can be rechecked under
    the original absolute energy budget without rewriting their evidence.
    """
    validate_inventory(report, spec)
    singles = {(r['case'], r['initial_xi'], r['perturbation']): r
               for r in report['records'] if r['execution'] == 'jit'}
    converged, parities = [], []
    for record in report['records']:
        if record['execution'] == 'jit_vmap':
            for start, member in zip(record['initial_xi'], record['members'], strict=True):
                converged.append((record['case'], start, member))
                parities.append((member, singles[record['case'], start, False]))
        elif record['status'] == 'converged':
            converged.append((record['case'], record['initial_xi'], record))
            if record['execution'] == 'host':
                parities.append((record, singles[record['case'], record['initial_xi'], False]))
    tau = 64 * np.finfo(np.float64).eps**.75
    models, stopped = {}, {}
    summaries = []
    for name, start, record in converged:
        require(record['proper_active_beliefs'] is True and record['all_gates_pass'] is True
                and record['numerical_status'] == 0 and record['reason'] == 'certified', 'invalid terminal status')
        require(set(record['certificates']) == set(CERTIFICATES), 'exact terminal certificates required')
        certificates = np.asarray([*record['certificates'].values(), record['relative_energy_change']])
        require(np.all(np.isfinite(certificates)) and np.all((0 <= certificates) & (certificates <= tau)),
                'saved terminal certificates exceed frozen tolerance')
        if name not in models:
            models[name] = load_case(name)
        model = models[name]
        reference = references[name, start]
        if (name, start) not in stopped:
            stopped[name, start] = output_arrays(evaluate(model, *Coordinate(model).unpack(
                np.asarray(reference['message_coordinate']))))
        held = evaluate_native_coordinate(model, record['coordinate'])
        require(held['proper_active_beliefs'] and max(held['certificates'].values()) <= tau,
                'independent terminal certificates exceed frozen tolerance')
        arithmetic, comparison = compare_outputs(record['outputs'], output_arrays(held), stopped[name, start],
                                                  reference, model.shape[0], np.float64)
        for label, groups in (('arithmetic', arithmetic), ('propagated comparison', comparison)):
            for group, result in groups.items():
                require(result['passed'], f'{name}/{start} {group}: independent {label} failed')
        for key in ('held_coordinate_arithmetic', 'output_comparison_by_group'):
            require(set(record[key]) == set(OUTPUT_GROUPS), f'{key}: exact output groups required')
        require(record['outputs']['energy'] == [record['free_energy'], record['dual_energy']],
                'saved objectives disagree with energy outputs')
        xi = np.asarray(record['xi'])
        require(xi.shape == model.offsets.shape and np.array_equal(xi, held['xi']) and np.all(xi >= 0),
                'saved xi disagrees with coordinate')
        require(np.array_equal(np.asarray(record['inactive_xi']), xi[~model.auxiliary_support])
                and np.all(xi[~model.auxiliary_support] == 0), 'inactive xi must remain exact zero')
        active = int(model.support[:-1].sum() + model.support[1:].sum())
        dimension = model.shape[-1]
        require(record['N'] == len(record['coordinate'])
                and record['M'] == record['N'] + active * (1 + dimension + dimension * (dimension + 1) // 2),
                'saved coordinate/residual inventory disagrees')
        t, k, d = model.shape
        for region, shape, mask in (
                ('separators', (t, k, d), model.support),
                ('regions', (t - 1, k, k, 2 * d), model.support[:-1, :, None] & model.support[1:, None, :])):
            covariance = np.reshape(record['outputs'][region + '_covariance'], (*shape, shape[-1]))
            probability = np.reshape(record['outputs'][region + '_probability'], shape[:-1])
            require(np.all(np.linalg.eigvalsh(covariance[mask]) > 0), 'saved active covariance must be proper')
            require(np.all(probability[mask] > 0) and np.all(probability[~mask] == 0),
                    'saved probabilities must preserve active support')
        summaries.append(dict(case=name, initial_xi=start, arithmetic=arithmetic, comparison=comparison,
                              independent_certificates=held['certificates']))
    for actual, expected in parities:
        validate_saved_parity(actual, expected)
    return summaries
