"""Independent saved-coordinate qualification; this module runs no optimizer.

Stored comparison summaries are descriptive only. Qualification reconstructs
the native coordinate order with NumPy and evaluates the independent reference.
Vmap members are positional: members[i] must match the single start initial_xi[i].
"""

from collections import Counter

import numpy as np

from hybrid_joint_reference import evaluate, load_case, tilted, zero_messages
from hybrid_joint_solver_reference import Coordinate, output_arrays


OUTPUT_GROUPS = ('messages', 'xi', 'energy') + tuple(
    f'{region}_{field}' for region in ('regions', 'separators')
    for field in ('probability', 'displacement', 'mean', 'covariance', 'second')) + tuple(
    f'learning_{field}' for field in
    ('target', 'exposure', 'centered_linear', 'centered_precision', 'linear', 'precision'))
CERTIFICATES = ('natural', 'label_mass', 'conditional_mean', 'conditional_covariance', 'auxiliary', 'primal_dual')
COUNTERS = ('iterations', 'evaluations', 'rejected_trials', 'linear_iterations', 'gauss_newton_steps')
REJECTIONS = ('none', 'numerical', 'xi_boundary', 'insufficient_decrease')
REPORT_SCHEMA = 2
FAILED_COORDINATE_POLICY = 'all_nan_after_invalid_message'
# Each representation is tied to the packing source that defines it.
PACKING_POLICIES = {
    1: ('retain_active_start_after_invalid_message',
        'fff149fa22c3ab7e36a6deb1b1eec10397671e3f74b323d1b5a688e524dc421e'),
    2: (FAILED_COORDINATE_POLICY, 'd39dced160c0890964007d3e0004cf84833c35cd0527592efc9b8644b7c819c6'),
}


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
    independent = vector.copy()
    independent[native_order(coordinate)] = vector[:coordinate.message_size]
    return evaluate(model, *coordinate.unpack(independent))


def native_order(coordinate):
    return sorted(range(coordinate.message_size), key=lambda i:
                  (coordinate.entries[i][2], coordinate.entries[i][1], coordinate.entries[i][0],
                   *coordinate.entries[i][3]))


def independent_initializer(model, initial_xi, perturb):
    coordinate = Coordinate(model)
    xi = np.where(model.auxiliary_support, initial_xi, 0.)
    _, _, _, messages = tilted(model, zero_messages(model), xi)
    independent = coordinate.pack(messages, xi)
    vector = np.concatenate((independent[native_order(coordinate)], independent[coordinate.message_size:]))
    if perturb:
        vector[:coordinate.message_size] += .01 * np.sin(np.arange(coordinate.message_size) + 1)
    return vector


def validate_initial_work(record, model, initial_xi, perturb):
    """Check the initial loop guard and the unchanged zero-update coordinate."""
    wanted = independent_initializer(model, initial_xi, perturb)
    if record['iterations'] != 0:
        initial = evaluate_native_coordinate(model, wanted)
        require(max(initial['certificates'].values()) > 64 * np.finfo(np.float64).eps**.75,
                'certified initializer cannot have accepted updates')
        return
    vector = np.asarray(record['coordinate'], dtype=np.float64)
    atol, rtol = arithmetic_tolerance('messages', model.shape[0], np.float64)
    require(vector.shape == wanted.shape and np.all(np.isfinite(vector))
            and np.all(abs(vector - wanted) <= atol + rtol * abs(wanted)),
            'zero-update coordinate disagrees with independent initializer')


def validate_report_policy(report):
    schema = report.get('report_schema', 1)
    require(type(schema) is int and schema in PACKING_POLICIES, 'unsupported report schema')
    policy, source_hash = PACKING_POLICIES[schema]
    declared = report.get('failed_coordinate_policy', policy if 'report_schema' not in report else None)
    require(declared == policy, 'failed-coordinate policy disagrees with report schema')
    require(report['source_sha256']['src/cffg/hybrid_joint.py'] == source_hash,
            'failed-coordinate policy disagrees with packing source')
    if schema == REPORT_SCHEMA:
        require('tests/hybrid_joint_report.py' in report['source_sha256'], 'current report requires validator source manifest')
    return schema


def validate_counters(record, budgets):
    """Bound the counters of the frozen converged and zero-step executions."""
    require({*COUNTERS, 'rejection_counts', 'last_rejection'} <= set(record), 'missing required saved counters')
    require(all(type(record[name]) is int and record[name] >= 0 for name in COUNTERS),
            'counters must be nonnegative integers')
    iterations, evaluations, rejected, linear, fallback = (record[name] for name in COUNTERS)
    require(iterations <= budgets['outer'], 'iterations exceed the outer budget')
    # An accepted fallback must move from the zero-initialized CGLS step.
    require(fallback <= iterations and fallback <= linear <= fallback * budgets['linear'],
            'fallback or linear counters exceed their budgets')
    require(fallback * budgets['trials'] <= rejected <= (iterations + fallback) * budgets['trials'] - iterations,
            'rejected trials disagree with the direction budgets')
    require(evaluations == 2 + iterations + rejected, 'evaluation counter disagrees with accepted/rejected trials')
    counts = record['rejection_counts']
    require(set(counts) == set(REJECTIONS)
            and all(type(value) is int and value >= 0 for value in counts.values()),
            'rejection counters require exact nonnegative integer bins')
    require(counts['none'] == 0 and sum(counts.values()) == rejected, 'rejection histogram disagrees with counter')
    last = record['last_rejection']
    require(last in REJECTIONS and (last == 'none' or counts[last] > 0), 'last rejection disagrees with histogram')
    if record['status'] == 'diverged' or budgets['outer'] == 0:
        require((iterations, evaluations, rejected, linear, fallback) == (0, 2, 0, 0, 0),
                'zero-step execution counters disagree with initialization/certification')


def validate_state_fields(record, model):
    required = {'coordinate', 'xi', 'certificates', 'relative_energy_change', 'free_energy', 'dual_energy',
                'proper_active_beliefs', 'inactive_xi', 'N', 'M', 'numerical_status', *COUNTERS,
                'rejection_counts', 'last_rejection'}
    require(required <= set(record), 'missing required saved state fields')
    coordinate = Coordinate(model)
    size = coordinate.message_size + len(coordinate.auxiliary_indices)
    active = int(model.support[:-1].sum() + model.support[1:].sum())
    dimension = model.shape[-1]
    require(type(record['N']) is int and type(record['M']) is int and record['N'] == size
            and record['M'] == size + active * (1 + dimension + dimension * (dimension + 1) // 2),
            'saved coordinate/residual inventory disagrees')
    vector, xi = (np.asarray(record[key], dtype=np.float64) for key in ('coordinate', 'xi'))
    require(vector.shape == (size,) and xi.shape == model.offsets.shape, 'saved coordinate/xi shape disagrees')
    require(np.array_equal(xi[model.auxiliary_support], vector[coordinate.message_size:], equal_nan=True),
            'saved xi disagrees with coordinate')
    inactive = np.asarray(record['inactive_xi'], dtype=np.float64)
    require(np.array_equal(inactive, xi[~model.auxiliary_support]) and np.all(inactive == 0),
            'inactive xi must remain exact zero')
    require(set(record['certificates']) == set(CERTIFICATES), 'exact terminal certificates required')
    require(all(np.asarray(record['certificates'][name]).shape == () for name in CERTIFICATES)
            and all(np.asarray(record[name]).shape == () for name in
                    ('free_energy', 'dual_energy', 'relative_energy_change')), 'saved objectives/certificates must be scalars')
    require(type(record['numerical_status']) is int and 0 <= record['numerical_status'] < 64,
            'invalid numerical status flags')
    require(type(record['proper_active_beliefs']) is bool
            and record['proper_active_beliefs'] == (record['numerical_status'] == 0),
            'proper-belief flag disagrees with numerical status')
    return coordinate, vector, xi


def validate_terminal_variant(record, model, schema):
    """Check finite exhaustion and the declared invalid-input payloads."""
    coordinate, vector, xi = validate_state_fields(record, model)
    if record['status'] != 'converged':
        require(not {'outputs', 'all_gates_pass', 'held_coordinate_arithmetic', 'output_comparison_by_group'} & set(record),
                'nonconverged state cannot carry unqualified convergence outputs')
    certificates = np.asarray([record['certificates'][name] for name in CERTIFICATES], dtype=np.float64)
    energy = np.asarray([record['free_energy'], record['dual_energy']], dtype=np.float64)
    change = np.asarray(record['relative_energy_change'], dtype=np.float64)
    if record['execution'] == 'outer_zero':
        require(record['proper_active_beliefs'] and np.all(np.isfinite(certificates)) and np.all(certificates >= 0),
                'zero-budget state must have proper finite certificates')
        validate_initial_work(record, model, record['initial_xi'], record['perturbation'])
        atol, rtol = arithmetic_tolerance('messages', model.shape[0], np.float64)
        held = evaluate_native_coordinate(model, vector)
        expected_certificates = np.asarray([held['certificates'][name + '_residual'] for name in CERTIFICATES])
        require(np.all(abs(certificates - expected_certificates) <= atol + rtol * abs(expected_certificates)),
                'zero-budget certificates disagree with independent evaluation')
        energy_atol, _ = arithmetic_tolerance('energy', model.shape[0], np.float64)
        require(np.all(np.isfinite(energy)) and np.all(abs(energy - [held['free_energy'], held['dual_energy']]) <= energy_atol),
                'zero-budget energy disagrees with independent evaluation')
        tau = 64 * np.finfo(np.float64).eps**.75
        if max(expected_certificates) > tau:
            require(record['status'] == 'exhausted' and np.isposinf(change),
                    'zero-budget exhaustion must retain its unresolved certificate and initial energy-change sentinel')
        else:
            require(record['status'] == 'converged' and np.isfinite(change) and 0 <= change <= tau,
                    'zero-budget convergence requires complete independent certificates')
        return
    # Persisted Natural flags: INVALID=1, EMPTY_SUPPORT=2, IMPROPER=4,
    # UNDERFLOW=16, NONFINITE=32. All these failed normalizations are nonfinite.
    flags = 1 | 2 | 4 | 32
    if record['case'] == 'underflow_source_t2':
        flags |= 16
    require(record['numerical_status'] == flags and record['proper_active_beliefs'] is False,
            'failed initialization status disagrees with declared variant')
    require(np.all(np.isnan(vector[:coordinate.message_size])), 'failed message coordinate must be NaN')
    expected_xi = np.zeros_like(model.offsets)
    expected_xi[model.auxiliary_support] = record['initial_xi'] if schema == 1 else np.nan
    require(np.array_equal(xi, expected_xi, equal_nan=True), 'failed xi payload disagrees with declared variant/policy')
    require(np.all(np.isnan(energy)) and np.isposinf(change), 'failed initialization objectives/sentinel disagree')
    expected_certificates = np.full(len(CERTIFICATES), np.nan)
    if not coordinate.auxiliary_indices:
        expected_certificates[CERTIFICATES.index('auxiliary')] = 0.
    require(np.array_equal(certificates, expected_certificates, equal_nan=True),
            'failed initialization certificates disagree with active inventory')


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
        validate_counters(record, wanted)
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
        validate_counters(member, budgets)


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
    schema = validate_report_policy(report)
    validate_inventory(report, spec)
    singles = {(r['case'], r['initial_xi'], r['perturbation']): r
               for r in report['records'] if r['execution'] == 'jit'}
    converged, parities = [], []
    for record in report['records']:
        model = load_case(record['case'])
        if record['execution'] == 'outer_zero':
            validate_terminal_variant(record, model, schema)
        if record['execution'] == 'jit_vmap':
            for start, member in zip(record['initial_xi'], record['members'], strict=True):
                validate_state_fields(member, model)
                validate_initial_work(member, model, start, False)
                converged.append((record['case'], start, member))
                parities.append((member, singles[record['case'], start, False]))
        elif record['status'] == 'converged':
            validate_state_fields(record, model)
            validate_initial_work(record, model, record['initial_xi'], record['perturbation'])
            converged.append((record['case'], record['initial_xi'], record))
            if record['execution'] == 'host':
                parities.append((record, singles[record['case'], record['initial_xi'], False]))
        elif record['execution'] != 'outer_zero':
            validate_terminal_variant(record, model, schema)
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
