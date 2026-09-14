"""Bounded developer-only D12 compilation measurements; never execute a solve.

Inspect: uv run python tools/d12_compile_scaling.py plan
Run after review: uv run python tools/d12_compile_scaling.py run --output /tmp/d12-compile-001
The output directory must be new. A failed/resource-stopped case ends the plan.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SPEC = Path(__file__).with_name('d12_compile_spec.json')
REPORT_SCHEMA = 'cffg.d12-compile-measurement.v1'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec():
    spec = json.loads(SPEC.read_text())
    expected = [('t32-full', 32, 'full'), ('t32-inference', 32, 'inference'), ('t100-full', 100, 'full')]
    if spec['schema'] != 'cffg.d12-compile-plan.v1':
        raise ValueError('unknown compilation plan schema')
    if [(c['id'], c['T'], c['outputs']) for c in spec['cases']] != expected:
        raise ValueError('the prospective three-case inventory changed')
    if any(c['K'] != 4 or c['d'] != 2 for c in spec['cases']):
        raise ValueError('the frozen moderate shapes changed')
    if spec['budgets'] != dict(outer=200, trials=24, linear=128, backtrack=.5, armijo=.0001):
        raise ValueError('qualified solver budgets changed')
    for name in ('lower_seconds', 'compile_seconds', 'worker_seconds', 'rss_bytes',
                 'sample_seconds', 'terminate_grace_seconds'):
        value = spec['limits'][name]
        if type(value) not in (int, float) or value <= 0:
            raise ValueError(f'invalid resource limit: {name}')
    if spec['runtime']['jax_enable_x64'] is not False or spec['runtime']['jax_explicit_x64_dtypes'] != 'error':
        raise ValueError('native float32 requires x64 disabled and explicit x64 rejected')
    if spec['runtime']['backend'] != 'cpu' or spec['fixture']['initial_xi'] != 0:
        raise ValueError('the fixed CPU/initial-xi configuration changed')
    return spec


def identities():
    paths = sorted((ROOT / 'src/cffg').glob('hybrid_*.py'))
    paths += sorted((ROOT / 'tools').glob('d12_compile*.*'))
    paths += [ROOT / 'uv.lock', ROOT / 'pyproject.toml', ROOT / 'src/cffg/__init__.py',
              ROOT / 'tests/fixtures/hybrid-joint-candidate-spec.json']
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return dict(git_revision=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                spec_sha256=digest(SPEC), sources={str(p.relative_to(ROOT)): digest(p) for p in paths})


def child_environment():
    env = dict(os.environ)
    env.update(JAX_ENABLE_X64='false', JAX_EXPLICIT_X64_DTYPES='error', JAX_PLATFORMS='cpu',
               JAX_ENABLE_COMPILATION_CACHE='false', JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES='',
               JAX_LOG_COMPILES='true', JAX_EXPLAIN_CACHE_MISSES='true')
    env.pop('JAX_COMPILATION_CACHE_DIR', None)
    return env


def native_jax():
    # The supervisor sets these before Python starts; repeat config before workload imports.
    import jax
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_explicit_x64_dtypes', 'error')
    jax.config.update('jax_platforms', 'cpu')
    jax.config.update('jax_enable_compilation_cache', False)
    jax.config.update('jax_persistent_cache_enable_xla_caches', '')
    if jax.config.x64_enabled or jax.default_backend() != 'cpu':
        raise RuntimeError('native float32 CPU setup failed')
    return jax


def checked_arrays(value):
    from cffg.hybrid_joint_storage import arrays
    result = arrays(value)
    for leaf in result:
        dtype = leaf['dtype']
        if ('float' in dtype and dtype != 'float32') or 'complex' in dtype:
            raise ValueError(f'unexpected floating leaf: {leaf}')
    return result


def ir_dtypes(module):
    """Inspect all StableHLO value types, including nested loop/branch blocks."""
    if module is None:
        raise RuntimeError('StableHLO unavailable: cannot establish internal dtype evidence')
    counts = Counter()

    def record(value):
        for dtype in re.findall(r'(?:bf16|f16|f32|f64|f128|complex<[^>]+>)(?!\w)', str(value.type)):
            counts[dtype] += 1

    def visit(operation):
        for value in (*operation.operands, *operation.results):
            record(value)
        for region in operation.regions:
            for block in region.blocks:
                for value in block.arguments:
                    record(value)
                for nested in block.operations:
                    visit(nested)

    visit(module.operation)
    if set(counts) != {'f32'}:
        raise ValueError(f'unexpected StableHLO floating types: {dict(counts)}')
    return dict(value_type_occurrences=dict(counts),
                meaning='Occurrences, not unique buffers or bytes; includes nested operands/results/block arguments')


def xla_memory(executable):
    analysis = executable.memory_analysis()
    if analysis is None:
        return dict(available=False, substitute='logical argument/output leaves and independent wait4 process high-water')
    names = ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes',
             'generated_code_size_in_bytes', 'host_argument_size_in_bytes', 'host_output_size_in_bytes',
             'host_temp_size_in_bytes', 'host_alias_size_in_bytes', 'peak_memory_in_bytes')
    return dict(available=True, estimates={name: None if getattr(analysis, name, None) is None
                                         else int(getattr(analysis, name)) for name in names},
                meaning='XLA estimates; backend-specific extra fields are not process RSS')


def compile_program(function, arguments, report, checkpoint):
    """Keep Lowered/Compiled alive together; one lower and one compile, zero calls."""
    import jax
    descriptors = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), arguments)
    report['arguments'] = checked_arrays(descriptors)
    checkpoint('lower')
    start = time.perf_counter_ns()
    lowered = jax.jit(function).lower(*descriptors)
    report['lower_seconds'] = (time.perf_counter_ns() - start) / 1e9
    checkpoint('lower_inspection')
    report['lowered_outputs'] = checked_arrays(lowered.out_info)
    report['stablehlo_dtypes'] = ir_dtypes(lowered.compiler_ir('stablehlo'))
    checkpoint('compile')
    start = time.perf_counter_ns()
    executable = lowered.compile()
    report['compile_seconds'] = (time.perf_counter_ns() - start) / 1e9
    checkpoint('compiled_inspection')
    report['compiled_outputs'] = checked_arrays(executable.out_info)
    report['xla_memory'] = xla_memory(executable)
    report['calls'] = dict(lower=1, compile=1, execute=0)
    # Objects remain live through reporting; process exit releases them without serialization.
    return lowered, executable


def worker(case_id, output):
    spec = load_spec()
    case = next(c for c in spec['cases'] if c['id'] == case_id)
    report = dict(schema=REPORT_SCHEMA, case=case, execution='not run', status='started',
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  identity=identities(), pid=os.getpid(), measurements={})

    def checkpoint(stage):
        report.update(stage=stage, stage_started_ns=time.monotonic_ns())
        write_json(output / 'worker.json', report)

    checkpoint('setup')
    try:
        jax = native_jax()
        import jaxlib
        import numpy as np
        import cffg
        from dataclasses import asdict
        from cffg import hybrid_joint_solver as solver, hybrid_joint_storage as storage
        from d12_compile_fixture import context, inference_outputs

        if Path(cffg.__file__).resolve().parent != ROOT / 'src/cffg':
            raise RuntimeError('cffg imported from another checkout')
        if asdict(solver.Budgets()) != spec['budgets']:
            raise ValueError('solver defaults disagree with the frozen plan')
        clock = time.get_clock_info('perf_counter')
        report['runtime'] = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                machine=platform.machine(), jax=jax.__version__, jaxlib=jaxlib.__version__,
                                numpy=np.__version__, backend=jax.default_backend(),
                                devices=[str(d) for d in jax.devices()],
                                clock=dict(implementation=clock.implementation, monotonic=clock.monotonic,
                                           adjustable=clock.adjustable, resolution=clock.resolution),
                                environment={k: os.environ.get(k) for k in ('JAX_ENABLE_X64', 'JAX_PLATFORMS',
                                    'JAX_EXPLICIT_X64_DTYPES', 'JAX_ENABLE_COMPILATION_CACHE',
                                    'JAX_COMPILATION_CACHE_DIR', 'XLA_FLAGS')},
                                config={k: v.name if hasattr(v, 'name') else
                                        (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                                        for k, v in jax.config.values.items() if 'cache' in k or 'x64' in k})
        coordinate, parameters, sites = context(case['T'])
        initial_xi = jax.numpy.asarray(spec['fixture']['initial_xi'], jax.numpy.float32)
        jax.block_until_ready((coordinate.metadata(), parameters, sites, initial_xi))
        report['fixed_metadata'] = checked_arrays(coordinate.metadata())
        report['coordinate_sizes'] = dict(N=coordinate.size, M=coordinate.residual_size)
        report['fixture_hashes'] = [dict(leaf=leaf['path'], sha256=hashlib.sha256(np.asarray(value).tobytes()).hexdigest())
            for leaf, value in zip(checked_arrays((parameters, sites, initial_xi)),
                                   jax.tree.leaves((parameters, sites, initial_xi)))]
        complete = storage.programs(coordinate, solver.Budgets(**spec['budgets']))['solve']
        function = complete if case['outputs'] == 'full' else lambda *args: inference_outputs(complete(*args))
        retained = compile_program(function, (parameters, sites, initial_xi), report['measurements'], checkpoint)
        if identities() != report['identity']:
            raise RuntimeError('source or repository state changed during measurement')
        report.update(status='complete', retained_objects=len(retained),
                      reuse='No executable calls or cross-process reuse; objects released when this child exits')
        checkpoint('complete')
        return 0
    except BaseException as error:
        report.update(status='failed', failure=dict(type=type(error).__name__, message=str(error),
                                                  traceback=traceback.format_exc()))
        write_json(output / 'worker.json', report)
        traceback.print_exc()
        return 1


def rss_bytes(raw, system):
    if system == 'Darwin':
        return int(raw)
    if system == 'Linux':
        return int(raw * 1024)
    raise ValueError('ru_maxrss units are only established for Darwin and Linux')


def limit_reason(elapsed, current_rss, progress, limits, now_ns):
    if elapsed > limits['worker_seconds']:
        return 'worker_seconds'
    if current_rss is not None and current_rss > limits['rss_bytes']:
        return 'rss_bytes'
    stage = progress.get('stage')
    if stage in ('lower', 'compile') and (now_ns - progress['stage_started_ns']) / 1e9 > limits[f'{stage}_seconds']:
        return f'{stage}_seconds'
    return None


def cache_log_evidence(path):
    phrases = dict(persistent_hits='Persistent compilation cache hit',
                   persistent_misses='PERSISTENT COMPILATION CACHE MISS', tracing_misses='TRACING CACHE MISS')
    counts = dict.fromkeys(phrases, 0)
    with path.open() as stream:
        for line in stream:
            for name, phrase in phrases.items():
                counts[name] += phrase in line
    return dict(log='compiler.log', log_sha256=digest(path), observed_log_occurrences=counts,
                meaning='All worker compilations, including small layout setup; absent log messages do not prove a cold backend')


def signal_worker(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def supervise(command, output, limits, environment):
    """Measure the actual worker, not uv; preserve partial evidence after stops."""
    system = platform.system()
    rss_bytes(0, system)
    start = time.monotonic()
    stop = None
    stop_started = None
    progress = {}
    snapshots = 0
    peak_sample = 0
    with (output / 'stdout.log').open('w') as stdout, (output / 'compiler.log').open('w') as stderr, \
            (output / 'rss-samples.jsonl').open('w') as samples:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    break
                progress_path = output / 'worker.json'
                if progress_path.exists():
                    progress = json.loads(progress_path.read_text())
                sample = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)],
                                        text=True, capture_output=True, timeout=5)
                current = int(sample.stdout.strip()) * 1024 if sample.returncode == 0 and sample.stdout.strip() else None
                if current is not None:
                    snapshots += 1
                    peak_sample = max(peak_sample, current)
                elapsed = time.monotonic() - start
                samples.write(json.dumps(dict(seconds=elapsed, rss_bytes=current, stage=progress.get('stage'))) + '\n')
                samples.flush()
                reason = limit_reason(elapsed, current, progress, limits, time.monotonic_ns())
                if reason and stop is None:
                    stop, stop_started = reason, time.monotonic()
                    write_json(output / 'resource-stop.json', dict(reason=stop, elapsed_seconds=elapsed,
                                                                 rss_bytes=current, progress=progress))
                    signal_worker(process.pid, signal.SIGTERM)
                if stop is not None and time.monotonic() - stop_started > limits['terminate_grace_seconds']:
                    signal_worker(process.pid, signal.SIGKILL)
                time.sleep(limits['sample_seconds'])
        except BaseException:
            signal_worker(process.pid, signal.SIGKILL)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            stop = stop or 'supervisor_error'
            (output / 'supervisor-error.log').write_text(traceback.format_exc())
    report = dict(exit_code=process.returncode, stop=stop, worker_seconds=time.monotonic() - start,
                  cache_evidence=cache_log_evidence(output / 'compiler.log'),
                  process_high_water=dict(mechanism='parent os.wait4 of actual worker PID; process lifetime, not phase peaks',
                                          raw_ru_maxrss=usage.ru_maxrss, raw_unit='bytes' if system == 'Darwin' else 'KiB',
                                          bytes=rss_bytes(usage.ru_maxrss, system)),
                  rss_samples=dict(count=snapshots, maximum_bytes=peak_sample if snapshots else None,
                                   meaning='sampled current RSS; transient peaks may be missed'))
    write_json(output / 'process.json', report)
    return report


def validate_completed(report, process):
    if process['exit_code'] != 0 or process['stop'] is not None:
        raise ValueError('worker failed or hit a diagnostic limit')
    if report.get('schema') != REPORT_SCHEMA or report.get('status') != 'complete' or report.get('execution') != 'not run':
        raise ValueError('missing, incomplete or executed measurement')
    measured = report['measurements']
    if measured['calls'] != dict(lower=1, compile=1, execute=0):
        raise ValueError('unexpected compile/execute inventory')
    for field in ('lower_seconds', 'compile_seconds'):
        if not isinstance(measured[field], (int, float)) or not 0 <= measured[field] < float('inf'):
            raise ValueError('missing or invalid phase timing')
    for name in ('arguments', 'lowered_outputs', 'compiled_outputs'):
        if not measured[name]:
            raise ValueError('empty leaf evidence')
        for leaf in measured[name]:
            if ('float' in leaf['dtype'] and leaf['dtype'] != 'float32') or 'complex' in leaf['dtype']:
                raise ValueError('saved evidence contains a non-float32 floating leaf')
    if measured['stablehlo_dtypes']['value_type_occurrences'].keys() != {'f32'}:
        raise ValueError('invalid internal dtype evidence')
    if report['runtime']['config']['jax_enable_x64'] is not False:
        raise ValueError('x64 was enabled')
    if report['runtime']['config']['jax_explicit_x64_dtypes'] != 'ERROR':
        raise ValueError('explicit x64 requests were permitted')
    if report['runtime']['config']['jax_enable_compilation_cache'] is not False:
        raise ValueError('persistent cache was enabled')
    if report['runtime']['config']['jax_persistent_cache_enable_xla_caches'] != '':
        raise ValueError('XLA auxiliary caches were enabled')
    if report['runtime']['backend'] != 'cpu':
        raise ValueError('unexpected backend')
    memory = measured['xla_memory']
    if not memory['available'] and not memory['substitute']:
        raise ValueError('missing substitute for unavailable XLA memory analysis')
    if memory['available']:
        for name in ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes'):
            value = memory['estimates'][name]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('invalid XLA memory estimate')
    if process['process_high_water']['bytes'] <= 0:
        raise ValueError('missing independent process high-water evidence')


def run(output):
    spec, identity = load_spec(), identities()
    if identity['git_status']:
        raise ValueError('commit setup and use a clean checkout before the measurement')
    if output.resolve().is_relative_to(ROOT):
        raise ValueError('use a new evidence directory outside the checkout')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'spec.json', spec)
    summary = dict(schema=REPORT_SCHEMA, identity=identity, status='started', execution='not run', cases=[])
    write_json(output / 'summary.json', summary)
    for case in spec['cases']:
        directory = output / case['id']
        directory.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--case', case['id'], '--output', str(directory)]
        write_json(directory / 'invocation.json', dict(command=command, limits=spec['limits']))
        process = supervise(command, directory, spec['limits'], child_environment())
        entry = dict(case=case, process=process)
        try:
            report = json.loads((directory / 'worker.json').read_text())
            validate_completed(report, process)
            if report['identity'] != identity:
                raise ValueError('measurement source identity changed')
            entry.update(status='complete', lower_seconds=report['measurements']['lower_seconds'],
                         compile_seconds=report['measurements']['compile_seconds'],
                         xla_memory=report['measurements']['xla_memory'])
        except (ValueError, KeyError, TypeError, OSError) as error:
            entry.update(status='failed', failure=str(error))
        summary['cases'].append(entry)
        summary['status'] = 'failed' if entry['status'] == 'failed' else 'running'
        write_json(output / 'summary.json', summary)
        if entry['status'] == 'failed':
            print(f"Stopped at {case['id']}; partial evidence: {output}")
            return 1
    summary['status'] = 'complete'
    write_json(output / 'summary.json', summary)
    print(f'Compilation-only evidence: {output}')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', '_worker'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', choices=('t32-full', 't32-inference', 't100-full'))
    args = parser.parse_args()
    if args.action == 'plan':
        print(json.dumps(load_spec(), indent=2))
        return 0
    if args.output is None or (args.action == '_worker' and args.case is None):
        parser.error('run requires --output; _worker also requires --case')
    args.output = args.output.resolve()
    return worker(args.case, args.output) if args.action == '_worker' else run(args.output)


if __name__ == '__main__':
    raise SystemExit(main())
