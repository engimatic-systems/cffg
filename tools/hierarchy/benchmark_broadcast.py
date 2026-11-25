"""Measure balanced versus historical Broadcast using the frozen local active-vision data.

Run after the compiled deterministic sweep loop is installed:
uv run python tools/hierarchy/benchmark_broadcast.py CONSUMER_DIR --output RESULTS.json
"""

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from scipy.special import softmax

from cffg.colony import Colony, _Instance
from cffg.compiler import compile_template
from cffg.core.serialize import structure_hash
from cffg.engine import Engine


def chain_reference(model, grid):
    """Reconnect only the outer equality nodes, as in plates.py at 51ae6a7."""
    prefix = 'grid/pix/_bcast/z/'
    first = model.factors[prefix + 'eq0'].edges['a']
    leaves = [f'grid/pix/{r}/{c}/cls' for r in range(grid) for c in range(grid)]
    for i in range(len(leaves) - 1):
        name = f'{prefix}eq{i}'
        tail = leaves[-1] if i == len(leaves) - 2 else f'{prefix}link{i}'
        model.factors[name] = replace(model.factors[name], edges={
            'a': first if i == 0 else f'{prefix}link{i-1}', 'b': leaves[i], 'c': tail,
        })


def timings(build_model, protocol, counts, pixels, balanced, width, sample_count, out):
    start = perf_counter()
    model = build_model()
    if not balanced:
        chain_reference(model, 7)
    arena = compile_template(model)
    eng = Engine(arena, schedule='flood')
    state = eng.init_state()
    group = next(group for group in arena.groups if group.type == 'matrix_cat')
    expected_log = (jax.scipy.special.digamma(jnp.asarray(counts)) -
                    jax.scipy.special.digamma(jnp.asarray(counts.sum(axis=1, keepdims=True))))
    state = replace(state, params={**state.params, dict(group.parameter_paths)['logf']: expected_log})
    jax.block_until_ready(state.messages)
    expected_log = np.asarray(expected_log)
    key = structure_hash(model)
    colony = Colony()
    colony.engines[key] = eng
    names, indices = [], []
    for name in group.members:
        _, _, row, col, obs = name.split('/')
        i = int(obs.removeprefix('obs'))
        names.append(f'grid/pix/{row}/{col}/p{i}')
        indices.append((4*int(row)+i//4, 4*int(col)+i%4))
    rr, cc = np.asarray(indices).T
    records = pixels[:, rr, cc].astype(np.intp)
    references = softmax(expected_log[np.arange(len(indices)), records].sum(axis=1), axis=1)
    setup_s = perf_counter() - start

    out.update(width=width, setup_s=setup_s, warm=[])

    def measure(ids):
        start = perf_counter()
        for i, image in enumerate(ids):
            obs = dict(zip(names, map(int, records[image]), strict=True))
            colony.instances[i] = _Instance(key, eng.bind_data(state, obs))
        infer_start = perf_counter()
        if width == 1:
            results = {0: eng.run(colony.instances[0].state, protocol['sweep_budget'], tol=protocol['tol'])}
        else:
            results = colony.run(protocol['sweep_budget'], tol=protocol['tol'])
        infer_done = perf_counter()
        marginals = []
        iterations = []
        for i in range(width):
            result = results[i]
            try:
                verified = result.require_converged()
            except RuntimeError:
                out['failed_measurement'] = dict(
                    first_image=int(ids[0]), last_image=int(ids[-1]),
                    wall_s=perf_counter()-start, inference_s=infer_done-infer_start,
                    statuses=[r.status.value for r in results.values()],
                    sweeps=[int(r.iterations) for r in results.values()],
                )
                raise
            marginals.append(eng.marginals(verified, edges=['head/zc'])['head/zc'])
            iterations.append(int(result.iterations))
        end = perf_counter()
        error = float(np.max(np.abs(marginals - references[ids])))
        record = dict(wall_s=end-start, inference_s=infer_done-infer_start, sweeps=iterations,
                      abs_posterior_error=error if np.isfinite(error) else None)
        if not np.isfinite(error) or error > 1e-8:
            out['failed_measurement'] = dict(record, first_image=int(ids[0]), last_image=int(ids[-1]))
            raise ArithmeticError(f'Broadcast posterior error {error} on images {ids[0]}–{ids[-1]}')
        return record

    out['first'] = measure(np.arange(width))
    for i in range(0, sample_count, width):
        out['warm'].append(measure(np.arange(i, i+width)))
    out['warm_images_s'] = sample_count/sum(x['wall_s'] for x in out['warm'])
    out['warm_s_per_group_sweep'] = (sum(x['inference_s'] for x in out['warm']) /
                                     sum(max(x['sweeps']) for x in out['warm']))


def sizes(build_model, balanced):
    model = build_model(28,1)
    if not balanced:
        chain_reference(model,28)
    state = Engine(compile_template(model), schedule='flood').init_state()
    return dict(factors=len(model.factors), variables=len(model.variables),
                initialized_state_bytes=sum(x.nbytes for x in jax.tree.leaves(state)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('consumer', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(Path(__file__).with_name('broadcast_protocol.json').read_text())
    for filename, expected in protocol['consumer_sha256'].items():
        if hashlib.sha256((args.consumer/filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Active-vision input changed: {filename}')
    sys.path.insert(0, str(args.consumer/'src'))
    from active_vision.model import build_model
    if not jax.config.x64_enabled or jax.default_backend() != 'cpu':
        raise ValueError('Benchmark requires the frozen CPU float64 configuration')
    with gzip.open(args.consumer/'data/t10k-images-idx3-ubyte.gz','rb') as f:
        magic, count, height, width = struct.unpack('>IIII', f.read(16))
        if (magic,count,height,width) != (2051,10000,28,28):
            raise ValueError('Unexpected MNIST archive header')
        pixels = np.frombuffer(f.read(), dtype=np.uint8).reshape(count,height,width) >= 128
    counts = np.load(args.consumer/'artifacts/counts.npy', allow_pickle=False)
    report = dict(hardware=platform.uname()._asdict(), jax=jax.__version__,
                  jaxlib=jaxlib.__version__, backend=jax.default_backend(),
                  protocol=protocol, arms={})
    if args.output.exists():
        raise FileExistsError(f'Choose a fresh output path: {args.output}')

    def publish():
        temp = args.output.with_suffix(args.output.suffix + '.partial')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(args.output)

    report['status'] = 'running'
    publish()
    for topology in ('chain','balanced'):
        balanced = topology == 'balanced'
        arm = report['arms'][topology] = {}
        for mode in ('single', 'batch', 'scale'):
            phase = arm[mode] = {}
            try:
                if mode == 'scale':
                    phase.update(sizes(build_model, balanced))
                elif mode == 'single':
                    timings(build_model,protocol,counts,pixels,balanced,1,protocol['single']['images'],phase)
                else:
                    width = protocol['batch']['width']
                    timings(build_model,protocol,counts,pixels,balanced,width,
                            width*protocol['batch']['timed_batches'],phase)
                phase['status'] = 'complete'
            except Exception as exc:
                phase.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                print(f'{topology} {mode}: {phase["error"]}', file=sys.stderr, flush=True)
            finally:
                publish()

    old, new = report['arms'].values()
    gates = {'all_phases_complete': all(phase['status'] == 'complete'
                                        for arm in (old, new) for phase in arm.values())}
    if old['scale']['status'] == new['scale']['status'] == 'complete':
        gates['same_784_sizes'] = old['scale'] == new['scale']
    for mode in ('single','batch'):
        if new[mode]['status'] == 'complete':
            gates[f'{mode}_throughput'] = (
                new[mode]['warm_images_s'] >= protocol['min_balanced_images_per_second_single_and_batch'])
        if old[mode]['status'] == new[mode]['status'] == 'complete':
            def mean_sweeps(arm):
                return np.mean([s for rec in arm[mode]['warm'] for s in rec['sweeps']])
            gates[f'{mode}_fewer_sweeps'] = bool(mean_sweeps(new) < mean_sweeps(old))
    report.update(gates=gates, status='passed' if all(gates.values()) else 'failed')
    publish()
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
