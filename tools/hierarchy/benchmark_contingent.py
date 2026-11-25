"""Measure one paid-cue policy depth per fresh CPU process (JAX x64 disabled)."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch
import weakref

import jax
import numpy as np

from cffg.bundle import save_bundle
from cffg.contingent import ContingentPlanner
from cffg.core.serialize import structure_hash
from cffg.engine import Engine
from cffg.experiments import behavioral_epistemics as cue
from cffg.models.contingent import build_contingent_rollout


def buffer_bytes(states):
    buffers = {x.unsafe_buffer_pointer(): x.nbytes for state in states
               for x in jax.tree.leaves(state) if x.nbytes}
    return sum(buffers.values())


def observed_arena_bytes(planner):
    """Count distinct live numerical State buffers across public Engine boundaries."""
    live = {}
    peak = 0

    def record(*states):
        nonlocal peak
        for state in states:
            live[id(state)] = weakref.ref(state)
        present = [s for ref in live.values() if (s := ref()) is not None]
        peak = max(peak, buffer_bytes(present))

    record(planner._base, planner._map_base)
    run, bind = Engine.run, Engine.bind_data

    def run_tracked(engine, state, *args, **kwargs):
        result = run(engine, state, *args, **kwargs)
        record(state, result.state)
        return result

    def bind_tracked(engine, state, *args, **kwargs):
        result = bind(engine, state, *args, **kwargs)
        record(state, result)
        return result

    with patch.object(Engine, "run", run_tracked), patch.object(Engine, "bind_data", bind_tracked):
        planner.solve()
    return peak


def measure(depth):
    if jax.default_backend() != "cpu" or jax.config.x64_enabled:
        raise ValueError("Use JAX_PLATFORMS=cpu JAX_ENABLE_X64=0")
    task = cue.CueTask(cue_quality=.9, reward_utility=4., cue_cost=1.)
    source = dict(d=cue.initial_state(), Bs=cue.transition_stack(), A=cue.observation_matrix(task),
                  c=cue.goal_statistic(task))
    start = perf_counter()
    template, arrays = build_contingent_rollout(f"paid-cue-{depth}", depth=depth, **source)
    built = perf_counter()
    planner = ContingentPlanner(template, arrays)
    jax.block_until_ready((planner._base, planner._map_base))
    prepared = perf_counter()
    first = planner.solve()
    cold = perf_counter()
    warm = []
    for _ in range(3):
        before = perf_counter()
        again = planner.solve()
        warm.append(perf_counter() - before)
        if again != first:
            raise RuntimeError("Repeated policy differed from the cold result")
    peak_arena = observed_arena_bytes(planner)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"policy.cffgz"
        save_bundle(path, template, arrays)
        bundle_bytes = path.stat().st_size
    entries = sum(arrays[key].size for factor in template.factors.values()
                  for key in factor.param_blobs.values())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    files = ["src/cffg/contingent.py", "src/cffg/models/contingent.py",
             "src/cffg/models/contingent_validation.py", "tools/hierarchy/benchmark_contingent.py"]
    root = Path(__file__).resolve().parents[2]
    return dict(depth=depth, jax=jax.__version__, device=str(jax.devices()[0]), dtype="float32",
                build_s=built-start, prepare_s=prepared-built, first_s=cold-prepared,
                warm_s=warm, warm_median_s=float(np.median(warm)), histories=len(planner.histories),
                factors=len(template.factors), entries=entries, source_bytes=sum(x.nbytes for x in arrays.values()),
                bundle_bytes=bundle_bytes, peak_state_buffers_at_engine_boundaries=peak_arena,
                process_peak_rss_bytes=rss if sys.platform == "darwin" else rss*1024,
                root_action=first.action(), reachable_histories=len(first.branches),
                expected_loss=first.expected_loss, native_free_energy=first.free_energy,
                structure_sha256=structure_hash(template),
                source_sha256={f: sha256((root/f).read_bytes()).hexdigest() for f in files})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", required=True, type=int, choices=(1, 2))
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a fresh output file")
    result = measure(args.depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
