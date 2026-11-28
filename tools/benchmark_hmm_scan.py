"""Fresh CPU/x64 timings for exact scheduled HMM propagation and its references.

Run with the optional bench dependency group installed. Each CFFG propagation
returns the entire message arena and the attributed failure fact.
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cffg.benchmarks.scheduling import _median_ms, _reference_hmm, hmm_chain
from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus
from cffg.schedule_program import make_propagate


def measure(n):
    from dynamax.hidden_markov_model.inference import hmm_smoother

    k = 2
    a = b = np.eye(k)
    d = np.full(k, .5)
    data = {f"y{n}": 0}
    observations = np.full(n, -1)
    observations[-1] = 0
    template = hmm_chain(n, k)
    start = time.perf_counter()
    arena = compile_template(template, arrays={"A": a, "B": b, "d": d})
    compiled = time.perf_counter()
    engine = Engine(arena)
    scheduled = time.perf_counter()
    state = engine.bind_data(engine.init_state(), data)
    args = (state.messages, state.g_eta, state.g_lam, state.ef_eta, state.g_obs,
            state.g_mask, state.ef_obs, state.ef_mask, state.params)
    start_cold = time.perf_counter()
    result = engine.run(state, 1)
    jax.block_until_ready(result.state)
    cold = time.perf_counter() - start_cold
    assert result.status is RunStatus.CONVERGED and result.iterations == 1
    selected = make_propagate(arena, engine.schedule.stages).hmm_fused
    assert selected and all(selected)
    generic = eqx.filter_jit(make_propagate(arena, engine.schedule.stages, fuse_hmm=False))

    def timed(fn):
        return _median_ms(lambda: jax.block_until_ready(fn()), 40, warmup=20)

    raw = engine._propagate(*args)
    for ours, other in zip(jax.tree.leaves(raw), jax.tree.leaves(generic(*args)), strict=True):
        np.testing.assert_allclose(ours, other, atol=128*np.finfo(float).eps,
                                   rtol=128*np.finfo(float).eps)
    unfused_ms = timed(lambda: generic(*args))
    fused_ms = timed(lambda: engine._propagate(*args))
    status_ms = timed(lambda: engine.run(state, 1).state)
    readout_ms = timed(lambda: engine.marginals(result.state))

    # The original low-level ceiling projects the smoothed_probs field inside JIT.
    # Its shape is (T, K), including the posterior at every observed state.
    likelihoods = jnp.zeros((n, k)).at[-1, 1].set(-jnp.inf)
    smoother = jax.jit(lambda ll: hmm_smoother(jnp.asarray(b @ d), jnp.asarray(b.T),
                                               ll).smoothed_probs)
    strict_ms = timed(lambda: smoother(likelihoods))
    matched_ms, matched = _reference_hmm(k, b, a, d, observations, 40)
    marginals = engine.marginals(result.state)
    state_posterior = np.stack([marginals[f"za{i}"] for i in range(1, n+1)])
    np.testing.assert_allclose(smoother(likelihoods), state_posterior, atol=1e-14)
    np.testing.assert_allclose(matched, state_posterior, atol=1e-14)

    flood = Engine(arena, schedule="flood")
    def flooding():
        return flood.run(state, 3*n + 10, tol=1e-12)
    flood_result = flooding()
    assert flood_result.status is RunStatus.CONVERGED
    flood_ms = timed(lambda: flooding().state)
    return dict(T=n, K=k, fused_scan_stages=len(selected), message_values=state.messages.size,
                iterations=result.iterations, max_posterior_error=float(np.max(abs(matched-state_posterior))),
                compile_ms=1e3*(compiled-start), schedule_ms=1e3*(scheduled-compiled),
                cold_status_ms=1e3*cold, generic_propagation_ms=unfused_ms, fused_propagation_ms=fused_ms,
                fused_status_ms=status_ms, readout_ms=readout_ms, flooding_ms=flood_ms,
                flood_iterations=flood_result.iterations, matched_full_ms=matched_ms,
                strict_smoothed_probs_ms=strict_ms, fused_over_strict=fused_ms/strict_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        parser.error("choose a fresh output path")
    jax.config.update("jax_enable_x64", True)
    assert jax.default_backend() == "cpu"
    cpu = (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
           if platform.system() == "Darwin" else platform.processor())
    record = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  hardware=cpu, os=platform.platform(), jax=jax.__version__, platform=jax.default_backend(),
                  x64=jax.config.x64_enabled, warmups=20, repetitions=40,
                  strict_output="low-level Dynamax smoothed_probs[T,K] projected inside JIT, prepared log likelihoods",
                  matched_output="CategoricalHMM.smoother posterior object with smoothed_probs read outside JIT",
                  cffg_output="all directed message values, Gaussian/EF blocks, attributed failure",
                  rows=[])
    for n in (32, 64, 128, 256):
        row = measure(n)
        record["rows"].append(row)
        print(json.dumps(row), flush=True)
    primary = record["rows"][-1]
    record["target_met"] = primary["fused_propagation_ms"] <= .396 and primary["fused_over_strict"] <= 20
    options.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
