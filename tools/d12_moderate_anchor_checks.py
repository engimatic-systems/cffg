"""Bounded native local evaluation against the independent moderate reference.

Run with uv run python tools/d12_moderate_anchor_checks.py. Evaluates two supplied
nonstationary points, without initialization, optimization, complete-program
lowering/compilation/execution, or storage measurement. Timings measure only the
NumPy reference arithmetic, not JAX compilation or convergence.
"""

import json
import time

from d12_compile_scaling import checked_arrays, native_jax


def check():
    jax = native_jax()
    import numpy as np
    from cffg import hybrid_separator as candidate
    from d12_compile_fixture import context
    from d12_moderate_payload import flatten, validate_belief
    from d12_moderate_reference import evaluate, validate_reference

    # This is the local fixed-coordinate evaluator, with no solver state or loop.
    local = jax.jit(candidate.evaluate)
    rows = []
    for horizon in (32, 100):
        coordinate, parameters, sites = context(horizon)
        vector = np.float32(0.01) * np.sin(
            np.arange(coordinate.size, dtype=np.float32) + 1
        )
        vector[coordinate.message_size :] = (
            np.arange(coordinate.size - coordinate.message_size) % 2
        )
        messages, xi = coordinate.unpack(jax.numpy.asarray(vector))
        parameters = parameters._replace(xi=xi)
        checked_arrays((parameters, sites, messages))
        result = jax.block_until_ready(local(parameters, sites, messages))
        checked_arrays(result)
        assert int(result.status) == 0
        arrays = flatten(
            dict(state=dict(coordinate=vector, evaluation=dict(local=result)))
        )
        for name in ("regions", "separators"):
            belief = getattr(result, name)
            for field in ("mean", "second", "full_covariance"):
                arrays[name + "." + field] = np.asarray(getattr(belief, field))
            validate_belief(arrays, name)
        started = time.perf_counter()
        reference = evaluate(vector, horizon)
        seconds = time.perf_counter() - started
        validate_reference(arrays, dict(T=horizon, K=4, d=2))
        rows.append(
            dict(
                T=horizon,
                reference_seconds=seconds,
                primal=float(result.free_energy),
                reference_primal=reference[3],
                dual=float(result.dual_energy),
                reference_dual=reference[4],
            )
        )
    return dict(
        native=not jax.config.x64_enabled,
        fixed_points=rows,
        complete_program_calls=dict(lower=0, compile=0, execute=0),
        optimizer_calls=0,
    )


if __name__ == "__main__":
    print(json.dumps(check()))
