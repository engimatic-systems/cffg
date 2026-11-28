#!/usr/bin/env python3
"""Profile certified point training on a deterministic 60-transition CSTR plate.

Launch in a fresh process with a disabled persistent JAX cache. The timed path
uses NeuralTrainer without replacing numerical operations. A separate replay
synchronizes the matrix-free GMRES adjoint alone and compares the reassembled
gradient with the production fused derivative. The replay incurs extra JIT
compiles and is outside all trainer timings.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
from unittest.mock import patch

from benchmark_ordinary_iteration import compile_log, hardware


ROOT = Path(__file__).resolve().parents[1]


def capture_command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def finite(value):
    return float(value) if math.isfinite(float(value)) else None


class Clock:
    def __init__(self, jax):
        self.jax = jax
        self.events = []
        self.compiles = None

    def call(self, name, operation, *args, ready=None, **kwargs):
        start_index = len(self.compiles)
        start = time.perf_counter()
        try:
            result = operation(*args, **kwargs)
            if ready is not None:
                self.jax.block_until_ready(ready(result))
            return result
        finally:
            elapsed = time.perf_counter() - start
            events = self.compiles[start_index:]
            self.events.append({
                "phase": name, "seconds": elapsed, "compiles": events,
                "xla_compile_s": sum(e["seconds"] for e in events if e["phase"] == "XLA compilation"),
            })


def summarize(events):
    grouped = {}
    for event in events:
        grouped.setdefault(event["phase"], []).append(event)
    return {name: {
        "calls": len(rows), "median_s": statistics.median(r["seconds"] for r in rows),
        "total_s": sum(r["seconds"] for r in rows),
        "xla_compile_s": sum(r["xla_compile_s"] for r in rows),
    } for name, rows in grouped.items()}


def inference_fields(result):
    status = {
        "status": result.status.value, "iterations": result.iterations,
        "termination_reason": result.termination_reason.value,
        "outer_map_evaluations": result.outer_map_evaluations,
        "trust_trials": result.trial_evaluations, "linear_iterations": result.linear_iterations,
        "final_free_energy": finite(result.free_energies[-1]) if result.free_energies else None,
        "outer_residual": finite(result.outer_residual),
        "diagnostic": asdict(result.failure) if result.failure is not None else None,
    }
    return status


def step_fields(step):
    derivative = step.derivative
    return {
        "applied": step.applied, "message": step.message, "optimizer_step": step.state.step,
        "objective": finite(step.objective), "inference": inference_fields(step.inference),
        "weights": step.state.weights.tolist(),
        "derivative": None if derivative is None else {
            "status": derivative.status.value, "message": derivative.message,
            "root_residual": finite(derivative.root_residual),
            "adjoint_residual": finite(derivative.adjoint_residual),
            "adjoint_gain": finite(derivative.adjoint_gain),
            "gradient": None if derivative.gradient is None else derivative.gradient.tolist(),
        },
    }


def param_metadata(trainer):
    arena = trainer.engine.arena
    chart = trainer._derivative.chart
    return {
        "schedule": trainer.engine.schedule.kind, "logical_slots": arena.n_slots,
        "root_coordinates": chart.size, "dtype": trainer.dtype.name,
        "parameter_shape": list(trainer.initial_weights.shape),
        "parameter_blobs": [asdict(blob) for blob in trainer.parameters.blobs],
    }


def setup_trainer(clock, template, arrays, observations, config, report):
    from cffg.engine import Engine
    from cffg.training import NeuralTrainer

    run = Engine.run

    def reference(engine, *args, **kwargs):
        solved = clock.call("constructor.reference_inference", run, engine, *args,
                            ready=lambda result: result.state, **kwargs)
        report["reference"] = inference_fields(solved)
        return solved

    with patch.object(Engine, "run", reference):
        trainer = clock.call("constructor.total", NeuralTrainer, template, arrays, observations,
                             config=config, ready=lambda t: t.reference.state)
    report["program"] = param_metadata(trainer)
    return trainer


def fit_trainer(clock, trainer, warm_repetitions, report):
    phases = ("first", "warmup") + ("warm",)*warm_repetitions
    records = report["steps"] = []
    infer = trainer.infer
    derivative = trainer._derivative
    certificate, differentiate = derivative._certificate, derivative._differentiate
    state = trainer.initial_state()
    first = None
    for phase in phases:
        def forward(*args, **kwargs):
            return clock.call(f"{phase}.forward_inference", infer, *args,
                              ready=lambda result: result.state, **kwargs)

        def certify(*args, **kwargs):
            return clock.call(f"{phase}.derivative_certificate", certificate, *args,
                              ready=lambda output: output, **kwargs)

        def differentiate_measured(*args, **kwargs):
            return clock.call(f"{phase}.fused_derivative", differentiate, *args,
                              ready=lambda output: output, **kwargs)

        with (patch.object(trainer, "infer", forward),
              patch.object(derivative, "_certificate", certify),
              patch.object(derivative, "_differentiate", differentiate_measured)):
            step = clock.call(f"{phase}.step_total", trainer.step, state,
                              ready=lambda result: (result.state.weights, result.state.first_moment,
                                                    result.state.second_moment))
        records.append({"phase": phase, **step_fields(step)})
        state = step.require_applied()
        if first is None:
            first = step
    return first


def adjoint_replay(clock, trainer, step, warm_repetitions):
    """Time explicitly separated operators using the same inputs and GMRES settings."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.scipy.sparse.linalg import gmres

    from cffg.fixed_point import RootProgram, root_context
    from cffg.outer import OuterArrays, chart_coordinates

    derivative = trainer._derivative
    engine, chart, params = trainer.engine, derivative.chart, derivative.parameters
    program = RootProgram(chart, engine._iter_step, engine._reconstruct)
    tolerance = derivative.tolerance
    cfg = trainer.config
    state = step.inference.state
    arrays = OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)
    context = root_context(chart, state)._replace(base=arrays._replace(ef_eta=state.ef_eta.reshape(-1)))
    coordinates = chart_coordinates(chart, arrays)
    weights = step.derivative.weights

    def residual(value, theta, dynamic):
        dynamic = dynamic._replace(params=params.inject(theta, dynamic.params))
        return program.residual(value, dynamic)

    def energy(value, theta, dynamic):
        dynamic = dynamic._replace(params=params.inject(theta, dynamic.params))
        return program.energy(value, dynamic, engine._ledger.belief_kernel)

    energy_derivative = jax.jit(jax.grad(energy, argnums=(0, 1)))

    @jax.jit
    def solve(u, theta, dynamic, rhs):
        if not chart.size:
            return jnp.zeros_like(u), jnp.asarray(0., theta.dtype), jnp.asarray(0., theta.dtype)
        _, transpose = jax.vjp(lambda value: residual(value, theta, dynamic), u)

        def operator(vector):
            return transpose(vector)[0]

        adjoint, _ = gmres(operator, rhs, tol=tolerance, atol=0.,
                           restart=min(cfg.krylov_dimension, chart.size),
                           maxiter=cfg.adjoint_iterations, solve_method="incremental")
        scale = jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(theta.dtype).tiny)
        return adjoint, jnp.linalg.norm(operator(adjoint) - rhs)/scale, jnp.linalg.norm(adjoint)/scale

    @jax.jit
    def correction(u, theta, dynamic, adjoint, partial):
        _, transpose = jax.vjp(lambda value: residual(u, value, dynamic), theta)
        return partial - transpose(adjoint)[0]

    energy_u, energy_theta = clock.call("replay.energy_derivative", energy_derivative,
                                       coordinates, weights, context, ready=lambda output: output)
    adjoint, linear, gain = clock.call("replay.adjoint_first", solve, coordinates, weights,
                                      context, energy_u, ready=lambda output: output)
    for _ in range(warm_repetitions):
        adjoint, linear, gain = clock.call("replay.adjoint_warm", solve, coordinates, weights,
                                          context, energy_u, ready=lambda output: output)
    gradient = clock.call("replay.parameter_correction", correction, coordinates, weights,
                          context, adjoint, energy_theta, ready=lambda output: output)
    truth = np.asarray(step.derivative.require_valid())
    np.testing.assert_allclose(np.asarray(gradient), truth, atol=2e-9, rtol=2e-9)
    if not np.isfinite(float(linear)) or float(linear) > tolerance:
        raise ValueError("isolated adjoint failed its original-system residual check")
    return {
        "root_coordinates": int(chart.size), "krylov_dimension": cfg.krylov_dimension,
        "maximum_restarts": cfg.adjoint_iterations, "adjoint_relative_residual": float(linear),
        "adjoint_gain": float(gain), "working_tolerance": tolerance,
        "gradient": gradient.tolist(), "max_absolute_gradient_difference": float(np.max(abs(gradient-truth))),
        "attribution": "Identical inputs and GMRES controls. Separate executables change fusion.",
    }


def oracle(clock, template, arrays, observations, config, report):
    import jax.numpy as jnp
    import numpy as np

    from physical_mean_training_fixture import scalar_marginal

    trainer = setup_trainer(clock, template, arrays, observations, config, report)
    initial = trainer.initial_state()
    stepped = trainer.step(initial)
    report["step"] = step_fields(stepped)
    fitted = stepped.require_applied()
    centers = np.array([arrays[f"z{i}"][0] for i in range(len(observations))], np.float64)
    data = np.array([y[0] for y in observations.values()], np.float64)
    prior_var = float(arrays["Pz"][0, 0])
    output_var = float(arrays["Q"][0, 0]) + float(arrays["R"][0, 0])
    exact, gradient, jacobian = scalar_marginal(initial.weights, centers, data, prior_var, output_var)
    if np.linalg.matrix_rank(jacobian) != len(gradient):
        raise ArithmeticError("cooling design does not distinguish both point parameters")
    actual = np.asarray(stepped.derivative.require_valid())
    delta = np.eye(2).tolist() + [[.6, .8]]
    expected_direction = np.asarray(delta) @ gradient
    actual_direction = np.asarray(delta) @ actual
    limit = 3e-4 if actual.dtype == np.float32 else 2e-8
    np.testing.assert_allclose(actual_direction, expected_direction, atol=limit, rtol=limit)
    np.testing.assert_allclose(stepped.objective, exact, atol=limit, rtol=limit)

    # Compare a public re-solve to centered differences of an independent
    # marginal, whose exact gradient above does not depend on CFFG messages.
    direction = np.asarray(delta[-1])
    h = np.finfo(actual.dtype).eps**(1/3)
    values = []
    for sign in (1, -1):
        perturbed = replace(initial, weights=jnp.asarray(initial.weights + sign*h*direction, dtype=actual.dtype))
        result = trainer.infer(perturbed)
        values.append(trainer.engine.bfe(result.require_converged()))
    difference = (values[0] - values[1])/(2*h)
    np.testing.assert_allclose(difference, expected_direction[-1], atol=16*limit, rtol=16*limit)
    report["independent_oracle"] = {
        "objective": float(exact), "gradient": gradient.tolist(), "direction_vectors": delta,
        "expected_directional": expected_direction.tolist(), "actual_directional": actual_direction.tolist(),
        "public_resolved_finite_difference": float(difference), "difference_step": float(h),
        "mean_design_rank": int(np.linalg.matrix_rank(jacobian)), "tolerance": limit,
    }

    next_step = trainer.step(fitted).require_applied()
    with tempfile.TemporaryDirectory(prefix="cffg-trainer-checkpoint-") as directory:
        file = Path(directory)/"continuation.npz"
        trainer.save_checkpoint(file, fitted)
        restored = setup_trainer(clock, template, arrays, observations, config, {})
        resumed = restored.step(restored.load_checkpoint(file)).require_applied()
    for field in ("weights", "first_moment", "second_moment"):
        np.testing.assert_array_equal(getattr(next_step, field), getattr(resumed, field))
    if next_step.step != resumed.step or next_step.identity != resumed.identity:
        raise RuntimeError("checkpoint changed optimizer identity or position")
    report["checkpoint"] = {"bitwise_continuation": True, "optimizer_step": resumed.step}

    no_data = {}
    try:
        vacuous = setup_trainer(clock, template, arrays, dict.fromkeys(observations), config, no_data)
    except Exception as error:
        if no_data.get("reference", {}).get("status") == "converged" or "reference" not in no_data:
            raise
        report["no_observations"] = {**no_data, "error": str(error)}
    else:
        update = vacuous.step(vacuous.initial_state())
        np.testing.assert_allclose(update.derivative.require_valid(), 0., atol=limit, rtol=0.)
        report["no_observations"] = {**no_data, "step": step_fields(update)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--case", choices=("retained", "oracle"), default="retained")
    cli.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    cli.add_argument("--warm-repetitions", type=int, default=3)
    cli.add_argument("--timing-note", default="No scheduling provenance was supplied.")
    args = cli.parse_args()
    if args.warm_repetitions < 1:
        cli.error("warm repetitions must be positive")
    if args.output.exists():
        cli.error("output already exists")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = str(args.dtype == "float64").lower()
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"

    import cffg
    import jax
    import jaxlib
    import numpy as np
    from cffg.core.serialize import structure_hash
    from cffg.neural_artifacts import array_digest
    from cffg.training import TrainingConfig
    from physical_mean_training_fixture import cstr_training, cooling_training

    if Path(cffg.__file__).resolve().parent != ROOT/"src/cffg":
        raise RuntimeError(f"import resolved to a different CFFG checkout: {cffg.__file__}")
    if jax.default_backend() != "cpu":
        raise ValueError("expected the CPU backend")
    dtype = np.dtype(args.dtype)
    template, arrays, observations = (cstr_training if args.case == "retained" else cooling_training)(dtype)
    config = TrainingConfig(learning_rate=.002, inference_iterations=50,
                            inference_tolerance=1e-8 if dtype == np.float64 else 2e-5)
    sources = (Path(__file__), ROOT/"tools/physical_mean_training_fixture.py",
               ROOT/"tools/cstr_runtime_fixture.py", ROOT/"tools/benchmark_ordinary_iteration.py",
               ROOT/"src/cffg/training.py", ROOT/"src/cffg/implicit.py", ROOT/"uv.lock")
    frozen_sources = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    head = capture_command("git", "rev-parse", "HEAD")
    report = {
        "status": "started", "case": args.case, "runtime_head": head,
        "source_sha256": frozen_sources,
        "hardware": hardware(), "runtime": {"python": platform.python_version(), "jax": jax.__version__,
            "jaxlib": jaxlib.__version__, "numpy": np.__version__, "dtype": dtype.name,
            "persistent_jax_cache": False, "device": str(jax.devices()[0])},
        "timing_note": args.timing_note, "config": asdict(config),
        "fixture": {"structure_hash": structure_hash(template), "transitions": len(observations),
            "input_shape": [len(observations), arrays["Pz"].shape[0]],
            "output_shape": [len(observations), arrays["R"].shape[0]],
            "arrays_sha256": array_digest(arrays), "data_sha256": array_digest(observations)},
        "protocol": {
            "warm_repetitions": args.warm_repetitions, "post_update_warmups": 1,
            "sync": "block on state, weights, moments and nested JIT outputs at measured boundaries",
            "inspection": "nested wall spans overlap; per-phase compile entries also appear in enclosing spans",
            "historical": ("Original observations and invocation unavailable. Rebuild the tied four-coordinate "
                           "training graph on synthetic concentration, temperature and coolant grids, "
                           "using the retained RK4 mean, timestep and substep count. "
                           "The working parameters, covariance and solver controls are frozen here."),
            "fixture_domain": ("60 CSTR transitions in float64 for performance. "
                               "Five cooling transitions in float32 and float64 for correctness."),
        },
    }
    clock = Clock(jax)
    try:
        with compile_log(jax) as events:
            clock.compiles = events
            if args.case == "oracle":
                oracle(clock, template, arrays, observations, config, report)
            else:
                trainer = setup_trainer(clock, template, arrays, observations, config, report)
                first = fit_trainer(clock, trainer, args.warm_repetitions, report)
                report["isolated_adjoint"] = adjoint_replay(clock, trainer, first, args.warm_repetitions)
        current_sources = {str(path.relative_to(ROOT)): digest(path) for path in sources}
        if capture_command("git", "rev-parse", "HEAD") != head or current_sources != frozen_sources:
            raise RuntimeError("benchmark sources changed during execution")
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["measurements"] = clock.events
        report["summary"] = summarize(clock.events)
        write_report(args.output, report)
    print(json.dumps({"case": args.case, "status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
