"""Matched recent CSTR runtime and certificate comparison.

Run this same absolute script using each independent worktree's ``uv run``.
The script does not install packages, change the experiment, or shorten plans.
Example: uv run python /path/to/tools/benchmark_adaptive_runtime.py --output /tmp/baseline.json
Then: uv run python tools/benchmark_adaptive_runtime.py --compare BASE.json OPT.json --report REPORT.html
"""

import os

os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse
import hashlib
import html
import json
import logging
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cffg
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from cffg.fixed_point import RootProgram, root_context
from cffg.outer_chart import OuterArrays, build_outer_chart, chart_coordinates, outer_chart_signature
from cstr_runtime_fixture import (
    BUDGET, CONFIGURATION, HORIZON, PAST, TOLERANCE, Binder, make_engine, observations,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def executables(engine):
    """Outer solver executable variants, not provider cache misses."""
    return [sum(kernel._cache_size() for plan in engine._outer_kernel_cache.values()
                if (kernel := plan[index]) is not None)
            for index in range(len(next(iter(engine._outer_kernel_cache.values()), (None, None))))]


class CompileRecorder(logging.Handler):
    """Capture every JAX JIT compilation in the timed boundary, including helpers."""

    def __init__(self):
        super().__init__()
        self.names = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("Compiling "):
            self.names.append(message.partition(" with ")[0])


def measure(output, method):
    engine, template, arrays = make_engine(method)
    base, binder = engine.init_state(), Binder(engine)
    assert base.g_eta.dtype == jnp.float64
    fixture = dict(past=PAST, horizon=HORIZON, configuration=CONFIGURATION,
                   method=method, budget=BUDGET, tolerance=TOLERANCE, schedule="auto",
                   arrays={key: value.tolist() for key, value in arrays.items()},
                   factors=list(template.factors), reset="original init_state every plan")
    metadata = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                    cwd=str(Path.cwd()), package=str(Path(cffg.__file__).resolve()),
                    python=sys.version, jax=jax.__version__, jaxlib=jaxlib.__version__,
                    numpy=np.__version__, platform=platform.platform(), processor=platform.processor(),
                    cpu_count=os.cpu_count(),
                    hardware=(subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                              if platform.system() == "Darwin" else platform.processor()),
                    lock_sha256=hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
                    fixture_sha256=digest(fixture),
                    fixture_source_sha256=hashlib.sha256(Path(__file__).with_name("cstr_runtime_fixture.py").read_bytes()).hexdigest(),
                    environment={key: os.environ[key] for key in ("JAX_ENABLE_X64", "JAX_PLATFORMS", "XLA_FLAGS",
                                                                  "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    assert str(Path.cwd().resolve()) in metadata["package"], "wrong worktree's installed library"
    report = dict(metadata=metadata, fixture=fixture, cases=[])
    certificate_kernels = {}
    recorder = CompileRecorder()
    logger = logging.getLogger("jax")
    logger.addHandler(recorder)
    logger.propagate = False
    # No previous solution is used, even when observations change.
    scenarios = [("cold", 0, False, 0.), *[("repeated-base", 0, False, 0.)] * 3,
                 *[("changed", change, False, 0.) for change in (1, 2, 3)],
                 ("populated-history", 1, True, 1 / 3), ("history-preview-change", 2, True, 2 / 15)]
    for index, (kind, change, history, goal) in enumerate(scenarios):
        data = observations(change, history=history, goal=goal)
        for name, value in data.items():
            if value is not None:
                assert np.isfinite(value).all() and value.shape == binder.rows[name][1].shape
        data_hash = digest({key: None if value is None else value.tolist() for key, value in data.items()})
        before = executables(engine)
        recorder.names.clear()
        started = time.perf_counter()
        with jax.log_compiles(True):
            result = engine.run(binder.bind(base, data), BUDGET, tol=TOLERANCE)
            jax.block_until_ready(result.state)
        elapsed = time.perf_counter() - started
        case = dict(index=index, kind=kind, synthetic_history=history, goal=goal,
                    observation_change=change, data_sha256=data_hash, seconds=elapsed,
                    status=result.status.value, reason=result.termination_reason.value,
                    failure=None if result.failure is None else str(result.failure),
                    iterations=result.iterations, maps=result.outer_map_evaluations,
                    trials=result.trial_evaluations, linear=result.linear_iterations,
                    provider_misses=result.solver_compilations,
                    jit_compilations=list(recorder.names),
                    jit_compilation_count=len(recorder.names),
                    executables_before=before, executables_after=executables(engine),
                    reported_root=float(result.outer_residual))
        if result.status.value == "converged":
            marginals = engine.gaussian_marginals(result.state)
            posterior = {name: dict(mean=np.asarray(mean).tolist(), covariance=np.asarray(covariance).tolist())
                         for name, (mean, covariance) in marginals.items()
                         if name not in template.ports}
            for mean, covariance in (marginals[name] for name in posterior):
                assert np.isfinite(mean).all() and np.isfinite(covariance).all()
                assert np.linalg.eigvalsh(covariance).min() > 0.
            state = result.state
            chart = build_outer_chart(engine.arena, OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta),
                                      engine._outer_eliminated)
            signature = outer_chart_signature(chart)
            if signature not in certificate_kernels:
                program = RootProgram(chart, engine._iter_step, engine._reconstruct)
                certificate_kernels[signature] = jax.jit(
                    lambda value, context, program=program: program.certificate(value, context, engine._bfe))
            certificate_kernel = certificate_kernels[signature]
            values = certificate_kernel(chart_coordinates(chart, chart.base), root_context(chart, state))
            current, successor, root, inner, status, _ = jax.device_get(values)
            fe = float(engine.bfe(state))
            assert np.isfinite([current, successor, root, inner, fe]).all()
            assert float(root) <= TOLERANCE and int(status) == 0
            assert abs(float(successor - current)) <= TOLERANCE * (1. + abs(float(current)))
            assert result.failure is None and chart.proper_reference
            # Repacking and reconstructing need not be bit-identical to returned arrays.
            drift = abs(float(current) - fe) / (1. + abs(fe))
            assert drift <= TOLERANCE
            case.update(FE=fe, fresh_chart_root=float(root), inner=float(inner), local_status=int(status),
                        FE_successor=float(successor), FE_reconstructed=float(current), FE_roundtrip_drift=drift,
                        FE_trace_final=float(result.free_energies[-1]), posterior=posterior,
                        coolant_kelvin=298.5 + 3.5 * posterior["s4"]["mean"][3])
        report["cases"].append(case)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in case.items() if key not in ("posterior", "jit_compilations")}), flush=True)


def compare(paths, destination):
    baseline, optimized = (json.loads(path.read_text()) for path in paths)
    for key in ("fixture_sha256", "fixture_source_sha256", "python", "jax", "jaxlib", "numpy",
                "environment", "lock_sha256", "hardware", "cpu_count"):
        assert baseline["metadata"][key] == optimized["metadata"][key], key
    assert len(baseline["cases"]) == len(optimized["cases"]) == 9
    errors, rows = [], []
    for base, fast in zip(baseline["cases"], optimized["cases"], strict=True):
        assert base["data_sha256"] == fast["data_sha256"]
        assert base["status"] == fast["status"] == "converged"
        mean_error, covariance_error = 0., 0.
        for name, posterior in base["posterior"].items():
            other = fast["posterior"][name]
            covariance = np.array(posterior["covariance"])
            whitening = np.linalg.cholesky(covariance)
            delta = np.array(other["mean"]) - posterior["mean"]
            mean_error = max(mean_error, np.max(np.abs(np.linalg.solve(whitening, delta))))
            delta_cov = np.array(other["covariance"]) - covariance
            whitened = np.linalg.solve(whitening, np.linalg.solve(whitening, delta_cov).T).T
            covariance_error = max(covariance_error, np.max(np.abs(whitened)))
        energy_error = abs(fast["FE"] - base["FE"]) / (1. + abs(base["FE"]))
        assert mean_error <= 1e-3 and covariance_error <= 1e-3 and energy_error <= 1e-6
        errors.append(dict(index=base["index"], mean=mean_error, covariance=covariance_error,
                           energy=energy_error, coolant_kelvin=abs(fast["coolant_kelvin"] - base["coolant_kelvin"])))
        rows.append([base["index"], base["kind"], f'{base["seconds"]:.4f}', f'{fast["seconds"]:.4f}',
                     f'{base["seconds"] / fast["seconds"]:.1f}×',
                     f'{base["iterations"]} / {fast["iterations"]}', f'{base["maps"]} / {fast["maps"]}',
                     f'{base["linear"]} / {fast["linear"]}'])
    summaries = []
    for kind in ("cold", "repeated-base", "changed", "populated-history", "history-preview-change"):
        a, b = (statistics.median(case["seconds"] for case in report["cases"] if case["kind"] == kind)
                for report in (baseline, optimized))
        summaries.append([kind, f"{a:.4f} s", f"{b:.4f} s", f"{a / b:.1f}×"])
        if kind != "cold":
            assert a / b >= 100, f"warm target missed: {kind}"
    for case in optimized["cases"][1:]:
        assert case["provider_misses"] == 0
        assert case["executables_before"] == case["executables_after"]
        assert not case["jit_compilations"], "JAX recompiled inside a warm decision"

    def table(headers, data):
        row = lambda cells, tag: "<tr>" + "".join(f"<{tag}>{html.escape(str(cell))}</{tag}>" for cell in cells) + "</tr>"
        return "<table>" + row(headers, "th") + "".join(row(cells, "td") for cells in data) + "</table>"

    maximum = {key: max(item[key] for item in errors) for key in ("mean", "covariance", "energy", "coolant_kelvin")}
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><title>CFFG adaptive runtime</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.5}
table{border-collapse:collapse;margin:20px 0}td,th{padding:8px 14px;border:1px solid #ccc;text-align:right}
th{background:#eef3f8}td:first-child,th:first-child{text-align:left}code{font-size:14px}</style>
<h1>Native adaptive CSTR runtime</h1>
<p>Matched independent worktrees/environments. Full P4/H15, 19 nonlinear transitions, four RK4 substeps,
true constants, original Q/R/prior/goals, auto schedule, budget 100, tolerance 10⁻⁶.
Every plan restarts from the same original base. Timing includes bulk data binding, run and synchronization;
construction and post-run readout/certificate auditing are outside both timers.</p>
<p>The closed-loop invocation was not retained: the initial source observation and three labeled synthetic
changes and two synthetic populated-history/preview cases are used, not a claimed replay of the 59-step episode.
The 7 ms host-linearized controller is not a parity oracle.
The earlier 112–125 s report is not the denominator; these are newly measured locked environments.</p>"""
    document += table(["Workload", "Baseline median", "Optimized median", "Speedup"], summaries)
    document += "<p>Warm target ≥100× passed; 1600× remains a stretch target. Cold first-plan time includes compilation.</p>"
    document += table(["Plan", "Kind", "Baseline s", "Optimized s", "Speedup", "Iterations B/O", "Maps B/O", "Krylov B/O"], rows)
    document += """<h2>Correctness and cache checks</h2><p>All nine matched plans converged under the unchanged full objective/root certificate.
The original host acceptance check and an independent fresh-chart certificate both pass; all non-DATA edge beliefs are finite/proper.
No nonlinear coefficients are frozen. No warm-start is used. No JAX JIT compilation occurs inside any warm timed decision,
including changed values, masks, populated history and preview. Outer solver executable variants are also recorded separately.</p>"""
    document += table(["Maximum difference across all edges/plans", "Measured", "Acceptance"],
                      [["Posterior-whitened mean", f'{maximum["mean"]:.3g}', "≤ 0.001"],
                       ["Posterior-whitened covariance", f'{maximum["covariance"]:.3g}', "≤ 0.001"],
                       ["Normalized full free energy", f'{maximum["energy"]:.3g}', "≤ 0.000001"],
                       ["Coolant action (kelvin)", f'{maximum["coolant_kelvin"]:.3g}', "reported, not clipped"]])
    document += """<h2>Implementation and scope</h2><p>Batched chart conversion; static layout with dynamic, solve-fixed metrics;
bounded fully adaptive Anderson proposals with exact-domain rejection and the original trust fallback;
reuse of linearized Jacobian products. Proposals are not a global convergence or same-basin theorem.
Discrete/EF/functional routes remain on trust inference. Sigma-point startup remains a separate failure.</p>"""
    document += """<h2>Reproduction</h2><p>Use the same <code>tools/benchmark_adaptive_runtime.py</code> absolute path in both worktrees,
each after <code>uv sync --frozen</code>; pass <code>--output</code>, then <code>--compare BASE OPT --report REPORT</code>.</p>"""
    for label, report in (("Baseline", baseline), ("Optimized", optimized)):
        document += f"<h3>{label}</h3><pre>{html.escape(json.dumps(report['metadata'], indent=2))}</pre>"
    document += "<p>Raw JSON retains every posterior, observation hash, fresh certificate, operation count and actual executable count.</p></html>"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document)
    print(json.dumps(dict(summary=summaries, maximum_parity_errors=maximum), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=("linearization", "sigma_point"), default="linearization")
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.compare:
        if not args.report:
            parser.error("--compare requires --report")
        compare(args.compare, args.report)
    elif args.output:
        measure(args.output, args.method)
    else:
        parser.error("provide --output or --compare")


if __name__ == "__main__":
    main()
