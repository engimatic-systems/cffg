"""Qualify public automatic Engine.run against the archived P4/H15 CPU float64 reactor.

Run with uv run --frozen python tools/qualify_directed_reactor.py --output REPORT.json.
--inventory validates the required archives without loading JAX. The measured worker
runs in a new process with an empty, disabled compilation cache. Its first action clock
starts immediately before uv launch and stops after public single-edge marginal readout and clipping.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
REASON = "directed_chain_certified"
STOCK_TICKS = (0, 5, 20, 25, 40)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def retained_paths(root):
    return (root / "cffg-cold-start/empty.json",
            root / "cffg-reactor-pilot/corrected-nine-windows.json")


def retained(root):
    stock_path, corrected_path = retained_paths(root)
    stock, corrected = read(stock_path), read(corrected_path)
    for archive in (stock, corrected):
        require([r["index"] for r in archive["cases"]] == list(range(9)), "Expected nine PR185 windows")
    require(all(c["status"] == "converged" and c["reason"] == "root_converged" and c["failure"] is None
                for c in stock["cases"]), "Missing archived coupled root")
    require(all(c["stock"]["certified"] for c in corrected["cases"])
            and corrected["stale"]["stock"]["certified"] is False, "Missing corrected reference verdict")
    return stock, corrected


def saved_windows(root):
    paired = root / "cffg-reactor-paired"
    replay = paired / "replay"
    summary = read(replay / "summary.json")
    require([s["seed"] for s in summary["stock"]] == list(range(1, 11)), "Expected ten held-out seeds")
    for study in summary["stock"]:
        seed = study["seed"]
        episode = paired / "run" / f"seed-{seed:02d}" / "directed"
        complete = read(episode / "episode.json")
        require(complete["complete"] and complete["decisions"] == 59, f"Incomplete {episode}")
        require([r["tick"] for r in study["ticks"]] == list(STOCK_TICKS), f"Unexpected stock ticks for seed {seed}")
        selected = {r["tick"]: r for r in study["ticks"]}
        for tick in range(59):
            path = episode / f"decision-{tick:02d}.json"
            payload = path.read_bytes()
            saved, sha = json.loads(payload), digest(payload)
            require(saved["tick"] == tick and saved["seed"] == seed and saved["status"] == "checked"
                    and saved["error"] is None, f"Unexpected decision identity/status in {path}")
            require(saved["masks"] == {k: v is not None for k, v in saved["data"].items()}
                    and set(saved["data"]) == {f"e{i}" for i in range(20)}, f"Incomplete DATA/masks in {path}")
            cert = stock = None
            if tick in selected:
                ref = selected[tick]
                cert = replay / ref["raw"]
                stock = read(cert)
                require(stock["tick"] == tick and stock["historical_parity"] and stock["same_state"]["certified"]
                        and stock["stock_status"] == "converged" and stock["stock_failure"] is None
                        and stock["decision_sha256"] == ref["decision_sha256"] == sha, f"Unmatched stock certificate {cert}")
            yield dict(seed=seed, tick=tick, decision=str(path), decision_sha256=sha,
                       stock_certificate=str(cert) if cert else None), saved, stock


def launch(args):
    if args.inventory:
        stock, _ = retained(args.archives)
        windows = list((info["seed"], info["tick"], cert is not None)
                       for info, _, cert in saved_windows(args.archives))
        print(json.dumps(dict(retained=len(stock["cases"]), held_out=len(windows),
                              held_out_stock=sum(w[2] for w in windows))))
        return 0
    require(args.output is not None, "Pass --output FILE for a measured run")
    with tempfile.TemporaryDirectory(prefix="cffg-reactor-cache-") as cache:
        env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu", JAX_COMPILATION_CACHE_DIR=cache,
                   JAX_ENABLE_COMPILATION_CACHE="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1",
                   XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
        command = ["uv", "run", "--frozen", "--no-sync", "python", __file__, "--worker",
                   "--archives", str(args.archives.resolve()), "--output", str(args.output.resolve())]
        env["CFFG_REACTOR_LAUNCH"] = str(perf_counter())
        return subprocess.run(command, env=env, cwd=ROOT).returncode


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", "--archive-root", type=Path, default=Path.home() / "agent-share",
                        help="Root containing cffg-cold-start, cffg-reactor-pilot, and cffg-reactor-paired")
    parser.add_argument("--output", type=Path, help="New JSON report path")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    if not args.worker:
        sys.exit(launch(args))

# In a measured worker, import the checked-out library before building the engine.
sys.path.insert(0, str(ROOT / "src"))
import jax
import numpy as np
import cffg
import cstr_runtime_fixture as fixture
import cstr_reactor_pilot as pilot

# The pilot normally binds these two references while loading the PCGym simulator.
pilot.np, pilot.fixture = np, fixture


def within(value, limit, label):
    require(np.isfinite(value) and value <= limit, f"{label} = {value} exceeds {limit}")


def scaled(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    require(actual.shape == reference.shape, "Shape mismatch")
    difference = np.abs(actual - reference) / (1 + np.abs(reference))
    require(np.isfinite(difference).all(), "Nonfinite comparison")
    return float(np.max(difference, initial=0))


def messages_error(messages, reference):
    require(len(messages) == len(reference) == 20, "Expected 20 reactor sites")
    result = {}
    for role in ("pred", "obs", "future"):
        errors = [scaled(current[role][i], ref[role][i]) for current, ref in zip(messages, reference, strict=True)
                  for i in (0, 1)]
        result[role] = max(errors)
    return result


@contextmanager
def capture(row):
    """Keep the status and completed measurements when one window fails a numerical check."""
    row["passed"] = False
    try:
        yield
    except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    else:
        row["passed"] = True


def plan(engine, binder, base, data, row):
    start = perf_counter()
    bound = binder.bind(base, data)
    result = engine.run(bound, fixture.BUDGET, tol=fixture.TOLERANCE)
    jax.block_until_ready(result.state)
    solved = perf_counter()
    row.update(status=result.status.value, reason=result.termination_reason.value, iterations=result.iterations,
               directed_attempts=result.directed_attempts, outer_map_evaluations=result.outer_map_evaluations,
               failure=str(result.failure) if result.failure else None, inner_status=result.inner_status.value,
               inner_residual=result.inner_residual, residual=result.residual, outer_residual=result.outer_residual,
               bind_run_sync_s=solved - start)
    require(result.status.value == "converged" and row["reason"] == REASON and result.failure is None,
            f"Expected a certified directed result, got {row['status']} / {row['reason']} / {row['failure']}")
    require(result.inner_status.value == "converged" and 0 <= result.residual == result.outer_residual
            and result.free_energies, "Missing directed residual, local status or final energy")
    within(result.residual, fixture.TOLERANCE, "reported message residual")
    mean, _ = engine.gaussian_marginal(result.state, f"s{fixture.PAST}")
    kelvin = float(298.5 + 3.5 * mean[3])
    require(np.isfinite(kelvin), "Nonfinite public action")
    clipped = float(np.clip(kelvin, fixture.CONFIGURATION["u_lo"], fixture.CONFIGURATION["u_hi"]))
    ready = perf_counter()
    row.update(kelvin=kelvin, clipped_kelvin=clipped, bind_to_action_s=ready - start,
               trace_FE=float(result.free_energies[-1]))
    return bound, result.state, ready


def local_certificate(engine, arrays, data, row, state):
    import benchmark_cstr_chain as chain

    post = engine.gaussian_marginals(state)
    full_action = float(298.5 + 3.5 * post[f"s{fixture.PAST}"][0][3])
    row["all_edges_coolant_error_K"] = abs(full_action - row["kelvin"])
    within(scaled((row["kelvin"] - 298.5) / 3.5, (full_action - 298.5) / 3.5),
           fixture.TOLERANCE, "single-edge/all-edge command difference")
    sites = pilot.stock_sites(engine.arena, state)
    beliefs = [post[f"s{i}"] for i in range(len(sites))]
    require(set(post) == set(engine.arena.edges), "Incomplete Gaussian marginal coverage")
    for mean, cov in post.values():
        require(np.isfinite(mean).all() and np.isfinite(cov).all(), "Nonfinite Gaussian marginal")
        np.linalg.cholesky(cov)
    energy = float(engine.bfe(state))
    row["FE"] = energy
    within(scaled(row["trace_FE"], energy), fixture.TOLERANCE, "run/returned-state FE difference")
    check = chain.check_candidate(arrays, data, sites, beliefs, energy)
    row["joint_gaussian_check"] = check
    within(scaled(row["kelvin"], check["coolant_kelvin"]), fixture.TOLERANCE, "readout/certified action")
    return sites, post


def stock_certificate(engine, bound, state, post, sites, row, old_post, old_energy, old_kelvin):
    # audit reconstructs *all* endpoints from the supplied site messages. Verify its
    # reconstructed slots against the actual return, including DATA-side predictives.
    full, _, _, _ = pilot._populate(engine, bound, sites)
    error = max(scaled(state.g_eta, full.g_eta), scaled(state.g_lam, full.g_lam))
    row["all_gaussian_slots_error"] = error
    within(error, fixture.TOLERANCE, "returned slots vs stock reconstruction")
    row["coupled_map_at_directed_state"] = audit = pilot.audit(engine, bound, sites)
    require(audit["certified"], "Stock raw/canonical successor certificate failed")
    within(scaled(audit["FE"], row["FE"]), fixture.TOLERANCE, "stock reconstructed FE")
    require(set(old_post) == {k for k in post if k.startswith("s")}, "Archived posterior coverage mismatch")
    errors = [pilot.posterior_errors(post[k], tuple(map(np.asarray, pair))) for k, pair in old_post.items()]
    row["stock_whitened_errors"] = worst = {k: float(np.max([e[k] for e in errors])) for k in ("mean", "covariance")}
    for key, error in worst.items():
        within(error, pilot.PARITY, f"whitened {key}")
    row.update(stock_FE=old_energy, stock_FE_error=scaled(row["FE"], old_energy), stock_kelvin=old_kelvin,
               stock_coolant_error_K=abs(row["kelvin"] - old_kelvin))
    within(row["stock_FE_error"], fixture.TOLERANCE, "stock FE difference")


def reference_certificate(row, sites, ref_sites, ref_energy, kelvin):
    row.update(reference_message_errors=messages_error(sites, ref_sites), reference_FE=ref_energy,
               reference_kelvin=kelvin, reference_coolant_error_K=abs(row["kelvin"] - kelvin))
    for direction, error in row["reference_message_errors"].items():
        within(error, fixture.TOLERANCE, f"reference {direction} message")
    within(scaled(row["FE"], ref_energy), fixture.TOLERANCE, "directed reference FE")
    # Compare normalized commands so clipping cannot hide an incorrect proposal.
    within(scaled((row["kelvin"] - 298.5) / 3.5, (kelvin - 298.5) / 3.5),
           fixture.TOLERANCE, "directed reference action")


def stale_control(engine, base, binder, arrays, state, post, row):
    import benchmark_cstr_chain as chain

    data = fixture.observations()
    sites = pilot.stock_sites(engine.arena, state)
    stock = pilot.audit(engine, binder.bind(base, data), sites)
    checks = dict(stock_rejected=not stock["certified"], local_rejected=False)
    try:
        chain.check_candidate(arrays, data, sites, [post[f"s{i}"] for i in range(20)], row["FE"])
    except ArithmeticError as exc:
        checks.update(local_rejected=True, local_reason=str(exc))
    return checks


def qualify_saved(engine, binder, base, arrays, root, report):
    for row, saved, cert in saved_windows(root):
        report["held_out"].append(row)
        with capture(row):
            data = {k: None if v is None else np.asarray(v) for k, v in saved["data"].items()}
            bound, state, _ = plan(engine, binder, base, data, row)
            sites, post = local_certificate(engine, arrays, data, row, state)
            row.update(archived_applied_kelvin=saved["applied_kelvin"], archived_fallback=saved["fallback"],
                       archived_deadline_miss=saved["deadline_miss"], archived_status=saved["status"])
            reference_certificate(row, sites, saved["candidate"]["messages"], saved["candidate"]["evidence_cost"],
                                  saved["proposed_kelvin"])
            if cert:
                stock_certificate(engine, bound, state, post, sites, row, cert["stock_posterior"],
                                  cert["stock_FE"], cert["stock_kelvin"])
                row.update(stock_reason=cert["stock_reason"], stock_residual=cert["stock_reported_root"],
                           stock_return_local_check_passed="stock_return_local_check" in cert,
                           stock_message_errors=messages_error(sites, cert["stock_messages"]))


def timing_summary(report):
    groups = defaultdict(list)
    for row in report["retained"]:
        groups[row["kind"]].append(row)
    groups["held_out_after_offline_audit"] = report["held_out"]
    summary = {}
    for kind, rows in groups.items():
        summary[kind] = {}
        for key in ("bind_run_sync_s", "bind_to_action_s"):
            samples = [r[key] for r in rows if key in r]
            if samples:
                median, p95 = np.percentile(samples, [50, 95])
                summary[kind][key] = dict(count=len(samples), median=median, p95=p95)
    return summary


def qualify(report, archive_root):
    require(Path(cffg.__file__).resolve().is_relative_to(ROOT / "src") and jax.config.x64_enabled
            and jax.default_backend() == "cpu" and not jax.config.jax_enable_compilation_cache,
            "Expected local library, float64 CPU and disabled compilation cache")
    start = perf_counter()
    engine, _, arrays = fixture.make_engine()
    binder, base = fixture.Binder(engine), engine.init_state()
    jax.block_until_ready(base)
    report.update(preparation_s=perf_counter() - start, objective_contract=engine.objective_contract,
                  structural_candidate=engine.directed_chain_plan is not None)
    require(report["structural_candidate"], "The auto schedule did not admit the reactor")
    require(base.g_eta.dtype == np.float64 and engine.objective_contract["approximate"]
            and engine.objective_contract["stationarity"] == "frozen Gaussian coefficients only",
            "Expected float64 adaptive frozen-surrogate contract")

    # The launch boundary excludes archive parsing and every external auditor.
    first = dict(index=0, kind="cold")
    report["retained"].append(first)
    first_data = fixture.observations()
    with capture(first):
        first_run = plan(engine, binder, base, first_data, first)
        report["uv_launch_to_first_action_s"] = first_run[-1] - float(os.environ["CFFG_REACTOR_LAUNCH"])
    stock, corrected = retained(archive_root)
    require({k: v.tolist() for k, v in arrays.items()} == stock["fixture"]["arrays"],
            "Reactor parameters differ from the archive")
    pending = []
    for case in stock["cases"]:
        if case["index"] == 0:
            data, row = first_data, first
        else:
            data = fixture.observations(case["observation_change"], history=case["synthetic_history"], goal=case["goal"])
            row = dict(index=case["index"], kind=case["kind"])
            report["retained"].append(row)
        payload = {k: None if v is None else v.tolist() for k, v in data.items()}
        require(digest(json.dumps(payload, sort_keys=True).encode()) == case["data_sha256"], "Input digest mismatch")
        row.update(data_sha256=case["data_sha256"], stock_reason=case["reason"],
                   stock_residual=case["fresh_chart_root"], stock_bind_run_sync_s=case["seconds"])
        if row.get("error"):
            continue
        with capture(row):
            bound, state, _ = first_run if case["index"] == 0 else plan(engine, binder, base, data, row)
            pending.append((case, data, bound, state, row))

    # The nine matched timing boundaries all finish before the first offline audit.
    import benchmark_cstr_chain as chain

    for case, data, bound, state, row in pending:
        with capture(row):
            sites, post = local_certificate(engine, arrays, data, row, state)
            ref_sites, energy = chain.solve(arrays, data)
            cert = corrected["cases"][case["index"]]
            reference_certificate(row, sites, ref_sites, energy, cert["coolant_kelvin"])
            within(scaled(row["FE"], cert["ledger"]), fixture.TOLERANCE, "corrected archived ledger")
            old_post = {k: (v["mean"], v["covariance"]) for k, v in case["posterior"].items()}
            stock_certificate(engine, bound, state, post, sites, row, old_post, case["FE"], case["coolant_kelvin"])
            if case["index"] == 8:
                negative = report["stale_history"] = stale_control(engine, base, binder, arrays, state, post, row)
                require(negative["local_rejected"] and negative["stock_rejected"], "Stale-history control accepted")

    qualify_saved(engine, binder, base, arrays, archive_root, report)
    report["timings"] = timing_summary(report)
    rows = report["retained"] + report["held_out"]
    failures = sum(not row["passed"] for row in rows)
    require(failures == 0, f"{failures} of {len(rows)} windows failed; see per-window status and error")


def git(*cmd):
    return subprocess.check_output(["git", "-C", ROOT, *cmd], text=True).strip()


def worker(args):
    report = dict(passed=False, retained=[], held_out=[], scope="saved input replay, fresh init_state for each decision",
                  timing="bind, default Engine.run and synchronize; through-action also reads s4 via gaussian_marginal and clips")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        try:
            qualify(report, args.archives)
            report["passed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            files = [*retained_paths(args.archives), args.archives / "cffg-reactor-paired/replay/summary.json",
                     Path(__file__), ROOT / "tools/cstr_runtime_fixture.py", ROOT / "tools/cstr_reactor_pilot.py",
                     ROOT / "tools/benchmark_cstr_chain.py", ROOT / "uv.lock"]
            sources = sorted((ROOT / "src/cffg").rglob("*.py"))
            source_hashes = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sources}
            report["provenance"] = dict(head=git("rev-parse", "HEAD"), worktree=git("status", "--short"),
                                        source_tree_sha256=digest(json.dumps(source_hashes, sort_keys=True).encode()),
                                        sha256={str(p): digest(p.read_bytes()) for p in files if p.exists()},
                                        cffg_path=cffg.__file__, jax=jax.__version__, numpy=np.__version__,
                                        python=sys.version, platform=platform.platform(),
                                        jax_cache_enabled=jax.config.jax_enable_compilation_cache)
            json.dump(pilot.serial(report), out, allow_nan=False, indent=2)
            out.write("\n")
    print(f"Passed {len(report['retained'])} retained and {len(report['held_out'])} held-out windows. {args.output}")


if __name__ == "__main__":
    worker(args)
