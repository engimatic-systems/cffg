"""Run one seed-0 PCGym CSTR episode and replay its saved decisions through Engine.run.

From the reviewed CFFG checkout, use uv run --frozen python tools/cstr_reactor_pilot.py
check or run --output NEW_DIRECTORY. The external validation launcher also times uv
startup. 'check' only inspects dependencies, compiles the topology and checks wiring.
"""

from time import perf_counter, time

ENTRY = perf_counter()
ENTRY_EPOCH = time()

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import tomllib
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path.home() / "experiments/cffg-testing/pc-gym/00/cffg"
REVIEWED_REV = "af864a0"
CHECKER_REV = "5d07e769946cc4e05cd56b3f0b4641a0b62d6387"
PCGYM_REV = "d26cbc0690024efef488dc6a304d37654934e525"
SEED, DEADLINE = 0, 1.0
PARITY = 1e-3
REPLAY_TICKS = (0, 5, 20, 25, 40)
PROTOCOL = {
    "pilot_seed": SEED, "held_out_paired_seeds": list(range(1, 11)),
    "environment": "PCGym common.envs.ENVS['cstr'], CasADi/CVODES, native reward",
    "schedule": "Ca 0.85 / 0.90 / 0.87, 20 samples each, 59 decisions",
    "observation": "normalized noisy Ca,T; 0.001 multiplicative Gaussian noise",
    "knowledge": "nominal true constants, P4/H15, known schedule t+1..t+15 capped at 59",
    "directed": "RK4 (4 substeps), one cold two-pass Taylor solve and joint verifier each tick",
    "comparator": "validation/v2_cstr_hostlin_baseline.py, v3b host successive linearization, "
                  "frozen prior_mean, evaluation mode, own RK4 (8 substeps), 4 sweeps, shifted previous plan",
    "compute": "one fresh CPU process per arm and seed, same machine, sequential runs, x64, "
               "one BLAS thread, JAX disk cache disabled; resident code after first tick",
    "budget": "1.0 wall second including input assembly, solve, verification/readout and actuator clip; "
              "synchronous soft deadline, discard late proposal; hold last applied or initial 298.5 K",
    "clip": "map proposed kelvin to normalized action, clip to [-1,1] (295..302 K)",
    "pairing": "same reset and NumPy noise seed; alternate arm order by seed parity; "
               "comparator history and next nominal plan must account for a held action",
    "outcomes": "native return, next true Ca RMSE and MAE vs reward-index setpoint, true state and "
                "applied actuator bound violations, proposed clips, misses, failed checks and holds; "
                "launch to first checked action, cold solve and cold/warm through-clip latency (median/p95)",
    "paired_reporting": "report all seeds, incomplete episodes and coverage; per-seed directed minus "
                        "host differences and mean with paired seed bootstrap 95% interval "
                        "(10000 resamples, RNG seed 9806); no return imputation",
    "replay": "after all actions, ticks 0,5,20,25,40 plus first failure/miss/box excursion; "
              "fresh stock init, 100 iterations at 1e-6. Historical verdict: local directed check "
              "and stock map/root and successor FE on that same candidate at 1e-6, converged stock, "
              "whitened posterior <= 1e-3 and scaled FE <= 1e-6. Report both outgoing message "
              "directions, raw coolant difference and approximate stock-return recheck separately.",
}


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", root, *args], text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    """Keep diagnostic NaN/Inf distinguishable in strict JSON."""
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if hasattr(value, "tolist"):
        return serial(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def save(path, record):
    with path.open("x") as f:
        json.dump(serial(record), f, allow_nan=False)
        f.write("\n")


def load_runtime(experiment):
    global chain, fixture, np
    # Append only the simulator dependencies. Its editable nested CFFG is an older checkout.
    external_site = experiment / ".venv/lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.extend([str(experiment), str(external_site)])
    os.environ.update(JAX_ENABLE_COMPILATION_CACHE="false", MKL_NUM_THREADS="1")
    start = perf_counter()
    import benchmark_cstr_chain as chain
    import cstr_runtime_fixture as fixture
    import numpy as np
    import cffg
    import jax
    import jaxlib
    import equinox
    from common import envs
    numeric_import = perf_counter() - start
    start = perf_counter()
    import pcgym
    import casadi
    sim_import = perf_counter() - start

    def lock_versions(root):
        return {p["name"]: p["version"] for p in tomllib.loads((root / "uv.lock").read_text())["package"]}

    locked, external = lock_versions(ROOT), lock_versions(experiment)
    modules = {}
    for mod, base, versions in [
        (cffg, ROOT / "src", locked), (np, ROOT / ".venv", locked), (jax, ROOT / ".venv", locked),
        (jaxlib, ROOT / ".venv", locked), (equinox, ROOT / ".venv", locked),
        (pcgym, external_site, external), (casadi, external_site, external),
    ]:
        path = Path(mod.__file__).resolve()
        version = metadata.version(mod.__name__)
        if not path.is_relative_to(base) or version != versions[mod.__name__]:
            raise RuntimeError(f"Unexpected {mod.__name__} import: {path}, {version}")
        modules[mod.__name__] = dict(file=str(path), version=version)
    origin = json.loads(metadata.distribution("pcgym").read_text("direct_url.json"))
    if origin["vcs_info"]["commit_id"] != PCGYM_REV or Path(envs.__file__).resolve() != experiment / "common/envs.py":
        raise RuntimeError("Unexpected PCGym source")
    if not jax.config.x64_enabled or jax.default_backend() != "cpu" or jax.config.jax_enable_compilation_cache:
        raise RuntimeError("Expected float64 CPU and disabled JAX disk cache")
    params = envs.ENVS["cstr"][1]()
    cfg = fixture.CONFIGURATION
    if not (params["N"] == 60 and params["tsim"] / params["N"] == cfg["dt"]
            and np.array_equal(params["o_space"]["low"][:2], cfg["x_lo"])
            and np.array_equal(params["o_space"]["high"][:2], cfg["x_hi"])
            and np.array_equal(params["a_space"]["low"], [cfg["u_lo"]])
            and np.array_equal(params["a_space"]["high"], [cfg["u_hi"]])
            and np.array_equal(params["SP"]["Ca"], np.repeat([.85, .90, .87], 20))
            and params["normalise_o"] and params["normalise_a"] and params["noise"]
            and params["noise_percentage"] == .001 and params["integration_method"] == "casadi"):
        raise RuntimeError("External CSTR configuration differs from the pilot protocol")

    git("merge-base", "--is-ancestor", REVIEWED_REV, "HEAD")
    reference = Path(chain.__file__).resolve()
    library = Path(cffg.__file__).resolve()
    external_sources = ["uv.lock", "common/envs.py", "common/plants.py", "common/physics.py",
                        "common/physics_agent.py", "cstr/experiment_physics.py",
                        "validation/v2_cstr_hostlin_baseline.py", "validation/v2_cstr_nonlinear_chain.py"]
    info = dict(
        head=git("rev-parse", "HEAD"), reviewed_revision=git("rev-parse", REVIEWED_REV),
        library_tree=git("rev-parse", "HEAD:src/cffg"), library_revision=git("rev-parse", "HEAD", root=library.parent),
        reference_path=str(reference), reference_revision=git("log", "-1", "--format=%H", "--", reference),
        retained_reference=git("rev-parse", "265ff14"), historical_stock=git("rev-parse", "f9df9d6"), required_checker=CHECKER_REV,
        nested_checkout=git("rev-parse", "HEAD", root=experiment / "cffg"),
        modules=modules, pcgym_source=origin, python=sys.executable, python_version=sys.version,
        os=platform.platform(), cpu=platform.processor(), cpu_count=os.cpu_count(),
        hardware=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin" else platform.machine(),
        devices=[str(dev) for dev in jax.devices()], arguments=sys.argv, experiment=str(experiment),
        numerical_import_s=numeric_import, simulator_import_s=sim_import,
        environment={k: os.environ.get(k) for k in (
            "JAX_PLATFORMS", "JAX_ENABLE_X64", "JAX_ENABLE_COMPILATION_CACHE", "XLA_FLAGS",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")},
        file_sha256={str(p): sha(p) for p in [Path(__file__), reference, Path(fixture.__file__), ROOT / "uv.lock",
                                               *(experiment / s for s in external_sources)]},
    )
    launcher = os.environ.get("CSTR_PILOT_LAUNCHER")
    if launcher:
        info["file_sha256"][launcher] = sha(Path(launcher))
    return pcgym.make_env, params, info


def window(params, history, observation, tick):
    schedule = np.asarray(params["SP"]["Ca"])
    indices = np.minimum(tick + np.arange(1, fixture.HORIZON + 1), len(schedule) - 1)
    goal = schedule[indices]
    low, high = (params["o_space"][key][0] for key in ("low", "high"))
    past = history[-fixture.PAST:]
    data = {f"e{i}": None for i in range(fixture.PAST)}
    for i, (obs, u) in enumerate(past, fixture.PAST - len(past)):
        data[f"e{i}"] = np.r_[obs[:2], u]
    data[f"e{fixture.PAST}"] = observation[:2]
    for i, g in enumerate(2 * (goal - low) / (high - low) - 1, fixture.PAST + 1):
        data[f"e{i}"] = np.array([g, 0.])
    return dict(tick=tick, observation=observation, history=past, preview_indices=indices,
                preview_Ca=goal, data=data, masks={k: v is not None for k, v in data.items()})


def decide(params, arrays, history, observation, tick):
    start = perf_counter()
    inputs = window(params, history, observation, tick)
    solve_start = perf_counter()
    candidate, checks, error = {}, None, None
    solved = None
    raw = clipped = None
    last = history[-1][1] if history else 0.
    try:
        messages, cost = chain.solve(arrays, inputs["data"])
        candidate.update(messages=messages, evidence_cost=cost)
        beliefs = [chain.posterior(site) for site in messages]
        candidate["beliefs"] = beliefs
        solved = perf_counter()
        checks = chain.check_candidate(arrays, inputs["data"], messages, beliefs, cost)
        # The verifier is the authority for the command.
        raw = (checks["coolant_kelvin"] - 298.5) / 3.5
        if not np.isfinite(raw):
            raise ArithmeticError("Nonfinite checked coolant")
        clipped = float(np.clip(raw, -1., 1.))
    except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    ready = perf_counter()
    late = ready - start > DEADLINE
    fallback = error is not None or late
    u = last if fallback else clipped
    command = np.array([u])
    applied = perf_counter()
    # Include construction of the final env.step argument in the deadline decision.
    if not fallback and applied - start > DEADLINE:
        late = fallback = True
        command[0] = u = last
        applied = perf_counter()
    return dict(**inputs, candidate=candidate, checks=checks, error=error,
                status="verification_failed" if error else "checked", deadline_miss=late or applied - start > DEADLINE,
                fallback=fallback, proposed_kelvin=checks["coolant_kelvin"] if checks else None,
                proposed_normalized=raw, clipped_normalized=clipped,
                applied_normalized=u, applied_kelvin=298.5 + 3.5 * u,
                solve_s=None if solved is None else solved - solve_start, through_clip_s=ready - start,
                observation_to_action_s=applied - start,
                applied_at=applied, checked_at=None if error else ready), command


def stock_sites(arena, state):
    """Read outgoing Gaussian messages from the compiled factor/role scatter endpoints."""
    slots = {(name, role): int(g.scatter[row, col]) for g in arena.gauss_groups
             for row, name in enumerate(g.members) for col, role in enumerate(g.roles)}
    eta, precision = np.asarray(state.g_eta), np.asarray(state.g_lam)

    def outgoing(name, role, edge):
        slot = slots[name, role]
        e = arena.edges[edge]
        if e.dim != 4 or slot not in (e.slot0, e.slot1):
            raise ValueError(f"Unexpected endpoint {name}.{role} at {edge}")
        return eta[arena.g_eta_indices(slot, 4)], precision[arena.g_lam_indices(slot, 4)]

    last = fixture.PAST + fixture.HORIZON
    sites = []
    for i in range(last + 1):
        s = f"s{i}"
        pred = outgoing(f"tr{i}" if i else "prior", "out", s)
        obs = outgoing(f"em{i}", "in", s + "o" if i < last else s)
        future = (outgoing(f"tr{i+1}", "in", s + "c") if i < last
                  else (np.zeros(4), np.zeros((4, 4))))
        sites.append(dict(pred=pred, obs=obs, future=future))
    return sites


def relative(value, ref):
    return float(np.max(np.abs(np.asarray(value) - ref) / (1 + np.abs(ref))))


def posterior_errors(belief, stock):
    mean, cov = belief
    center, reference = stock
    chol = np.linalg.cholesky(reference)
    d = np.linalg.solve(chol, cov - reference)
    return dict(mean=np.max(np.abs(np.linalg.solve(chol, mean - center))),
                covariance=np.max(np.abs(np.linalg.solve(chol, d.T))))


# Retained engine_audit.py from the cold-start study, with imports deferred for runtime pinning.
def _pair(row, key, t, dtype):
    eta, precision = (np.asarray(x, dtype=dtype) for x in row[key])
    if eta.shape != (4,) or precision.shape != (4, 4):
        raise ValueError(f"messages[{t}][{key!r}] needs shapes (4,) and (4, 4)")
    if not (np.isfinite(eta).all() and np.isfinite(precision).all()):
        raise ValueError(f"nonfinite natural parameters in messages[{t}][{key!r}]")
    return eta, precision


def _add(left, right):
    return left[0] + right[0], left[1] + right[1]


def _relative(actual, reference):
    return float(np.max(np.abs(actual - reference) / (1. + np.abs(reference)), initial=0.))


def _populate(engine, base, messages):
    import jax.numpy as jnp
    from cffg.outer_chart import OuterArrays

    arena = engine.arena
    if len(messages) != 20 or engine._reconstruct is None:
        raise ValueError("expected 20 timesteps and the PR185 mixed forest schedule")
    addresses = {
        (member, role): (int(group.scatter[row, col]), int(group.gather[row, col]))
        for group in arena.gauss_groups for row, member in enumerate(group.members)
        for col, role in enumerate(group.roles)
    }
    eta = np.zeros_like(np.asarray(base.g_eta))
    lam = np.zeros_like(np.asarray(base.g_lam))
    slots = set()
    transitions = set()
    predictive = set()
    eta_reference = np.ones(eta.shape, dtype=bool)
    lam_reference = np.ones(lam.shape, dtype=bool)
    symmetry = 0.

    def put(factor, role, name, pair):
        edge = arena.edges[name]
        own, opposite = addresses[factor, role]
        if {own, opposite} != {edge.slot0, edge.slot1} or own in slots:
            raise ValueError(f"unexpected compiled endpoints at {factor}.{role} on {name}")
        slots.add(own)
        ei = arena.g_eta_indices(own, edge.dim)
        li = arena.g_lam_indices(own, edge.dim)
        if pair is None:
            # DATA external side is vacuous even when bound. The em.out
            # predictive is the only side calculated solely by the forest.
            if edge.interior or opposite != edge.slot1 or opposite in slots:
                raise ValueError(f"expected a DATA half-edge at {name}")
            slots.add(opposite)
            predictive.add(own)
            eta_reference[ei] = False
            lam_reference[li] = False
        else:
            if edge.dim != 4 or not edge.interior:
                raise ValueError(f"expected a four-dimensional interior edge at {name}")
            eta[ei], lam[li] = pair
        return own

    for t, row in enumerate(messages):
        pred, obs, future = (_pair(row, key, t, eta.dtype) for key in ("pred", "obs", "future"))
        for _, precision in (pred, obs, future):
            asymmetry = np.max(np.abs(precision - precision.T))
            symmetry = max(symmetry, float(asymmetry / (1. + np.max(np.abs(precision)))))
        state_edge = f"s{t}"
        if t == 0:
            put("prior", "out", state_edge, pred)
        else:
            transitions.add(put(f"tr{t}", "out", state_edge, pred))
        emission_edge = state_edge
        if t < 19:
            emission_edge = f"s{t}o"
            put(f"eq{t}", "a", state_edge, _add(obs, future))
            put(f"eq{t}", "b", f"s{t}c", _add(pred, obs))
            put(f"eq{t}", "c", emission_edge, _add(pred, future))
            transitions.add(put(f"tr{t+1}", "in", f"s{t}c", future))
        elif any(np.any(x) for x in future):
            raise ValueError("future at terminal s19 must be vacuous")
        put(f"em{t}", "in", emission_edge, obs)
        put(f"em{t}", "out", f"e{t}", None)

    eliminated = engine._outer_eliminated["gaussian"]
    if (slots != set(range(arena.n_gslots)) or slots - eliminated != transitions
            or len(transitions) != 38 or len(predictive) != 20):
        raise ValueError("compiled slot coverage or forest elimination differs from the P4/H15 chain")

    values = engine._reconstruct(
        base.messages, jnp.asarray(eta), jnp.asarray(lam), base.ef_eta,
        base.g_obs, base.g_mask, base.ef_obs, base.ef_mask, base.params,
    )
    arrays = OuterArrays(*values[:4])
    # A wrong supplied prior, likelihood or equality cavity must not disappear
    # when the forest recomputes it from the actual parameters and bound DATA.
    error = max(_relative(np.asarray(arrays.g_eta)[eta_reference], eta[eta_reference]),
                _relative(np.asarray(arrays.g_lam)[lam_reference], lam[lam_reference]))
    return arrays, values[4], error, symmetry


def _state(base, arrays):
    from cffg.runtime_types import _replace
    return _replace(base, **arrays._asdict())


def _fact(fact):
    return {key: int(value) for key, value in fact._asdict().items()}


def audit(engine, base_with_bound_data, messages, *, tol=1e-6):
    """Return stock map/energy measurements and a combined certificate.

    FE is the assembled state's engine.bfe. FE_current and root use a fresh
    chart round trip, as in the PR185 benchmark. FE_successor is the canonical
    successor; FE_raw_successor uses reconstruction directly on the map output.
    host_* independently checks the original assembled point before chart decoding.
    Each scaled FE change is absolute change divided by 1 + abs(current FE).
    """
    import jax.numpy as jnp
    from cffg.fixed_point import RootProgram, root_context
    from cffg.outer_chart import build_outer_chart, chart_arrays, chart_coordinates

    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    base = base_with_bound_data
    arrays, assembly_fact, assembly_error, symmetry = _populate(engine, base, messages)
    state = _state(base, arrays)
    facts = {"assembly": _fact(assembly_fact)}
    report = dict(certified=False, tol=float(tol), facts=facts,
                  assembly_error=assembly_error, precision_asymmetry=symmetry)
    if not engine._state_is_finite(state):
        report["finite"] = False
        return report
    chart = build_outer_chart(engine.arena, arrays, engine._outer_eliminated)
    ctx = root_context(chart, state)
    program = RootProgram(chart, engine._iter_step, engine._reconstruct)
    x = chart_coordinates(chart, arrays, ctx.metrics)

    def reconstructed(value, label):
        result, fact = program.reconstruct_arrays(value, ctx)
        facts[label] = _fact(fact)
        return result

    def decode(coords, label):
        return reconstructed(chart_arrays(chart, coords, ctx.base, ctx.metrics), label)

    def mapped(value, label):
        result, inner, status, fact = program.map_arrays(value, ctx)
        facts[label] = _fact(fact)
        y = chart_coordinates(chart, result, ctx.metrics)
        root = float(jnp.max(jnp.abs(y - x) / (1. + jnp.abs(x)), initial=0.))
        return result, y, root, float(inner), int(status)

    def energy(value):
        return float(engine.bfe(_state(base, value)))

    # The stock host map reconstructs before its sweep and before each FE.
    host_current = reconstructed(arrays, "host_current")
    host_map, _, host_root, host_inner, host_status = mapped(host_current, "host_map")
    host_next = reconstructed(host_map, "host_successor")
    host_fe, host_next_fe = energy(host_current), energy(host_next)

    current = decode(x, "current")
    raw, y, root, inner, status = mapped(current, "map")
    raw_next = reconstructed(raw, "raw_successor")
    canonical = decode(y, "canonical_successor")
    fe, current_fe = energy(arrays), energy(current)
    raw_fe, next_fe = energy(raw_next), energy(canonical)

    marginals = engine.gaussian_marginals(state)
    min_eig = np.inf
    proper = 0
    for mean, cov in marginals.values():
        eig = (float(np.linalg.eigvalsh(cov).min())
               if np.isfinite(mean).all() and np.isfinite(cov).all() else -np.inf)
        min_eig = min(min_eig, eig)
        proper += int(eig > 0.)
    all_points = (arrays, host_current, host_map, host_next, current, raw, raw_next, canonical)
    finite = all(np.isfinite(np.asarray(v)).all() for point in all_points for v in point)
    feasible = all(not engine._has_zero_support(_state(base, point)) for point in all_points)
    changes = dict(
        FE_roundtrip_drift=abs(current_fe - fe) / (1. + abs(fe)),
        raw_FE_change=abs(raw_fe - current_fe) / (1. + abs(current_fe)),
        canonical_FE_change=abs(next_fe - current_fe) / (1. + abs(current_fe)),
        host_FE_change=abs(host_next_fe - host_fe) / (1. + abs(host_fe)),
    )
    plan = engine._outer_kernel_plan(chart, state, [])
    merit, gradient_norm = map(float, plan.root_diagnostics(x, jnp.ones_like(x), plan.context))
    report.update(merit=merit, gradient_norm=gradient_norm)
    scalars = (merit, gradient_norm, fe, current_fe, raw_fe, next_fe, host_fe, host_next_fe,
               root, inner, host_root, host_inner, *changes.values())
    finite = bool(finite and np.isfinite(scalars).all())
    passed = (finite and feasible and chart.proper_reference
              and all(f["reason"] == 0 for f in facts.values())
              and status == host_status == 0 and proper == len(engine.arena.edges)
              and max(root, host_root, assembly_error, symmetry, *changes.values()) <= tol)
    report.update(certified=bool(passed), root=root, FE=fe, FE_current=current_fe,
                  FE_successor=next_fe, FE_raw_successor=raw_fe, inner=inner, status=status,
                  host_root=host_root, host_inner=host_inner, host_status=host_status,
                  host_FE=host_fe, host_FE_successor=host_next_fe, finite=finite,
                  feasible=bool(feasible), proper_reference=bool(chart.proper_reference),
                  marginal_count=len(marginals), proper_marginals=proper,
                  min_covariance_eigenvalue=float(min_eig), **changes)
    return report


def replay(engine, binder, base, arrays, record, report):
    data = {k: None if v is None else np.asarray(v) for k, v in record["data"].items()}
    bound = binder.bind(base, data)
    candidate = record["candidate"]
    if "messages" in candidate:
        try:
            report["same_state"] = audit(engine, bound, candidate["messages"])
        except Exception:
            report["same_state_error"] = traceback.format_exc()
    stock = engine.run(bound, fixture.BUDGET, tol=fixture.TOLERANCE)
    report.update(stock_status=stock.status.value, stock_reason=stock.termination_reason.value,
                  stock_failure=str(stock.failure) if stock.failure else None, iterations=stock.iterations,
                  stock_reported_root=stock.outer_residual, stock_reported_residual=stock.residual)
    if stock.status.value != "converged" or stock.failure:
        return
    state = stock.state
    post = {k: pair for k, pair in engine.gaussian_marginals(state).items() if k.startswith("s")}
    fe = float(engine.bfe(state))
    stock_kelvin = 298.5 + 3.5 * post[f"s{fixture.PAST}"][0][3]
    report.update(stock_posterior=post, stock_FE=fe, stock_trace_FE=float(stock.free_energies[-1]),
                  stock_kelvin=stock_kelvin)
    checks = record["checks"]
    if checks and record["error"] is None:
        post_errors = {k: posterior_errors(candidate["beliefs"][int(k[1:].rstrip("co"))], pair)
                       for k, pair in post.items()}
        energy_error = relative(candidate["evidence_cost"], fe)
        report.update(posterior_errors=post_errors, FE_error=energy_error,
                      directed_residual=checks["local_message_residual"],
                      coolant_error_K=abs(checks["coolant_kelvin"] - stock_kelvin))
        report["historical_parity"] = bool(report.get("same_state", {}).get("certified")
                                           and energy_error <= fixture.TOLERANCE
                                           and all(v <= PARITY for pair in post_errors.values()
                                                   for v in pair.values()))
    # These diagnostic readouts have no historical acceptance limit.
    try:
        stock_messages = stock_sites(engine.arena, state)
        report["stock_messages"] = stock_messages
        if "messages" in candidate:
            errors = {k: [] for k in ("pred", "obs", "future")}
            for fast, original in zip(candidate["messages"], stock_messages, strict=True):
                for role in errors:
                    errors[role].append(dict(eta=relative(fast[role][0], original[role][0]),
                                             precision=relative(fast[role][1], original[role][1])))
            report["message_errors"] = errors
    except Exception:
        report["message_readout_error"] = traceback.format_exc()
        return
    try:
        beliefs = [post[f"s{i}"] for i in range(len(stock_messages))]
        report["stock_return_local_check"] = chain.check_candidate(arrays, data, stock_messages, beliefs, fe)
    except Exception:
        report["stock_return_local_check_error"] = traceback.format_exc()


def negative_controls(engine, binder, base, arrays, record):
    """A changed preview and a removed history row must each invalidate the original messages."""
    candidate = record["candidate"]
    messages = [{k: tuple(map(np.asarray, pair)) for k, pair in site.items()} for site in candidate["messages"]]
    beliefs = [tuple(map(np.asarray, pair)) for pair in candidate["beliefs"]]
    original = {k: None if v is None else np.asarray(v) for k, v in record["data"].items()}
    goal = f"e{fixture.PAST + fixture.HORIZON}"
    results = {}
    for label, edits in [("history_mask", {"e0": None}), ("preview", {goal: original[f"e{fixture.PAST + fixture.HORIZON - 1}"]})]:
        changed = original | edits
        row = results[label] = dict(edits=edits)
        try:
            chain.check_candidate(arrays, changed, messages, beliefs, candidate["evidence_cost"])
            row["local_rejected"] = False
        except ArithmeticError as exc:
            row.update(local_rejected=True, local_reason=str(exc))
        try:
            stock = audit(engine, binder.bind(base, changed), messages)
            row.update(same_state=stock, stock_rejected=not stock["certified"])
        except Exception:
            row.update(stock_rejected=False, audit_error=traceback.format_exc())
    altered = beliefs.copy()
    m, c = beliefs[fixture.PAST]
    altered[fixture.PAST] = (m + [0., 0., 0., 4.], c)
    try:
        chain.check_candidate(arrays, original, messages, altered, candidate["evidence_cost"])
        results["altered_action_belief"] = dict(local_rejected=False)
    except ArithmeticError as exc:
        results["altered_action_belief"] = dict(local_rejected=True, local_reason=str(exc))
    return results


def stock_replays(out, arrays, selected, episode_ticks, *, decisions_dir=None, stock_runtime=None):
    results = {}
    diagnostic_errors = {}
    failures = []
    controls_ok = False
    if stock_runtime is None:
        engine, _, _ = fixture.make_engine()
        stock_runtime = engine, fixture.Binder(engine), engine.init_state()
    engine, binder, base = stock_runtime
    for tick in sorted(selected):
        record = None
        result = dict(tick=tick, historical_parity=False)
        result_path = out / f"stock-{tick:02}.json"
        if tick >= episode_ticks:
            result["missing_decision"] = True
        else:
            decision = (decisions_dir or out) / f"decision-{tick:02}.json"
            record = json.loads(decision.read_text())
            result["decision_sha256"] = sha(decision)
            try:
                replay(engine, binder, base, arrays, record, result)
            except Exception:
                result["replay_error"] = traceback.format_exc()
        save(result_path, result)
        diagnostic_errors[tick] = [key for key in ("message_readout_error", "stock_return_local_check_error")
                                   if key in result]
        results[tick] = result["historical_parity"]
        if not result["historical_parity"]:
            failures.append(tick)
        if tick == 5:
            control = dict(tick=tick, passed=False)
            if record and result.get("same_state", {}).get("certified") and record["checks"] and not record["error"]:
                try:
                    tests = negative_controls(engine, binder, base, arrays, record)
                    controls_ok = all(v["local_rejected"] and v.get("stock_rejected", True) for v in tests.values())
                    control.update(cases=tests, passed=controls_ok)
                except Exception:
                    control["error"] = traceback.format_exc()
            else:
                control["unavailable"] = "Original window failed the positive control"
            save(out / "negative-controls.json", control)
    return dict(historical_parity_by_tick=results, first_stock_failure=min(failures, default=None),
                all_historical_parity=all(results.values()), diagnostic_errors=diagnostic_errors,
                negative_controls_passed=controls_ok)


def episode(out, make_env, params, arrays, launch, *, seed=SEED, controller=decide):
    start = perf_counter()
    env = make_env(params)
    np.random.seed(seed)  # PCGym draws its multiplicative observation noise from the global stream.
    obs, _ = env.reset(seed=seed)
    reset_s = perf_counter() - start
    history, errors, rewards, warm_times = [], [], [], []
    first = {}
    counts = dict(misses=0, holds=0, verification_failures=0, clips=0, state_box_violations=0,
                  applied_actuator_violations=0, applied_saturations=0)
    timings = dict(env_build_reset_s=reset_s, launch_to_first_checked_s=None, launch_to_first_applied_s=None)
    complete = False

    def remember(label, tick, record):
        if label not in first:
            first[label] = tick
            save(out / f"first-{label}.json", record)

    decisions = 0
    for tick in range(params["N"] - 1):
        record, command = controller(params, arrays, history, obs, tick)
        # Hand the selected array directly to PCGym before serializing any evidence.
        called = perf_counter()
        step_error = None
        try:
            plant_output = env.step(command)
        except Exception:
            step_error = traceback.format_exc()
        checked = record.pop("checked_at")
        applied = record.pop("applied_at")
        record["observation_to_plant_call_s"] = record["observation_to_action_s"] + (called - applied)
        if tick == 0:
            timings.update(launch_to_first_plant_call_s=called - launch, launch_to_first_applied_s=applied - launch, cold_solve_s=record["solve_s"],
                           cold_observation_to_action_s=record["observation_to_action_s"])
        else:
            warm_times.append(record["observation_to_action_s"])
        if checked is not None and timings["launch_to_first_checked_s"] is None:
            timings.update(launch_to_first_checked_s=checked - launch, first_checked_tick=tick)
        record["seed"] = seed
        save(out / f"decision-{tick:02}.json", record)
        decisions += 1
        u = float(command[0])
        history.append((obs.copy(), u))
        counts["misses"] += int(record["deadline_miss"])
        counts["holds"] += int(record["fallback"])
        counts["verification_failures"] += int(record["error"] is not None)
        counts["clips"] += int(record["clipped_normalized"] is not None
                               and record["clipped_normalized"] != record["proposed_normalized"])
        counts["applied_saturations"] += int(abs(u) == 1.)
        counts["applied_actuator_violations"] += int(not -1 <= u <= 1)
        for flag, label in [(record["error"], "verification-failure"), (record["deadline_miss"], "miss"),
                            (np.any(np.abs(obs[:2]) > 1), "input-box-excursion")]:
            if flag:
                remember(label, tick, record)
        if record["fallback"]:
            remember("failure", tick, record)
        if step_error is not None:
            failure = dict(decision=record, plant_error=step_error)
            remember("plant-failure", tick, failure)
            remember("failure", tick, failure)
            break
        try:
            obs, reward, terminated, truncated, _ = plant_output
            state = np.array(env.state, copy=True)
            target = params["SP"]["Ca"][env.t]
            signed_error = float(state[0] - target)
            reward = float(np.asarray(reward).item())
            violation = bool(np.any(state[:2] < params["o_space"]["low"][:2])
                             or np.any(state[:2] > params["o_space"]["high"][:2]))
            step = dict(tick=tick, reward=reward, next_observation=obs, next_true_state=state,
                        reward_setpoint_index=env.t, reward_setpoint_Ca=target, Ca_error=signed_error,
                        state_box_violation=violation, terminated=terminated, truncated=truncated)
            save(out / f"step-{tick:02}.json", step)
            if not np.isfinite(np.r_[obs, state, reward]).all() or env.t != tick + 1:
                raise ValueError("Invalid plant output or setpoint index")
            counts["state_box_violations"] += int(violation)
            if violation:
                remember("state-box-excursion", tick, dict(decision=record, step=step))
            errors.append(signed_error)
            rewards.append(reward)
            if terminated or truncated:
                complete = bool(terminated and not truncated and tick == params["N"] - 2)
                if not complete:
                    raise ValueError("Unexpected early episode termination or truncation")
                break
        except Exception:
            failure = dict(decision=record, plant_error=traceback.format_exc())
            remember("plant-failure", tick, failure)
            remember("failure", tick, failure)
            break
    if warm_times:
        timings.update(warm_median_s=np.median(warm_times), warm_p95_s=np.percentile(warm_times, 95))
    return dict(complete=complete, decisions=decisions, scored_steps=len(rewards), first=first, counts=counts,
                return_sum=sum(rewards), Ca_RMSE=np.sqrt(np.mean(np.square(errors))) if errors else None,
                Ca_MAE=np.mean(np.abs(errors)) if errors else None, timings=timings)


def self_check(params):
    from types import SimpleNamespace
    from unittest.mock import patch
    obs = np.array([.1, .2, .3])
    history = [(obs, .5)] * fixture.PAST
    a = window(params, [], obs, 0)
    b = window(params, history, obs, 5)
    end = window(params, history, obs, 58)
    assert sum(a["masks"].values()) == 16 and all(b["masks"].values())
    assert np.array_equal(b["data"]["e0"], [.1, .2, .5]) and np.array_equal(b["data"]["e4"], obs[:2])
    assert list(b["preview_Ca"][-2:]) == [.85, .90] and np.all(end["preview_indices"] == 59)
    # Exercise readout, clip, failed check and late hold without asking for a physical prediction.
    with (patch.object(chain, "solve", return_value=([], 0.)), patch.object(chain, "check_candidate") as checker):
        checker.return_value = {"coolant_kelvin": 305.5}
        record, u = decide(params, {}, history, obs, 5)
        assert record["proposed_normalized"] == 2 and u.item() == 1 and not record["fallback"]
        with patch(__name__ + ".DEADLINE", 0.):
            record, u = decide(params, {}, history, obs, 5)
            assert record["deadline_miss"] and record["fallback"] and u.item() == .5
        checker.side_effect = ArithmeticError("test rejection")
        record, u = decide(params, {}, [], obs, 0)
        assert record["error"] and u.item() == 0. and record["checks"] is None
    # Confirm scatter addressing against the actual compiled topology using distinguishable slot values.
    import cffg
    template, arrays = fixture.model()
    arena = cffg.compile_template(template, arrays=arrays)
    state = SimpleNamespace(g_eta=np.arange(arena.g_eta_starts[-1]), g_lam=np.arange(arena.g_lam_starts[-1]))
    sites = stock_sites(arena, state)
    sends = [site[k][0][0] for site in sites for k in ("obs", "pred")]
    sends.extend(site["future"][0][0] for site in sites[:-1])
    assert len(set(sends)) == 59 and len(sites) == 20

    # An approximate stock-state readout may fail the tighter local checker despite parity.
    from unittest.mock import Mock
    engine, binder = Mock(), Mock()
    engine.arena = arena
    post = [(np.zeros(4), np.eye(4)) for _ in sites]
    engine.gaussian_marginals.return_value = {f"s{i}": pair for i, pair in enumerate(post)}
    engine.bfe.return_value = 1.
    engine.run.return_value = SimpleNamespace(status=SimpleNamespace(value="converged"),
                                              termination_reason=SimpleNamespace(value="mock"),
                                              failure=None, iterations=1, outer_residual=0., residual=0.,
                                              state=state, free_energies=[1.])
    different = []
    for site in sites:
        eta, precision = site["future"]
        different.append({**site, "future": (eta + 1, precision)})
    record = dict(data=b["data"], candidate=dict(messages=different, beliefs=post, evidence_cost=1.),
                  checks=dict(coolant_kelvin=298.5, local_message_residual=0.), error=None)
    with (patch(__name__ + ".audit", return_value={"certified": True}) as certify,
          patch.object(chain, "check_candidate", side_effect=ArithmeticError("mock stock recheck"))):
        result = {}
        replay(engine, binder, None, arrays, record, result)
        assert result["historical_parity"] and result["stock_return_local_check_error"]
        assert result["message_errors"]["future"][-1]["eta"] == 1.
        certify.return_value = {"certified": False}
        replay(engine, binder, None, arrays, record, result)
        assert not result["historical_parity"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "run"])
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, help="new directory for one run")
    args = parser.parse_args()
    if (args.mode == "run") != (args.output is not None):
        parser.error("run requires --output NEW_DIRECTORY; check takes no output directory")
    out = args.output.resolve() if args.output else None
    if out:
        out.mkdir(parents=True, exist_ok=False)
    epoch = float(os.environ.get("CSTR_PILOT_LAUNCH_EPOCH", ENTRY_EPOCH))
    launch = ENTRY - (ENTRY_EPOCH - epoch)
    if sys.flags.optimize and args.mode == "check":
        parser.error("preflight assertions require a Python interpreter without -O")
    make_env, params, info = load_runtime(args.experiment.resolve())
    if args.mode == "check":
        self_check(params)
        print(json.dumps(dict(preflight="passed", head=info["head"], modules=info["modules"]), indent=2))
        return 0
    dirty = git("status", "--porcelain", "--", "src", "uv.lock", "tools/cstr_reactor_pilot.py",
                "tools/benchmark_cstr_chain.py", "tools/cstr_runtime_fixture.py")
    if dirty:
        raise RuntimeError(f"Pilot runtime sources differ from the recorded revision: {dirty}")
    _, arrays = fixture.model()
    save(out / "manifest.json", dict(protocol=PROTOCOL, parameters=params, model=arrays,
                                     configuration=fixture.CONFIGURATION, runtime=info, launch_epoch=epoch,
                                     launch_boundary="before uv" if "CSTR_PILOT_LAUNCH_EPOCH" in os.environ
                                     else "driver entry", pre_env_s=perf_counter() - launch))
    summary = episode(out, make_env, params, arrays, launch)
    save(out / "episode.json", summary)
    selected = set(REPLAY_TICKS).union(summary["first"].values())
    try:
        comparison = stock_replays(out, arrays, selected, summary["decisions"])
    except Exception:
        comparison = dict(all_historical_parity=False, negative_controls_passed=False,
                          replay_setup_error=traceback.format_exc())
    save(out / "replay.json", comparison)
    print(json.dumps(serial(dict(directory=str(out), episode=summary, replay=comparison)), indent=2))
    return 0 if summary["complete"] and comparison["all_historical_parity"] and comparison["negative_controls_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
