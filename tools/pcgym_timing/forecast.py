"""Matched development forecasts from native filtering marginals on one PCGym trajectory."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import cstr_runtime_fixture as fixture


HORIZONS = (1, 5, 15, 60)
TRUTH_SEED = 20261122
DRAW_SEED = 254901
VARIABLES = ("Ca", "T")


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _fixture_digest(arrays):
    return _digest({name: np.asarray(value).tolist() for name, value in arrays.items()})


def _commands(steps):
    tick = np.arange(steps)
    return .65 * np.sin(2 * np.pi * tick / 73) + .20 * np.cos(2 * np.pi * tick / 19)


def prepare(make_env, params, arrays, path, *, study_seed=None):
    """Run one uninterrupted noisy PCGym CSTR and save its complete, replayable truth manifest."""
    cfg = fixture.CONFIGURATION
    steps = params["N"] - 1
    if not (steps >= 599 and params["model"] == "cstr" and params["integration_method"] == "casadi"
            and params["normalise_a"] and params["normalise_o"] and params["noise"]
            and params["noise_percentage"] == .001 and not params.get("a_delta", False)
            and np.isclose(params["tsim"] / params["N"], cfg["dt"])
            and all(np.array_equal(params["o_space"][k][:2], cfg[x])
                    for k, x in (("low", "x_lo"), ("high", "x_hi")))
            and all(np.array_equal(params["a_space"][k], [cfg[u]])
                    for k, u in (("low", "u_lo"), ("high", "u_hi")))):
        raise ValueError("Expected a 599-step or longer PCGym CSTR with the pilot scales, noise and dt")
    commands = _commands(steps)
    truth_seed = TRUTH_SEED if study_seed is None else 100000 + study_seed
    draw_seed = DRAW_SEED if study_seed is None else 200000 + study_seed
    origins = np.linspace(fixture.PAST, steps - max(HORIZONS), 12, dtype=int)
    obs = np.empty((steps + 1, 2))
    truth = np.empty_like(obs)
    # PCGym draws multiplicative observation noise from NumPy's global RandomState.
    caller_rng = np.random.get_state()
    try:
        np.random.seed(truth_seed)
        env = make_env(params)
        try:
            observation, _ = env.reset(seed=truth_seed)
            obs[0], truth[0] = observation[:2], env.state[:2]
            if env.t != 0 or not (np.isfinite(obs[0]).all() and np.isfinite(truth[0]).all()):
                raise ValueError("Invalid PCGym initial time, observation or truth")
            for i, u in enumerate(commands, 1):
                observation, _, terminated, truncated, _ = env.step(np.array([u]))
                obs[i], truth[i] = observation[:2], env.state[:2]
                if env.t != i or bool(terminated) != (i == steps) or truncated:
                    raise RuntimeError(f"Unexpected PCGym time or termination at step {i}")
                if not (np.isfinite(obs[i]).all() and np.isfinite(truth[i]).all()):
                    raise ValueError(f"Nonfinite PCGym observation or truth at step {i}")
        finally:
            env.close()
    finally:
        np.random.set_state(caller_rng)

    manifest = dict(
        protocol="pcgym-cstr-open-loop-v1", split="development" if study_seed is None else "held-out",
        plant="PCGym CSTR, CasADi CVODES", truth_seed=truth_seed, resets=1, steps=int(steps), dt=cfg["dt"],
        origin_ticks=origins.tolist(), horizons=list(HORIZONS), draw_seed=draw_seed,
        initial_state=np.asarray(params["x0"]).tolist(), setpoints_Ca=list(params["SP"]["Ca"]),
        observation_noise_percentage=params["noise_percentage"],
        process_noise="Qp[:3,:3] in normalized (Ca, T, bias) coordinates",
        open_loop="0.65 sin(2 pi t / 73) + 0.20 cos(2 pi t / 19), normalized coolant",
        physical_units=["mol/L", "K"], fixture_sha256=_fixture_digest(arrays),
        controls=commands.tolist(), observations_normalized=obs.tolist(), truth_physical=truth.tolist(),
    )
    manifest["sha256"] = _digest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            json.dump(manifest, tmp, allow_nan=False)
            tmp.write("\n")
            tmp.close()
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return manifest


def _window(observations, controls, tick):
    past = fixture.PAST
    data = {f"e{i}": None for i in range(past + fixture.HORIZON + 1)}
    for i in range(past):
        j = tick - past + i
        data[f"e{i}"] = np.r_[observations[j], controls[j]]
    data[f"e{past}"] = observations[tick].copy()
    return data


@jax.jit
def _propagate(mean, chol, controls, draws, theta, q_chol):
    particles = mean + draws[0] @ chol.T

    def step(particles, inputs):
        command, noise = inputs

        def advance(state):
            augmented = jnp.concatenate((state, command[None]))
            return fixture.cstr_rk4(augmented, theta, fixture.CONFIGURATION)[:3]

        particles = jax.vmap(advance)(particles) + noise @ q_chol.T
        return particles, particles

    return jax.lax.scan(step, particles, (controls, draws[1:]))[1]


def _compile(particles):
    steps = max(HORIZONS)
    shapes = [(3,), (3, 3), (steps,), (steps + 1, particles, 3), (4,), (3, 3)]
    args = [jax.ShapeDtypeStruct(shape, np.float64) for shape in shapes]
    return _propagate.lower(*args).compile()


def _filter(adapter, data, row):
    start = perf_counter()
    try:
        result = adapter.solve(data, details=True)
        row["native_status"] = str(result["status"])
        if result["status"] != "ok":
            raise ValueError(f"Native filtering status {result['status']}: {result.get('error', '')}")
        # Host copies synchronize native work and isolate each result from subsequent adapter calls.
        mean = np.array(result["mean"], dtype=np.float64, copy=True)
        cov = np.array(result["cov"], dtype=np.float64, copy=True)
    finally:
        row["filter_s"] = perf_counter() - start

    if mean.shape != (4,) or cov.shape != (4, 4):
        raise ValueError(f"Expected the four-dimensional s4 marginal, got {mean.shape} and {cov.shape}")
    row["filter_finite"] = bool(np.isfinite(mean).all() and np.isfinite(cov).all())
    if not row["filter_finite"]:
        raise ValueError("Nonfinite s4 marginal")
    scale = float(np.max(abs(cov)))
    if scale == 0:
        raise ValueError("Zero s4 covariance")
    scaled = cov / scale
    asymmetry = float(np.max(abs(scaled - scaled.T)))
    row["covariance_relative_asymmetry"] = asymmetry
    if asymmetry > 128 * np.finfo(cov.dtype).eps:
        raise ValueError("Asymmetric s4 covariance")
    scaled = (scaled + scaled.T) / 2
    row["covariance_relative_min_eigenvalue"] = float(np.linalg.eigvalsh(scaled)[0])
    if row["covariance_relative_min_eigenvalue"] <= 0:
        raise ValueError("s4 covariance is not positive definite")
    cov = scaled * scale
    # Marginalize the inferred action. Each transition uses the exogenous command.
    chol = np.linalg.cholesky(cov[:3, :3])
    row.update(filter_proper=True, s4_mean=mean.tolist(), s4_cov=cov.tolist())
    return mean[:3], chol


def _draws(seed, tick, particles):
    rng = np.random.Generator(np.random.PCG64([seed, tick]))
    return rng.standard_normal((max(HORIZONS) + 1, particles, 3))


def _score(states, truth):
    cfg = fixture.CONFIGURATION
    low, high = (np.asarray(cfg[key]) for key in ("x_lo", "x_hi"))
    scores = {}
    for h in HORIZONS:
        physical = low + (states[h - 1, :, :2] + 1) * ((high - low) / 2)
        means = physical.mean(axis=0)
        lower, upper = np.quantile(physical, [.05, .95], axis=0)
        target = truth[h - 1]
        scores[str(h)] = {
            name: dict(mean=float(means[i]), truth=float(target[i]), error=float(means[i] - target[i]),
                       lower90=float(lower[i]), upper90=float(upper[i]), width90=float(upper[i] - lower[i]),
                       covered90=bool(lower[i] <= target[i] <= upper[i]))
            for i, name in enumerate(VARIABLES)
        }
    return scores


def _summary(rows):
    # Keep the same complete-origin denominator at every horizon and report failures alongside it.
    passed = [row["scores"] for row in rows if row["status"] == "ok"]
    summary = {}
    for h in map(str, HORIZONS):
        horizon = dict(n=len(passed))
        for name in VARIABLES:
            values = [scores[h][name] for scores in passed]
            errors = np.array([v["error"] for v in values])
            widths = np.array([v["width90"] for v in values])
            error_scale = max(abs(errors), default=0.) or 1.
            width_scale = max(widths, default=0.) or 1.
            horizon[name] = dict(
                rmse=float(error_scale * np.sqrt(np.mean((errors/error_scale) ** 2))) if values else None,
                mae=float(error_scale * np.mean(abs(errors/error_scale))) if values else None,
                coverage90=float(np.mean([v["covered90"] for v in values])) if values else None,
                mean_width90=float(width_scale * np.mean(widths/width_scale)) if values else None,
            )
        summary[h] = horizon
    return summary


def run(adapter, arrays, dataset, *, particles=512, study_seed=None):
    """Filter twelve fresh masked windows, then forecast matched posterior draws in physical units."""
    start = perf_counter()
    if isinstance(dataset, (str, Path)):
        dataset = json.loads(Path(dataset).read_text())
    if dataset["sha256"] != _digest({k: v for k, v in dataset.items() if k != "sha256"}):
        raise ValueError("Forecast manifest checksum mismatch")
    expected_split = "development" if study_seed is None else "held-out"
    if (dataset["protocol"] != "pcgym-cstr-open-loop-v1" or dataset["split"] != expected_split
            or (study_seed is not None and (dataset["truth_seed"] != 100000 + study_seed
                                            or dataset["draw_seed"] != 200000 + study_seed))
            or _fixture_digest(arrays) != dataset["fixture_sha256"]):
        raise ValueError("Forecast dataset protocol or numerical fixture mismatch")
    if not jax.config.x64_enabled or particles < 2:
        raise ValueError("Forecast propagation requires float64 JAX and at least two particles")

    timings = dict(propagation_compile_s=0.)
    controls = np.asarray(dataset["controls"])
    obs = np.asarray(dataset["observations_normalized"])
    truth = np.asarray(dataset["truth_physical"])
    theta = np.asarray(arrays["theta"], dtype=np.float64)
    q_chol = np.linalg.cholesky(np.asarray(arrays["Qp"], dtype=np.float64)[:3, :3])
    rows, marginals = [], {}
    for tick in dataset["origin_ticks"]:
        row = dict(tick=tick, native_status=None, filter_finite=None, filter_proper=None,
                   propagation_finite=None, draws_s=None, propagation_s=None)
        rows.append(row)
        try:
            marginals[tick] = _filter(adapter, _window(obs, controls, tick), row)
        except Exception as exc:
            if row["native_status"] == "ok":
                row["filter_proper"] = False
            row.update(status="filter_failed", error=f"{type(exc).__name__}: {exc}")

    if marginals:
        compile_start = perf_counter()
        try:
            executable = _compile(particles)
        except Exception as exc:
            for row in rows:
                if row["tick"] in marginals:
                    row.update(status="compile_failed", error=f"{type(exc).__name__}: {exc}")
            marginals.clear()
        finally:
            timings["propagation_compile_s"] = perf_counter() - compile_start

    lead = max(HORIZONS)
    for row in rows:
        tick = row["tick"]
        if tick not in marginals:
            continue
        try:
            draw_start = perf_counter()
            draws = _draws(dataset["draw_seed"], tick, particles)
            row["draws_s"] = perf_counter() - draw_start
            row["draws_sha256"] = hashlib.sha256(draws.astype("<f8", copy=False)).hexdigest()
            mean, chol = marginals[tick]
            propagation_start = perf_counter()
            try:
                states = np.asarray(executable(mean, chol, controls[tick:tick + lead], draws, theta, q_chol))
            finally:
                row["propagation_s"] = perf_counter() - propagation_start
            bad = ~np.isfinite(states).all(axis=-1)
            row.update(propagation_finite=not bool(bad.any()), nonfinite_particle_steps=int(bad.sum()))
            if bad.any():
                raise ValueError(f"Nonfinite forecast beginning at lead {int(np.where(bad)[0][0]) + 1}")
            row["scores"] = _score(states, truth[tick + 1:tick + lead + 1])
            # Scoring arithmetic can overflow even when the device output is finite.
            json.dumps(row["scores"], allow_nan=False)
            row["status"] = "ok"
        except Exception as exc:
            row.pop("scores", None)
            row.update(status="forecast_failed", error=f"{type(exc).__name__}: {exc}")

    passed = sum(row["status"] == "ok" for row in rows)
    summary = _summary(rows)
    timings.update(filter_total_s=sum(row["filter_s"] for row in rows),
                   draws_total_s=sum(row["draws_s"] or 0. for row in rows),
                   propagation_total_s=sum(row["propagation_s"] or 0. for row in rows),
                   batch_wall_s=perf_counter() - start)
    return dict(status="ok" if passed == len(rows) else "failed", complete=passed == len(rows),
                n_origins=len(rows), n_scored=passed,
                n_failed=len(rows) - passed, particles=particles, dataset_sha256=dataset["sha256"],
                draw_stream="NumPy PCG64([draw_seed, tick]), standard normals (61, particles, 3); first slice at s4",
                interval="marginal 5th to 95th empirical percentiles of latent states",
                physical_units=dataset["physical_units"], summary=summary, timings=timings, origins=rows)
