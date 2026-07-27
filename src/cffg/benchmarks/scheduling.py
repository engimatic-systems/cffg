"""Schedule-aware inference benchmark (cff-d56d).

Measures the models from the adversarial performance finding — the hostile
identity-dynamics HMM with only the final state observed (K=2, the recorded
protocol's shape), the ordinary fully observed K=8 HMM, and the d=4/m=2
LGSSM — with the phases the ticket requires kept separate:

- compile: template -> arena (grounding, parameter stacking)
- schedule: arena -> Engine (schedule build + executor closures, no tracing)
- cold: first status-bearing run (JIT tracing + compilation)
- kernel: the warm jitted two-pass propagation alone
- status: the warm full run() — one fused dispatch of propagation,
  certification sweep, support flags, and free-energy ledger, plus host
  status logic
- readout: posterior marginals from the converged state
- flood: the warm synchronous-flooding run of the same model (the
  pre-schedule baseline this ticket eliminates)

The reference is measured *live* on exactly matching shapes with the same
timing protocol (median of warm repetitions, block_until_ready), per the
2026-07-27 review: no ratio-derived numbers. The reference library is an
optional dependency (`uv sync --group bench`); without it the reference
column reads n/a.

Run:  uv run python -m cffg.benchmarks.scheduling [--reps 20]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from cffg.core.ir import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec


def hmm_chain(n_steps: int, k: int = 8) -> Template:
    """prior -> [trans -> eq -> obs] x T chain; DATA ports y1..yT."""
    t = Template(name=f"hmm_{n_steps}")

    def var(name: str, card: int) -> str:
        t.variables[name] = VariableSpec(name, card)
        return name

    for blob in ("d", "B", "A"):
        t.blobs[blob] = BlobRef(blob, f"blobs/{blob}.npy")
    t.factors["prior"] = FactorSpec(
        "prior", "categorical", edges={"out": var("z0", k)},
        param_blobs={"statistic": "d"},
    )
    prev = "z0"
    for i in range(1, n_steps + 1):
        t.factors[f"trans{i}"] = FactorSpec(
            f"trans{i}", "matrix_cat",
            edges={"out": var(f"za{i}", k), "in": prev},
            param_blobs={"matrix": "B"},
        )
        t.factors[f"eq{i}"] = FactorSpec(
            f"eq{i}", "equality",
            edges={"a": f"za{i}", "b": var(f"zb{i}", k), "c": var(f"zc{i}", k)},
        )
        t.factors[f"obs{i}"] = FactorSpec(
            f"obs{i}", "matrix_cat",
            edges={"out": var(f"x{i}", k), "in": f"zc{i}"},
            param_blobs={"matrix": "A"},
        )
        t.ports[f"y{i}"] = PortSpec(f"y{i}", f"x{i}", PortRole.DATA)
        prev = f"zb{i}"
    t.ports["tail"] = PortSpec("tail", prev, PortRole.GLUE)
    return t


def _median_ms(fn, reps: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times))


def _reference_hmm(k: int, b: np.ndarray, a: np.ndarray, obs, reps: int):
    """Live matched-shape HMM smoother timing, or None without the bench
    dependency group. ``obs`` is a [T] int array with -1 = missing."""
    try:
        import jax
        import jax.numpy as jnp
        from dynamax.hidden_markov_model import CategoricalHMM
    except ImportError:
        return None
    obs = np.asarray(obs)
    m_classes = a.shape[0]
    if np.any(obs < 0):
        # Missing observations: the reference library has no missing-data
        # path, so a uniform extra class stands in — a constant per-state
        # likelihood, hence posterior-identical after renormalization.
        a = np.concatenate([a, np.ones((1, k)) / m_classes], axis=0)
        a = a / a.sum(axis=0, keepdims=True)
        obs = np.where(obs < 0, m_classes, obs)
        m_classes += 1
    hmm = CategoricalHMM(k, 1, m_classes)
    params, _ = hmm.initialize(
        jax.random.PRNGKey(0),
        initial_probs=jnp.full(k, 1.0 / k),
        transition_matrix=jnp.asarray(b) + 1e-12,
        emission_probs=jnp.asarray(a.T[:, None, :]),
    )
    o = jnp.asarray(obs.reshape(-1, 1))
    smooth = jax.jit(lambda p, y: hmm.smoother(p, y))
    jax.block_until_ready(smooth(params, o).smoothed_probs)
    return _median_ms(
        lambda: jax.block_until_ready(smooth(params, o).smoothed_probs),
        max(reps, 40), warmup=20,
    )


def _reference_lgssm(f, q, h, r, m0, p0, ys, reps: int):
    try:
        import jax
        import jax.numpy as jnp
        from dynamax.linear_gaussian_ssm import LinearGaussianSSM
    except ImportError:
        return None
    lg = LinearGaussianSSM(f.shape[0], h.shape[0])
    params, _ = lg.initialize(
        jax.random.PRNGKey(0),
        initial_mean=jnp.asarray(m0), initial_covariance=jnp.asarray(p0),
        dynamics_weights=jnp.asarray(f), dynamics_covariance=jnp.asarray(q),
        emission_weights=jnp.asarray(h), emission_covariance=jnp.asarray(r),
    )
    yj = jnp.asarray(ys)
    smooth = jax.jit(lambda p, y: lg.smoother(p, y))
    jax.block_until_ready(smooth(params, yj).smoothed_means)
    return _median_ms(
        lambda: jax.block_until_ready(smooth(params, yj).smoothed_means),
        max(reps, 40), warmup=20,
    )


def bench_case(
    name: str,
    template,
    arrays: dict,
    data: dict,
    *,
    reps: int,
    flood_iters: int,
    tol: float,
    readout: str,
    ref_ms: float | None,
) -> dict:
    import jax

    from cffg.compiler import compile_template
    from cffg.engine import Engine

    out: dict = {"name": name}
    start = time.perf_counter()
    arena = compile_template(template, arrays=arrays)
    out["compile_ms"] = 1e3 * (time.perf_counter() - start)

    start = time.perf_counter()
    engine = Engine(arena)
    out["schedule_ms"] = 1e3 * (time.perf_counter() - start)
    out["schedule"] = engine.schedule.kind
    out["registerized"] = all(engine.scan_registerized) if engine.scan_registerized else False

    state = engine.bind_data(engine.init_state(), data)
    start = time.perf_counter()
    result = engine.run(state, 1, tol=tol)
    out["cold_ms"] = 1e3 * (time.perf_counter() - start)
    assert result.status.value == "converged", (name, result.status, result.residual)

    def kernel():
        jax.block_until_ready(
            engine._propagate(
                state.messages, state.g_eta, state.g_lam, state.ef_eta,
                state.g_obs, state.g_mask, state.ef_obs, state.ef_mask,
                state.params,
            )
        )

    out["kernel_ms"] = _median_ms(kernel, reps)
    out["status_ms"] = _median_ms(lambda: engine.run(state, 1, tol=tol), max(10, reps // 2))
    read = engine.gaussian_marginals if readout == "gauss" else engine.marginals
    out["readout_ms"] = _median_ms(lambda: read(result.state), reps)

    flood = Engine(arena, schedule="flood")
    fstate = flood.bind_data(flood.init_state(), data)
    fresult = flood.run(fstate, flood_iters, tol=tol)
    assert fresult.status.value == "converged", (name, fresult.status)
    out["flood_ms"] = _median_ms(
        lambda: flood.run(fstate, flood_iters, tol=tol), max(3, reps // 4)
    )
    out["flood_sweeps"] = fresult.iterations

    if ref_ms is not None:
        out["ref_ms"] = ref_ms
        out["kernel_x"] = out["kernel_ms"] / ref_ms
        out["status_x"] = out["status_ms"] / ref_ms
    return out


def _print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n== {title} ==")
    cols = (
        "name compile_ms schedule_ms cold_ms kernel_ms status_ms readout_ms "
        "flood_ms flood_sweeps ref_ms"
    ).split()
    print(" ".join(f"{c:>12}" for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "n/a")
            cells.append(f"{v:>12.4f}" if isinstance(v, float) else f"{v!s:>12}")
        print(" ".join(cells))
    for r in rows:
        if "ref_ms" in r:
            print(
                f"  {r['name']}: live matched-shape reference {r['ref_ms']:.4f} ms -> "
                f"kernel {r['kernel_x']:.1f}x, status-bearing run {r['status_x']:.1f}x "
                f"(flooding run was {r['flood_ms'] / r['ref_ms']:.0f}x)"
            )
    if len(rows) >= 2 and all("kernel_ms" in r for r in rows):
        t0, t1 = rows[0], rows[-1]
        size = t1["size"] / t0["size"]
        print(
            f"  scaling: sizes x{size:.0f} -> kernel x{t1['kernel_ms'] / t0['kernel_ms']:.1f}, "
            f"status x{t1['status_ms'] / t0['status_ms']:.1f} (linear would be x{size:.0f})"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)  # the recorded parity protocol
    from cffg.models.kalman import lgssm

    rng = np.random.default_rng(0)

    # Hostile: K=2 identity dynamics, only the final state observed — the
    # recorded protocol's exact shape (review, 2026-07-27).
    k = 2
    a_h = rng.uniform(0.1, 1.0, (k, k))
    a_h /= a_h.sum(axis=0)
    rows = []
    for n in (32, 64, 128, 256):
        obs = np.full(n, -1)
        obs[-1] = 1
        ref = _reference_hmm(k, np.eye(k), a_h, obs, args.reps)
        row = bench_case(
            f"hostile T={n}",
            hmm_chain(n, k),
            {"d": np.full(k, 1.0 / k), "B": np.eye(k), "A": a_h},
            {f"y{n}": 1},
            reps=args.reps,
            flood_iters=3 * n + 10,
            tol=1e-12,
            readout="disc",
            ref_ms=ref,
        )
        row["size"] = n
        rows.append(row)
    _print_rows("hostile identity-dynamics HMM K=2, only final state observed", rows)

    k = 8
    a_o = rng.uniform(0.1, 1.0, (k, k))
    a_o /= a_o.sum(axis=0)
    b_o = rng.uniform(0.1, 1.0, (k, k))
    b_o /= b_o.sum(axis=0)
    rows = []
    for n in (32, 64, 128, 256):
        obs = rng.integers(0, k, size=n)
        ref = _reference_hmm(k, b_o, a_o, obs, args.reps)
        row = bench_case(
            f"ordinary T={n}",
            hmm_chain(n, k),
            {"d": np.full(k, 1.0 / k), "B": b_o, "A": a_o},
            {f"y{i + 1}": int(obs[i]) for i in range(n)},
            reps=args.reps,
            flood_iters=3 * n + 10,
            tol=1e-12,
            readout="disc",
            ref_ms=ref,
        )
        row["size"] = n
        rows.append(row)
    _print_rows("ordinary K=8 HMM, fully observed", rows)

    # LGSSM at the recorded protocol's d=4/m=2 shape.
    d, m = 4, 2
    f = 0.9 * np.eye(d) + 0.05 * rng.standard_normal((d, d))
    q = 0.1 * np.eye(d)
    h = rng.standard_normal((m, d))
    r = 0.3 * np.eye(m)
    m0 = np.zeros(d)
    p0 = np.eye(d)
    rows = []
    for n in (16, 64, 256):
        ys = rng.standard_normal((n, m))
        t, arrays = lgssm(n, f, q, h, r, m0, p0)
        ref = _reference_lgssm(f, q, h, r, m0, p0, ys, args.reps)
        row = bench_case(
            f"lgssm T={n}",
            t,
            arrays,
            {f"y{i}": ys[i] for i in range(n)},
            reps=max(5, args.reps // 2),
            flood_iters=3 * n + 10,
            tol=1e-9,
            readout="gauss",
            ref_ms=ref,
        )
        row["size"] = n
        rows.append(row)
    _print_rows("LGSSM d=4/m=2", rows)


if __name__ == "__main__":
    main()
