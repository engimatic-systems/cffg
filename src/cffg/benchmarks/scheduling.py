"""Schedule-aware inference benchmark (cff-d56d).

Measures the models from the adversarial performance finding — the ordinary
K=8 HMM, the hostile identity-dynamics HMM with only the final state
observed, and the 2-state LGSSM — with the phases separated as the ticket
requires:

- compile: template -> arena (grounding, parameter stacking)
- schedule: arena -> Engine (schedule build + executor closures, no tracing)
- cold: first status-bearing run (JIT tracing + compilation)
- kernel: the warm jitted two-pass propagation alone
- status: the warm full run() — propagation + certification sweep +
  support/finiteness checks + free-energy evaluation
- readout: posterior marginals from the converged state
- flood: the warm synchronous-flooding run of the same model (the pre-
  schedule baseline this ticket eliminates)

Reference points, recorded 2026-07-10 on this benchmark's M2 Max CPU/x64
protocol against a dedicated SSM library (ticket cff-d56d): hostile HMM
T=256 warm 0.0198 ms (flooding was 4,206x slower at 83.24 ms; 623x at
T=32); ordinary HMM 315-399x; LGSSM 74x at T=16 and 137x at T=64. Where an
absolute reference is not recorded, a derived reference = measured flooding
time / recorded ratio is printed and labelled as such.

Run:  uv run python -m cffg.benchmarks.scheduling [--reps 20]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from cffg.core.ir import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec

# Recorded reference/ratio figures from the 2026-07-10 parity benchmark.
HOSTILE_REF_MS = {256: 0.0198}  # absolute reference, warm, ms
HOSTILE_FLOOD_RATIO = {32: 623.0, 256: 4206.0}
ORDINARY_FLOOD_RATIO = {256: 399.0}  # "315-399x": conservative end
LGSSM_FLOOD_RATIO = {16: 74.0, 64: 137.0}


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


def _median_ms(fn, reps: int) -> float:
    fn()  # warm the caches outside the timed region
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times))


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
    ref_ms: float | None = None,
    flood_ratio: float | None = None,
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
                state.messages, state.g_eta, state.g_lam,
                state.g_obs, state.g_mask, state.params,
            )
        )

    out["kernel_ms"] = _median_ms(kernel, reps)
    out["status_ms"] = _median_ms(lambda: engine.run(state, 1, tol=tol), max(5, reps // 2))
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

    def sweep():
        jax.block_until_ready(
            flood._step(
                fstate.messages, fstate.g_eta, fstate.g_lam, fstate.g_obs,
                fstate.g_mask, fstate.ef_eta, fstate.ef_obs, fstate.ef_mask,
                fstate.params, 0.0, False,
            )
        )

    # The recorded ratios predate the status-bearing run loop (per-sweep FE
    # and support checks), so the comparable flooding figure is raw sweeps:
    # sweep kernel time x sweeps-to-convergence.
    out["flood_kernel_ms"] = _median_ms(sweep, reps) * fresult.iterations

    if ref_ms is None and flood_ratio is not None:
        ref_ms = out["flood_kernel_ms"] / flood_ratio
        out["ref_derived"] = True
    if ref_ms is not None:
        out["ref_ms"] = ref_ms
        out["kernel_x"] = out["kernel_ms"] / ref_ms
        out["status_x"] = out["status_ms"] / ref_ms
    return out


def _print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n== {title} ==")
    cols = (
        "name compile_ms schedule_ms cold_ms kernel_ms status_ms readout_ms "
        "flood_kernel_ms flood_ms flood_sweeps"
    ).split()
    print(" ".join(f"{c:>12}" for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:>12.3f}" if isinstance(v, float) else f"{v!s:>12}")
        print(" ".join(cells))
    for r in rows:
        if "ref_ms" in r:
            src = "derived" if r.get("ref_derived") else "recorded"
            print(
                f"  {r['name']}: reference {r['ref_ms']:.4f} ms ({src}) -> "
                f"kernel {r['kernel_x']:.1f}x, status-bearing run {r['status_x']:.1f}x "
                f"(flooding sweeps were {r['flood_kernel_ms'] / r['ref_ms']:.0f}x)"
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
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)  # the recorded parity protocol
    from cffg.models.kalman import lgssm

    k = args.k
    rng = np.random.default_rng(0)
    a_mat = rng.uniform(0.1, 1.0, (k, k))
    a_mat /= a_mat.sum(axis=0)
    b_rand = rng.uniform(0.1, 1.0, (k, k))
    b_rand /= b_rand.sum(axis=0)
    d = np.full(k, 1.0 / k)

    rows = []
    for n in (32, 64, 128, 256):
        row = bench_case(
            f"hostile T={n}",
            hmm_chain(n, k),
            {"d": d, "B": np.eye(k), "A": a_mat},
            {f"y{n}": 1},
            reps=args.reps,
            flood_iters=3 * n + 10,
            tol=1e-12,
            readout="disc",
            ref_ms=HOSTILE_REF_MS.get(n),
            flood_ratio=HOSTILE_FLOOD_RATIO.get(n),
        )
        row["size"] = n
        rows.append(row)
    _print_rows("hostile identity-dynamics HMM, only final state observed", rows)

    rows = []
    for n in (32, 64, 128, 256):
        obs = rng.integers(0, k, size=n)
        row = bench_case(
            f"ordinary T={n}",
            hmm_chain(n, k),
            {"d": d, "B": b_rand, "A": a_mat},
            {f"y{i + 1}": int(obs[i]) for i in range(n)},
            reps=args.reps,
            flood_iters=3 * n + 10,
            tol=1e-12,
            readout="disc",
            flood_ratio=ORDINARY_FLOOD_RATIO.get(n),
        )
        row["size"] = n
        rows.append(row)
    _print_rows("ordinary K=8 HMM, fully observed", rows)

    f = np.array([[1.0, 0.3], [0.0, 0.9]])
    q = np.array([[0.05, 0.01], [0.01, 0.1]])
    h = np.array([[1.0, 0.5]])
    r = np.array([[0.2]])
    m0 = np.array([0.5, -0.3])
    p0 = np.array([[1.0, 0.2], [0.2, 0.8]])
    rows = []
    for n in (16, 64, 256):
        t, arrays = lgssm(n, f, q, h, r, m0, p0)
        ys = {f"y{i}": rng.normal(size=1) for i in range(n)}
        row = bench_case(
            f"lgssm T={n}",
            t,
            arrays,
            ys,
            reps=max(5, args.reps // 2),
            flood_iters=3 * n + 10,
            tol=1e-10,
            readout="gauss",
            flood_ratio=LGSSM_FLOOD_RATIO.get(n),
        )
        row["size"] = n
        rows.append(row)
    _print_rows("LGSSM (2-state, scalar observations)", rows)


if __name__ == "__main__":
    main()
