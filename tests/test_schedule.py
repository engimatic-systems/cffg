"""Schedule-aware inference (cff-d56d): exact two-pass directed schedules on
acyclic BP graphs must reproduce flooding's fixed point bit-for-bit at
tolerance, update each directed message O(1) times, stay deterministic under
template round-trips, and fall back to flooding (with a recorded reason) for
every structure outside the directed contract."""

from dataclasses import replace

import equinox as eqx
import jax
import numpy as np
import pytest

from cffg import engine as engine_mod
from cffg.compiler import compile_template
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.compose import compose, unroll
from cffg.core.serialize import from_json, to_json
from cffg.engine import Engine, RunStatus, _replace
from cffg.models.kalman import kalman_smoother, lgssm
from cffg.schedule import BatchStage, ScanStage, build_schedule
from cffg.schedule_program import make_exact_unit, make_propagate
from conftest import K, M, make_hmm_slice
from test_compose import make_state_prior

import jax.numpy as jnp

rng = np.random.default_rng(56)
T = 32  # long enough that uniform rounds fuse into lax.scan stages


def col_stochastic(rows, cols):
    m = rng.uniform(0.1, 1.0, size=(rows, cols))
    return m / m.sum(axis=0, keepdims=True)


D = rng.dirichlet(np.ones(K))
B = col_stochastic(K, K)
A = col_stochastic(M, K)


def hmm_template(n_steps=T):
    rolled = unroll(make_hmm_slice(), n_steps, {"tail": "head"})
    return compose(
        make_state_prior(), rolled, {"attach": "t0/head"},
        name="hmm", scope_a="prior", scope_b="chain",
    )


def hmm_arrays(n_steps=T):
    arrays = {"prior/d": D}
    for t in range(n_steps):
        arrays[f"chain/t{t}/A"] = A
        arrays[f"chain/t{t}/B"] = B
    return arrays


def hmm_engine(schedule="auto", n_steps=T, semiring="sum"):
    return Engine(
        compile_template(hmm_template(n_steps), arrays=hmm_arrays(n_steps), semiring=semiring),
        schedule=schedule,
    )


def forward_backward(obs):
    em = [np.ones(K) if o is None else A[o] for o in obs]
    alphas, log_cs = [D.copy()], []
    for t in range(len(obs)):
        a = em[t] * (B @ alphas[-1])
        c = a.sum()
        alphas.append(a / c)
        log_cs.append(np.log(c))
    betas = [np.ones(K)]
    for t in reversed(range(len(obs))):
        b = B.T @ (em[t] * betas[0])
        betas.insert(0, b / b.sum())
    smoothed = np.array([a * b for a, b in zip(alphas, betas)])
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return smoothed, float(np.sum(log_cs))


OBS = [int(v) for v in rng.integers(0, M, size=T)]


# -- exactness ---------------------------------------------------------------


def test_two_pass_hmm_matches_forward_backward_and_flooding():
    exact = hmm_engine("auto")
    flood = hmm_engine("flood")
    assert exact.schedule.kind == "two_pass"

    ports = {f"chain/t{t}/sense": OBS[t] for t in range(T)}
    st_e = exact.bind_data(exact.init_state(), ports)
    res = exact.run(st_e, 1, tol=1e-14)
    assert res.status is RunStatus.CONVERGED
    assert res.iterations == 1
    assert len(res.message_deltas) == 1

    st_f = flood.bind_data(flood.init_state(), ports)
    res_f = flood.run(st_f, 3 * T + 10, tol=1e-14)
    assert res_f.status is RunStatus.CONVERGED

    smoothed, loglik = forward_backward(OBS)
    marg_e = exact.marginals(res.state)
    marg_f = flood.marginals(res_f.state)
    np.testing.assert_allclose(marg_e["prior/z0"], smoothed[0], atol=1e-12)
    for t in range(T):
        np.testing.assert_allclose(
            marg_e[f"chain/t{t}/z_mid"], smoothed[t + 1], atol=1e-12
        )
    for name, q in marg_f.items():
        np.testing.assert_allclose(marg_e[name], q, atol=1e-12)
    np.testing.assert_allclose(res.free_energies[-1], -loglik, atol=1e-9)
    np.testing.assert_allclose(res.free_energies[-1], flood.bfe(res_f.state), atol=1e-9)


def test_two_pass_is_a_true_fixed_point():
    engine = hmm_engine("auto")
    ports = {f"chain/t{t}/sense": OBS[t] for t in range(T)}
    state = engine.bind_data(engine.init_state(), ports)
    result = engine.run(state, 1, tol=None)  # certificate applies even with tol=None
    assert result.status is RunStatus.CONVERGED
    stepped = engine.step(result.state)
    delta = float(jnp.max(jnp.abs(stepped.messages - result.state.messages)))
    assert delta < 1e-12


def test_long_range_information_flow_single_final_observation():
    """The hostile case: near-identity dynamics, only the last state observed.
    Evidence must traverse the whole chain in one schedule execution."""
    n = 24
    ident = 0.9 * np.eye(K) + 0.1 / K
    arrays = {"prior/d": D}
    for t in range(n):
        arrays[f"chain/t{t}/A"] = A
        arrays[f"chain/t{t}/B"] = ident
    engine = Engine(compile_template(hmm_template(n), arrays=arrays))
    assert engine.schedule.kind == "two_pass"
    obs = [None] * (n - 1) + [1]
    state = engine.bind_data(
        engine.init_state(), {f"chain/t{t}/sense": obs[t] for t in range(n)}
    )
    result = engine.run(state, 1, tol=1e-14)
    assert result.status is RunStatus.CONVERGED

    global B
    saved = B
    B = ident
    try:
        smoothed, loglik = forward_backward(obs)
    finally:
        B = saved
    marg = engine.marginals(result.state)
    np.testing.assert_allclose(marg["prior/z0"], smoothed[0], atol=1e-12)
    for t in range(n):
        np.testing.assert_allclose(
            marg[f"chain/t{t}/z_mid"], smoothed[t + 1], atol=1e-12
        )
    np.testing.assert_allclose(result.free_energies[-1], -loglik, atol=1e-9)


def test_two_pass_lgssm_matches_rts_smoother():
    n = 32
    f = np.array([[1.0, 0.3], [0.0, 0.9]])
    q = np.array([[0.05, 0.01], [0.01, 0.1]])
    h = np.array([[1.0, 0.5]])
    r = np.array([[0.2]])
    m0 = np.array([0.5, -0.3])
    p0 = np.array([[1.0, 0.2], [0.2, 0.8]])
    ys = [np.array([v]) for v in rng.normal(size=n)]
    ys[n // 2] = None  # missing observation stays conditioned correctly

    t, arrays = lgssm(n, f, q, h, r, m0, p0)
    engine = Engine(compile_template(t, arrays=arrays))
    assert engine.schedule.kind == "two_pass"
    assert any(isinstance(s, ScanStage) for s in engine.schedule.stages)

    state = engine.bind_data(engine.init_state(), {f"y{k}": ys[k] for k in range(n)})
    result = engine.run(state, 1, tol=1e-12)
    assert result.status is RunStatus.CONVERGED

    means, covs, loglik = kalman_smoother(ys, f, q, h, r, m0, p0)
    marg = engine.gaussian_marginals(result.state)
    for k in range(n):
        np.testing.assert_allclose(marg[f"x{k}"][0], means[k], atol=1e-8)
        np.testing.assert_allclose(marg[f"x{k}"][1], covs[k], atol=1e-8)
    np.testing.assert_allclose(result.free_energies[-1], -loglik, atol=1e-7)


def test_max_semiring_two_pass_matches_flooding_decode():
    exact = hmm_engine("auto", semiring="max")
    flood = hmm_engine("flood", semiring="max")
    assert exact.schedule.kind == "two_pass"
    # A tie-free observation sequence: exact MAP ties make the per-edge
    # argmax decode inconsistent (objective +inf) under EITHER schedule,
    # which is the documented max-product limitation, not a scheduling one.
    obs = [int(v) for v in np.random.default_rng(0).integers(0, M, size=T)]
    ports = {f"chain/t{t}/sense": obs[t] for t in range(T)}
    res = exact.run(exact.bind_data(exact.init_state(), ports), 1, tol=1e-12)
    assert res.status is RunStatus.CONVERGED
    res_f = flood.run(flood.bind_data(flood.init_state(), ports), 3 * T + 10, tol=1e-12)
    assert exact.decode(res.state) == flood.decode(res_f.state)
    np.testing.assert_allclose(
        exact.objective(res.state), flood.objective(res_f.state), atol=1e-9
    )


# -- structure ---------------------------------------------------------------


def test_each_factor_fires_exactly_twice_and_chains_scan():
    arena = compile_template(hmm_template(), arrays=hmm_arrays())
    sched = build_schedule(arena)
    assert sched.kind == "two_pass"
    assert any(isinstance(s, ScanStage) for s in sched.stages)

    fires: dict[tuple, int] = {}
    written_disc: set[int] = set()
    for stage in sched.stages:
        elems = stage.elems if isinstance(stage, ScanStage) else (stage,)
        for e in elems:
            g = arena.groups[e.group] if e.kind == "disc" else arena.gauss_groups[e.group]
            for mi in np.asarray(e.rows).reshape(-1):
                fires[(e.kind, e.group, int(mi))] = (
                    fires.get((e.kind, e.group, int(mi)), 0) + 1
                )
                if e.kind == "disc":
                    written_disc.update(int(s) for s in g.scatter[int(mi)])
    n_factors = sum(len(g.members) for g in arena.groups) + sum(
        len(g.members) for g in arena.gauss_groups
    )
    assert len(fires) == n_factors
    assert set(fires.values()) == {2}  # inward + outward, never more
    for e in arena.edges.values():  # every interior directed slot gets written
        if e.interior and not e.dim and not e.ef:
            assert e.slot0 in written_disc and e.slot1 in written_disc


def test_schedule_is_deterministic_across_template_round_trip():
    t = hmm_template()
    arrays = hmm_arrays()
    s1 = build_schedule(compile_template(t, arrays=arrays))
    s2 = build_schedule(compile_template(from_json(to_json(t)), arrays=arrays))
    assert s1.describe() == s2.describe()
    assert s1.rounds == s2.rounds


def test_short_models_unroll_without_scan_stages():
    arena = compile_template(hmm_template(3), arrays=hmm_arrays(3))
    sched = build_schedule(arena)
    assert sched.kind == "two_pass"
    assert all(isinstance(s, BatchStage) for s in sched.stages)


# -- boundaries: everything outside the contract floods ----------------------


def unit_categorical(constraints=()):
    return Template(
        "sched_unit",
        variables={"x": VariableSpec("x", 3)},
        factors={
            "prior": FactorSpec(
                "prior", "categorical", edges={"out": "x"},
                param_blobs={"statistic": "p"},
            )
        },
        ports={"data": PortSpec("data", "x", PortRole.DATA)},
        blobs={"p": BlobRef("p", "p")},
        constraints=list(constraints),
    )


def vmp_unit_template():
    return Template(
        "vmp_unit",
        variables={"z": VariableSpec("z", K), "x": VariableSpec("x", M)},
        factors={
            "prior": FactorSpec("prior", "categorical", edges={"out": "z"}, param_blobs={"statistic": "d"}),
            "obs": FactorSpec("obs", "matrix_cat", edges={"out": "x", "in": "z"}, param_blobs={"matrix": "A"}),
        },
        ports={"px": PortSpec("px", "x", PortRole.DATA)},
        blobs={"d": BlobRef("d", "d"), "A": BlobRef("A", "A")},
        constraints=[ConstraintSpec(ConstraintKind.MEAN_FIELD, "obs")],
    )


def test_vmp_bead_gets_mixed_schedule_with_flooding_parity():
    arrays = {"d": D, "A": A}
    mixed = Engine(compile_template(vmp_unit_template(), arrays=arrays))
    assert mixed.schedule.kind == "mixed"
    assert "vmp" in mixed.schedule.reason
    flood = Engine(compile_template(vmp_unit_template(), arrays=arrays), schedule="flood")
    st_m = mixed.bind_data(mixed.init_state(), {"px": 1})
    st_f = flood.bind_data(flood.init_state(), {"px": 1})
    res_m = mixed.run(st_m, 60, tol=1e-13)
    res_f = flood.run(st_f, 60, tol=1e-13)
    assert res_m.status is RunStatus.CONVERGED
    assert res_f.status is RunStatus.CONVERGED
    for name, q in flood.marginals(res_f.state).items():
        np.testing.assert_allclose(mixed.marginals(res_m.state)[name], q, atol=1e-9)
    np.testing.assert_allclose(mixed.bfe(res_m.state), flood.bfe(res_f.state), atol=1e-9)


def test_delta_constraint_forces_flooding():
    t = unit_categorical([ConstraintSpec(ConstraintKind.DELTA, "x")])
    sched = build_schedule(compile_template(t, arrays={"p": np.array([0.2, 0.3, 0.5])}))
    assert sched.kind == "flood"
    assert "delta" in sched.reason


def test_constraint_only_model_floods_and_constraint_chain_mixes():
    t = Template(
        "csp_unit",
        variables={"x0": VariableSpec("x0", 2), "x1": VariableSpec("x1", 2)},
        factors={
            "c": FactorSpec("c", "exactly_k", edges={"x0": "x0", "x1": "x1"}, param_blobs={"k": "k"}),
        },
        ports={
            "d0": PortSpec("d0", "x0", PortRole.DATA),
            "d1": PortSpec("d1", "x1", PortRole.DATA),
        },
        blobs={"k": BlobRef("k", "k")},
    )
    sched = build_schedule(compile_template(t, arrays={"k": np.asarray(1)}))
    assert sched.kind == "flood"  # nothing schedulable: only the constraint
    assert "no schedulable" in sched.reason

    t2 = Template(
        "csp_mixed",
        variables={
            "x0": VariableSpec("x0", 2),
            "x1": VariableSpec("x1", 2),
        },
        factors={
            "p0": FactorSpec("p0", "categorical", edges={"out": "x0"}, param_blobs={"statistic": "p"}),
            "c": FactorSpec("c", "exactly_k", edges={"x0": "x0", "x1": "x1"}, param_blobs={"k": "k"}),
        },
        ports={"d1": PortSpec("d1", "x1", PortRole.DATA)},
        blobs={"k": BlobRef("k", "k"), "p": BlobRef("p", "p")},
    )
    arena2 = compile_template(
        t2, arrays={"k": np.asarray(1), "p": np.array([0.3, 0.7])}
    )
    sched2 = build_schedule(arena2)
    assert sched2.kind == "mixed"
    assert "constraint" in sched2.reason
    mixed = Engine(arena2)
    flood = Engine(arena2, schedule="flood")
    st = mixed.bind_data(mixed.init_state(), {"d1": 0})
    res_m = mixed.run(st, 30, tol=1e-13)
    res_f = flood.run(flood.bind_data(flood.init_state(), {"d1": 0}), 30, tol=1e-13)
    assert res_m.status is RunStatus.CONVERGED and res_f.status is RunStatus.CONVERGED
    for name, q in flood.marginals(res_f.state).items():
        np.testing.assert_allclose(mixed.marginals(res_m.state)[name], q, atol=1e-9)


def test_cycle_gets_colored_schedule_with_flooding_parity():
    """A transition ring (loopy, contracting): colored Gauss–Seidel sweeps
    must reach the same loopy-BP fixed point flooding reaches."""
    t = Template(
        "ring",
        variables={
            "z1": VariableSpec("z1", K),
            "z1e": VariableSpec("z1e", K),
            "z2": VariableSpec("z2", K),
            "z3": VariableSpec("z3", K),
            "zo": VariableSpec("zo", K),
            "x": VariableSpec("x", M),
        },
        factors={
            "t1": FactorSpec("t1", "matrix_cat", edges={"out": "z2", "in": "z1e"}, param_blobs={"matrix": "B"}),
            "t2": FactorSpec("t2", "matrix_cat", edges={"out": "z3", "in": "z2"}, param_blobs={"matrix": "B"}),
            "t3": FactorSpec("t3", "matrix_cat", edges={"out": "z1", "in": "z3"}, param_blobs={"matrix": "B"}),
            "eq": FactorSpec("eq", "equality", edges={"a": "z1", "b": "z1e", "c": "zo"}),
            "obs": FactorSpec("obs", "matrix_cat", edges={"out": "x", "in": "zo"}, param_blobs={"matrix": "A"}),
        },
        ports={"px": PortSpec("px", "x", PortRole.DATA)},
        blobs={"B": BlobRef("B", "B"), "A": BlobRef("A", "A")},
    )
    arena = compile_template(t, arrays={"B": B, "A": A})
    sched = build_schedule(arena)
    assert sched.kind == "colored"
    assert sched.colors >= 2
    colored = Engine(arena)
    flood = Engine(arena, schedule="flood")
    st_c = colored.bind_data(colored.init_state(), {"px": 1})
    st_f = flood.bind_data(flood.init_state(), {"px": 1})
    res_c = colored.run(st_c, 200, tol=1e-12)
    res_f = flood.run(st_f, 200, tol=1e-12)
    assert res_c.status is RunStatus.CONVERGED
    assert res_f.status is RunStatus.CONVERGED
    assert res_c.iterations <= res_f.iterations  # GS never loses to Jacobi here
    for name, q in flood.marginals(res_f.state).items():
        np.testing.assert_allclose(colored.marginals(res_c.state)[name], q, atol=1e-8)
    np.testing.assert_allclose(
        colored.bfe(res_c.state), flood.bfe(res_f.state), atol=1e-8
    )


def test_two_pass_mode_raises_outside_contract():
    t = unit_categorical([ConstraintSpec(ConstraintKind.DELTA, "x")])
    arena = compile_template(t, arrays={"p": np.array([0.2, 0.3, 0.5])})
    with pytest.raises(ValueError, match="two_pass schedule unavailable"):
        build_schedule(arena, "two_pass")


# -- run() semantics on the exact path ---------------------------------------


def test_damping_and_hardening_requests_use_the_flooding_loop():
    engine = hmm_engine("auto", n_steps=4)
    ports = {f"chain/t{t}/sense": OBS[t] for t in range(4)}
    state = engine.bind_data(engine.init_state(), ports)
    result = engine.run(state, 400, tol=1e-13, damp=0.5)
    assert result.status is RunStatus.CONVERGED
    assert result.iterations > 1  # flooding loop, not the single execution
    result = engine.run(state, 20, tol=1e-13, harden_last=5)
    assert result.iterations > 1


def test_poisoned_state_diverges_before_propagation():
    engine = hmm_engine("auto", n_steps=4)
    state = engine.init_state()
    poisoned = _replace(state, messages=state.messages.at[0].set(jnp.nan))
    result = engine.run(poisoned, 1, tol=1e-12)
    assert result.status is RunStatus.DIVERGED
    assert result.iterations == 0


def test_zero_support_reports_infeasible_on_exact_path():
    t = unit_categorical()
    engine = Engine(compile_template(t, arrays={"p": np.array([1.0, 0.0, 0.0])}))
    assert engine.schedule.kind == "two_pass"
    state = engine.bind_data(engine.init_state(), {"data": 1})
    result = engine.run(state, 1, tol=1e-12)
    assert result.status is RunStatus.INFEASIBLE
    assert np.isposinf(result.free_energies[-1])


# -- certificate soundness and prebuilt-schedule validation ------------------


def test_doctored_prebuilt_schedule_is_rejected():
    arena = compile_template(hmm_template(8), arrays=hmm_arrays(8))
    sched = build_schedule(arena)
    half = replace(sched, stages=sched.stages[: len(sched.stages) // 2])
    with pytest.raises(ValueError, match="does not match this arena"):
        Engine(arena, schedule=half)
    other = compile_template(hmm_template(9), arrays=hmm_arrays(9))
    with pytest.raises(ValueError, match="does not match this arena"):
        Engine(other, schedule=sched)
    Engine(arena, schedule=sched)  # the genuine schedule is accepted


def test_permuted_scan_rows_are_rejected_despite_identical_describe():
    """describe() records scan dimensions, not row content — the fingerprint
    must seal the row tables themselves (2026-07-27 review, second round)."""
    arena = compile_template(hmm_template(T), arrays=hmm_arrays(T))
    sched = build_schedule(arena)
    stages = list(sched.stages)
    i, scan = next((i, s) for i, s in enumerate(stages) if isinstance(s, ScanStage))
    reversed_elems = tuple(
        replace(e, rows=np.ascontiguousarray(e.rows[::-1])) for e in scan.elems
    )
    stages[i] = replace(scan, elems=reversed_elems)
    tampered = replace(sched, stages=tuple(stages))
    assert tampered.describe() == sched.describe()  # the textual form ties
    with pytest.raises(ValueError, match="does not match this arena"):
        Engine(arena, schedule=tampered)


def test_float32_missing_phase_cannot_certify_at_t256():
    """The reviewer's exploit (2026-07-27): drop the outward pass on a long
    float32 chain. The stale-message residual (~1e-3) is far above the
    scale-relative certificate, so the run must not report converged — a
    diameter-scaled floor (64 eps rounds ≈ 3.9e-3 in f32 at T=256) would
    have swallowed it."""
    n = 256
    with jax.enable_x64(False):
        arena = compile_template(hmm_template(n), arrays=hmm_arrays(n))
        engine = Engine(arena)
        assert engine.schedule.kind == "two_pass"
        inward_only = make_propagate(
            arena, engine.schedule.stages[: len(engine.schedule.stages) // 2]
        )
        engine._exact_unit = eqx.filter_jit(
            make_exact_unit(
                arena, inward_only, engine_mod._make_step(arena), engine_mod._make_bfe(arena)
            )
        )  # bypasses the fingerprint guard on purpose: this pins the
        # *certificate* itself, not the schedule validation
        obs = [int(v) for v in np.random.default_rng(1).integers(0, M, size=n)]
        state = engine.bind_data(
            engine.init_state(), {f"chain/t{t}/sense": obs[t] for t in range(n)}
        )
        result = engine.run(state, 1, tol=None)
        assert result.status is not RunStatus.CONVERGED
        with pytest.raises(Exception):
            result.require_converged()


def test_float32_genuine_two_pass_still_certifies_at_t256():
    n = 256
    with jax.enable_x64(False):
        engine = Engine(compile_template(hmm_template(n), arrays=hmm_arrays(n)))
        obs = [int(v) for v in np.random.default_rng(1).integers(0, M, size=n)]
        state = engine.bind_data(
            engine.init_state(), {f"chain/t{t}/sense": obs[t] for t in range(n)}
        )
        result = engine.run(state, 1, tol=None)
        assert result.status is RunStatus.CONVERGED


def mixed_scale_template(n):
    """The f32 chain plus an *unrelated* scalar Gaussian island whose
    precision is 1e12: any global certificate scale would be dominated by
    the Gaussian block."""
    from cffg.core.ir import BlobRef as BR, FactorSpec as FS, VariableSpec as VS

    t = hmm_template(n)
    t.variables["gx"] = VS("gx", dim=1)
    t.variables["gy"] = VS("gy", dim=1)
    t.factors["gprior"] = FS(
        "gprior", "gaussian", edges={"out": "gx"},
        param_blobs={"mean": "gm", "cov": "gc"},
    )
    t.factors["glin"] = FS(
        "glin", "gaussian_linear", edges={"out": "gy", "in": "gx"},
        param_blobs={"matrix": "gA", "offset": "gb", "cov": "gc"},
    )
    t.ports["gout"] = PortSpec("gout", "gy", PortRole.GLUE)
    for b in ("gm", "gc", "gA", "gb"):
        t.blobs[b] = BR(b, b)
    arrays = dict(hmm_arrays(n))
    arrays.update(
        gm=np.zeros(1), gc=np.full((1, 1), 1e-12),
        gA=np.ones((1, 1)), gb=np.zeros(1),
    )
    return t, arrays


def test_float32_mixed_scales_cannot_widen_the_certificate():
    """Second-round review exploit: a huge-precision Gaussian island must
    not raise the certificate for a stale discrete chain — the residual
    comparison is elementwise, not global-scale."""
    n = 256
    with jax.enable_x64(False):
        t, arrays = mixed_scale_template(n)
        arena = compile_template(t, arrays=arrays)
        engine = Engine(arena)
        assert engine.schedule.kind == "two_pass"
        obs = [int(v) for v in np.random.default_rng(1).integers(0, M, size=n)]
        state = engine.bind_data(
            engine.init_state(), {f"chain/t{t}/sense": obs[t] for t in range(n)}
        )
        # The genuine schedule still certifies despite the 1e12 block.
        assert engine.run(state, 1, tol=None).status is RunStatus.CONVERGED
        # A missing outward pass must not.
        inward_only = make_propagate(
            arena, engine.schedule.stages[: len(engine.schedule.stages) // 2]
        )
        engine._exact_unit = eqx.filter_jit(
            make_exact_unit(
                arena, inward_only, engine_mod._make_step(arena), engine_mod._make_bfe(arena)
            )
        )
        result = engine.run(state, 1, tol=None)
        assert result.status is not RunStatus.CONVERGED


# -- EF directed passes ------------------------------------------------------


def test_ef_tree_runs_two_pass_with_flooding_parity():
    from test_ef_engine import gamma_poisson

    t, arrays = gamma_poisson(6)
    arena = compile_template(t, arrays=arrays)
    exact = Engine(arena)
    assert exact.schedule.kind == "two_pass"
    flood = Engine(arena, schedule="flood")
    data = {f"count{i}": c for i, c in enumerate((3, 5, 1, None, 4, 2))}
    res_e = exact.run(exact.bind_data(exact.init_state(), data), 1, tol=1e-12)
    assert res_e.status is RunStatus.CONVERGED
    res_f = flood.run(flood.bind_data(flood.init_state(), data), 40, tol=1e-12)
    assert res_f.status is RunStatus.CONVERGED
    me, mf = exact.ef_marginals(res_e.state), flood.ef_marginals(res_f.state)
    for name in mf:
        np.testing.assert_allclose(me[name].params, mf[name].params, atol=1e-10)
    np.testing.assert_allclose(exact.bfe(res_e.state), flood.bfe(res_f.state), atol=1e-10)
