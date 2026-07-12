"""Kalman tier (cff-4e5z): the continuous mirror of the HMM parity tier.

BP on the LGSSM tree must reproduce the RTS smoother exactly, and the Bethe
ledger must equal −log p(y) — the continuous tree identity. Also pins the
serialization round-trip and the vmap/colony contract for Gaussian arenas.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cffg.compiler import CompileError, compile_template
from cffg.core.serialize import from_json, to_json
from cffg.engine import Engine, _replace
from cffg.models.kalman import kalman_smoother, lgssm

T, N, M = 5, 2, 1
F = np.array([[1.0, 0.3], [0.0, 0.9]])
Q = np.array([[0.05, 0.01], [0.01, 0.1]])
H = np.array([[1.0, 0.5]])
R = np.array([[0.2]])
M0 = np.array([0.5, -0.3])
P0 = np.array([[1.0, 0.2], [0.2, 0.8]])

YS = [np.array([0.9]), np.array([1.4]), np.array([0.7]), np.array([-0.2]), np.array([0.3])]


def run_engine(ys):
    t, arrays = lgssm(T, F, Q, H, R, M0, P0)
    engine = Engine(compile_template(t, arrays=arrays))
    state = engine.bind_data(engine.init_state(), {f"y{k}": ys[k] for k in range(T)})
    state, deltas = engine.run(state, 3 * T + 10, tol=1e-12)
    assert deltas[-1] < 1e-12  # tree: exact convergence, not near-convergence
    return engine, state


def test_smoothed_marginals_match_rts():
    engine, state = run_engine(YS)
    means, covs, _ = kalman_smoother(YS, F, Q, H, R, M0, P0)
    marg = engine.gaussian_marginals(state)
    for k in range(T):
        mean, cov = marg[f"x{k}"]
        np.testing.assert_allclose(mean, means[k], atol=1e-8)
        np.testing.assert_allclose(cov, covs[k], atol=1e-8)
        if k < T - 1:  # every leg of the equality carries the same marginal
            np.testing.assert_allclose(marg[f"x{k}o"][0], means[k], atol=1e-8)


def test_bfe_is_negative_log_likelihood():
    engine, state = run_engine(YS)
    _, _, loglik = kalman_smoother(YS, F, Q, H, R, M0, P0)
    np.testing.assert_allclose(engine.bfe(state), -loglik, atol=1e-7)


def test_missing_observation():
    ys = list(YS)
    ys[2] = None
    engine, state = run_engine(ys)
    means, covs, loglik = kalman_smoother(ys, F, Q, H, R, M0, P0)
    marg = engine.gaussian_marginals(state)
    for k in range(T):
        np.testing.assert_allclose(marg[f"x{k}"][0], means[k], atol=1e-8)
        np.testing.assert_allclose(marg[f"x{k}"][1], covs[k], atol=1e-8)
    np.testing.assert_allclose(engine.bfe(state), -loglik, atol=1e-7)


def test_predictive_readout_on_observed_edge():
    # The observed edge's exterior marginal is the *cavity* predictive: the
    # forward message excludes y0's own likelihood (BP semantics), i.e. the
    # distribution of y0 given every OTHER observation — the innovation
    # quantity a model-scoring loop wants.
    engine, state = run_engine(YS)
    means_loo, covs_loo, _ = kalman_smoother([None, *YS[1:]], F, Q, H, R, M0, P0)
    mean_y, cov_y = engine.gaussian_marginals(state)["y0"]
    np.testing.assert_allclose(mean_y, H @ means_loo[0], atol=1e-8)
    np.testing.assert_allclose(cov_y, H @ covs_loo[0] @ H.T + R, atol=1e-8)


def test_serialization_round_trip_compiles_and_agrees():
    t, arrays = lgssm(T, F, Q, H, R, M0, P0)
    back = from_json(to_json(t))
    engine = Engine(compile_template(back, arrays=arrays))
    state = engine.bind_data(engine.init_state(), {f"y{k}": YS[k] for k in range(T)})
    state, _ = engine.run(state, 3 * T + 10)
    means, _, _ = kalman_smoother(YS, F, Q, H, R, M0, P0)
    np.testing.assert_allclose(
        engine.gaussian_marginals(state)["x3"][0], means[3], atol=1e-8
    )


def test_vmap_batch_equals_sequential():
    # Colony contract: Gaussian instances sharing a structure batch too.
    t, arrays = lgssm(T, F, Q, H, R, M0, P0)
    engine = Engine(compile_template(t, arrays=arrays))
    seqs = [YS, [y + 0.5 for y in YS], [None, *YS[1:]]]
    states = [
        engine.bind_data(engine.init_state(), {f"y{k}": s[k] for k in range(T)})
        for s in seqs
    ]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

    def one(st):
        msgs, g_eta, g_lam, ef_eta = engine._step(
            st.messages, st.g_eta, st.g_lam, st.g_obs, st.g_mask,
            st.ef_eta, st.ef_obs, st.ef_mask, st.params,
        )
        return _replace(st, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta)

    step_v = eqx.filter_vmap(one)
    for _ in range(3 * T + 10):
        batched = step_v(batched)
    for b in range(len(seqs)):
        solo, _ = engine.run(states[b], 3 * T + 10)
        np.testing.assert_allclose(
            np.asarray(batched.g_eta[b]), np.asarray(solo.g_eta), atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(batched.g_lam[b]), np.asarray(solo.g_lam), atol=1e-10
        )


def test_lgssm_renders_at_every_level(viz_gallery):
    # The viz pipeline is domain-agnostic (it never reads cardinality), so the
    # Gaussian chain renders today; Gaussian-specific glyphs are cff-bxk4.
    import xml.etree.ElementTree as ET

    from cffg.viz import render_svg

    t, _ = lgssm(3, F, Q, H, R, M0, P0)
    for level in (0, 4):
        svg = render_svg(t, level=level)
        ET.fromstring(svg)
        viz_gallery(svg, tag=f"LGSSM T=3, level {level}")


def test_lgssm_with_glue_ports_renders(viz_gallery):
    # A single Kalman slice the way the rollout machinery would see it:
    # glue in/out plus the emission branch — the composable unit.
    import xml.etree.ElementTree as ET

    from cffg.core.ir import PortRole, PortSpec
    from cffg.viz import render_svg

    t, _ = lgssm(2, F, Q, H, R, M0, P0)
    del t.factors["prior"]
    t.ports["head"] = PortSpec("head", "x0", PortRole.GLUE)
    svg = render_svg(t, level=4)
    ET.fromstring(svg)
    viz_gallery(svg, tag="Kalman slice with glue port")


def test_data_port_elsewhere_rejected():
    import pytest

    from cffg.core.ir import FactorSpec, PortRole, PortSpec, Template, VariableSpec

    t = Template(name="bad")
    t.variables["x"] = VariableSpec("x", dim=1)
    t.factors["prior"] = FactorSpec(
        "prior", "gaussian", edges={"out": "x"}, param_blobs={"mean": "m", "cov": "v"}
    )
    from cffg.core.ir import BlobRef

    t.blobs = {"m": BlobRef("m", "m.npy"), "v": BlobRef("v", "v.npy")}
    t.ports["x"] = PortSpec("x", "x", PortRole.DATA)
    with pytest.raises(CompileError, match="gaussian_linear 'out'"):
        compile_template(t, arrays={"m": np.zeros(1), "v": np.eye(1)})


def test_bad_covariance_rejected():
    import pytest

    t, arrays = lgssm(2, F, Q, H, R, M0, P0)
    arrays["R"] = np.array([[-1.0]])
    with pytest.raises(CompileError, match="positive definite"):
        compile_template(t, arrays=arrays)
