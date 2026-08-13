"""Gaussian conjugate learning (cffg.conj + Engine.gaussian_node_stats):
D4's update algebra against MLE concentration, and the acceptance pin —
recovering known LGSSM parameters from data through the engine's smoother
in a Dirichlet-style trial loop."""

import jax.numpy as jnp
import numpy as np
import pytest

from cffg import conj, recurrence, set_params
from cffg.compiler import compile_template
from cffg.engine import Engine, _replace
from cffg.learn import Learnables


def _rng():
    return np.random.default_rng(7)


# --- update algebra ----------------------------------------------------------


def test_niw_concentrates_at_empirical_mle():
    rng = _rng()
    mu_true, cov_true = np.array([1.0, -2.0]), np.array([[0.5, 0.2], [0.2, 0.8]])
    xs = rng.multivariate_normal(mu_true, cov_true, size=4000)
    post = conj.vague_niw(2).update(xs, np.einsum("ni,nj->nij", xs, xs))
    mean, cov_roll = post.rollover()
    np.testing.assert_allclose(mean, xs.mean(axis=0), atol=1e-6)
    emp_cov = np.cov(xs.T, bias=True)
    np.testing.assert_allclose(cov_roll, emp_cov, atol=0.02)  # O(1/n) prior drag
    np.testing.assert_allclose(post.mean_cov(), emp_cov, atol=0.02)


def test_niw_soft_stats_equal_hard_stats_in_expectation():
    """Soft statistics (E[x], E[xxᵀ] with within-trial variance) must fold
    the variance into Ψ — the D4 'within' term."""
    means = np.array([[1.0], [3.0]])
    var = 0.25
    seconds = np.array([[[1.0 + var]], [[9.0 + var]]])
    post = conj.vague_niw(1).update(means, seconds)
    hard = conj.vague_niw(1).update(means, np.array([[[1.0]], [[9.0]]]))
    assert post.psi[0, 0] == pytest.approx(hard.psi[0, 0] + 2 * var)


def test_mniw_recovers_regression():
    rng = _rng()
    a_true, b_true, v_true = np.array([[0.9, -0.3], [0.1, 0.7]]), np.array([0.5, -1.0]), 0.04
    xs = rng.normal(size=(5000, 2))
    ys = xs @ a_true.T + b_true + rng.normal(scale=np.sqrt(v_true), size=(5000, 2))
    xt = np.concatenate([xs, np.ones((5000, 1))], axis=1)
    post = conj.vague_mniw(2, 2).update(
        syy=ys.T @ ys, syx=ys.T @ xt, sxx=xt.T @ xt, n=5000
    )
    a_hat, b_hat, cov_hat = post.rollover()
    np.testing.assert_allclose(a_hat, a_true, atol=0.02)
    np.testing.assert_allclose(b_hat, b_true, atol=0.02)
    np.testing.assert_allclose(cov_hat, v_true * np.eye(2), atol=0.01)


def test_broken_stats_are_loud():
    with pytest.raises(ValueError, match="positive definite"):
        conj.vague_mniw(1, 1).update(
            syy=np.array([[-5.0]]), syx=np.zeros((1, 2)), sxx=np.eye(2), n=1.0
        )


def test_pg_weights_natural_update_rollover_and_loudness():
    prior = conj.vague_pg_weights(2, 1, strength=0.2)
    linear = np.array([[[0.3, -0.1]], [[-0.2, 0.4]]])
    quadratic = np.array(
        [
            [[[0.8, 0.1], [0.1, 0.5]]],
            [[[0.6, -0.2], [-0.2, 0.7]]],
        ]
    )
    posterior = prior.update(linear, quadratic)
    expected_precision = prior.precision + quadratic
    expected_mean = np.linalg.solve(
        expected_precision, linear[..., None]
    )[..., 0]
    weights, offsets = posterior.rollover()
    np.testing.assert_allclose(posterior.precision, expected_precision)
    np.testing.assert_allclose(posterior.mean(), expected_mean)
    np.testing.assert_allclose(weights, expected_mean[..., :-1])
    np.testing.assert_allclose(offsets, expected_mean[..., -1])
    np.testing.assert_allclose(
        posterior.covariance(), np.linalg.inv(expected_precision)
    )
    unchanged = posterior.update(
        np.zeros((3, *linear.shape)), np.zeros((3, *quadratic.shape))
    )
    np.testing.assert_array_equal(unchanged.eta, posterior.eta)
    np.testing.assert_array_equal(unchanged.precision, posterior.precision)

    with pytest.raises(ValueError, match="positive semidefinite"):
        prior.update(linear, -quadratic)
    with pytest.raises(ValueError, match="finite"):
        prior.update(np.full_like(linear, np.nan), quadratic)
    with pytest.raises(ValueError, match="positive definite"):
        conj.PGWeights(np.zeros_like(linear), np.zeros_like(quadratic))


def test_pg_weights_recover_soft_stick_dynamics_in_one_vectorized_batch():
    true_weights = np.array([[[1.1]], [[-0.8]]])
    true_offsets = np.array([[-0.25], [0.35]])
    points_per_source = 40
    drive = np.tile(np.linspace(-3.0, 3.0, points_per_source), 2)
    source = np.repeat(np.arange(2), points_per_source)
    logits = (
        true_weights[source, 0, 0] * drive + true_offsets[source, 0]
    )
    success = 1.0 / (1.0 + np.exp(-logits))
    q_in = np.eye(2)[source]
    q_out = np.stack([success, 1.0 - success], axis=-1)
    drive_mean = drive[:, None]
    drive_second = np.square(drive)[:, None, None]
    prior = conj.vague_pg_weights(2, 1)
    weights = np.zeros_like(true_weights)
    offsets = np.zeros_like(true_offsets)

    for _ in range(8):
        count = drive.shape[0]
        stats = recurrence.weight_stats(
            q_out,
            q_in,
            drive_mean,
            drive_second,
            np.broadcast_to(weights, (count, *weights.shape)),
            np.broadcast_to(offsets, (count, *offsets.shape)),
        )
        posterior = prior.update(stats.linear, stats.quadratic)
        weights, offsets = posterior.rollover()

    assert np.all((q_out > 0.0) & (q_out < 1.0))
    np.testing.assert_allclose(weights, true_weights, atol=0.01)
    np.testing.assert_allclose(offsets, true_offsets, atol=0.005)


# --- engine stats + trial-loop recovery (acceptance) ---------------------------


def _lgssm_template(d, dy):
    from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec

    t = Template(name="lgssm_learn")
    for n in ("m0", "P0", "A", "boff", "Q", "H", "h0", "R"):
        t.blobs[n] = BlobRef(n, f"blobs/{n}.npy")
    t.variables["x0"] = VariableSpec("x0", dim=d)
    t.variables["x1"] = VariableSpec("x1", dim=d)
    t.variables["y1"] = VariableSpec("y1", dim=dy)
    t.factors["prior"] = FactorSpec(
        "prior", "gaussian", edges={"out": "x0"}, param_blobs={"mean": "m0", "cov": "P0"}
    )
    t.factors["trans"] = FactorSpec(
        "trans", "gaussian_linear", edges={"out": "x1", "in": "x0"},
        param_blobs={"matrix": "A", "offset": "boff", "cov": "Q"},
    )
    t.factors["em"] = FactorSpec(
        "em", "gaussian_linear", edges={"out": "y1", "in": "x1"},
        param_blobs={"matrix": "H", "offset": "h0", "cov": "R"},
    )
    t.ports["obs"] = PortSpec("obs", "y1", PortRole.DATA)
    return t


def test_stats_readout_matches_hand_computation():
    """gaussian_node_stats on a tiny observed model: the transition node's
    joint stats must equal the smoothed pairwise moments computed by hand
    from the same beliefs."""
    d = 1
    t = _lgssm_template(d, d)
    arrays = {
        "m0": np.zeros(d), "P0": np.eye(d), "A": np.array([[0.8]]),
        "boff": np.zeros(d), "Q": np.array([[0.3]]), "H": np.eye(d),
        "h0": np.zeros(d), "R": np.array([[0.1]]),
    }
    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.bind_data(eng.init_state(), {"obs": np.array([1.2])})
    st = eng.run(st, 20, tol=1e-13).require_converged()
    stats = eng.gaussian_node_stats(st)
    gm = eng.gaussian_marginals(st)
    # prior node: E[x0], E[x0²] match the x0 marginal (tree exactness)
    m0, c0 = gm["x0"]
    np.testing.assert_allclose(stats["prior"]["mean"], m0, atol=1e-9)
    np.testing.assert_allclose(stats["prior"]["second"], c0 + np.outer(m0, m0), atol=1e-9)
    # transition node marginals: Sxx corner blocks match edge marginals
    m1, c1 = gm["x1"]
    np.testing.assert_allclose(stats["trans"]["Syy"], c1 + np.outer(m1, m1), atol=1e-9)
    np.testing.assert_allclose(stats["trans"]["Sxx"][:d, :d], c0 + np.outer(m0, m0), atol=1e-9)
    np.testing.assert_allclose(stats["trans"]["Sxx"][d, :d], m0, atol=1e-9)
    # observed emission node: y is the bound data
    np.testing.assert_allclose(stats["em"]["Syy"], np.outer([1.2], [1.2]), atol=1e-12)
    assert stats["em"]["mask"] == 1.0
    assert all(bool(s["proper"]) for s in stats.values())


def test_stats_readout_marks_mutated_invalid_precision_improper():
    t = _lgssm_template(1, 1)
    arrays = {
        "m0": np.zeros(1), "P0": np.eye(1), "A": np.ones((1, 1)),
        "boff": np.zeros(1), "Q": np.eye(1), "H": np.eye(1),
        "h0": np.zeros(1), "R": np.eye(1),
    }
    eng = Engine(compile_template(t, arrays=arrays))
    st = eng.bind_data(eng.init_state(), {"obs": np.array([0.0])})
    st = eng.run(st, 10, tol=1e-12).require_converged()
    params = dict(st.params)
    key = next(k for k in params if "gaussian_linear" in k and k.endswith(":W"))
    params[key] = -jnp.ones_like(params[key])
    bad = _replace(st, params=params)

    stats = eng.gaussian_node_stats(bad)["trans"]
    assert not bool(stats["proper"])
    assert np.isnan(stats["Syy"]).all()
    assert np.isnan(stats["Syx"]).all()
    assert np.isnan(stats["Sxx"]).all()
    assert np.isposinf(eng.bfe(bad))


@pytest.mark.slow
def test_trial_loop_recovers_lgssm_parameters():
    """Acceptance (cff-7c5o): the Dirichlet-style trial loop — infer with
    current expected parameters, accumulate gaussian_node_stats, MNIW-update,
    roll E-parameters back — recovers the generating transition matrix and
    noise from one-step trials with observed endpoints."""
    rng = _rng()
    a_true, q_true = 0.7, 0.09
    d = 1
    t = _lgssm_template(d, d)
    base = {
        "m0": np.zeros(d), "P0": np.eye(d) * 4.0,  # vague start prior
        "A": np.array([[0.0]]), "boff": np.zeros(d), "Q": np.array([[1.0]]),
        "H": np.eye(d), "h0": np.zeros(d), "R": np.array([[1e-4]]),
    }
    from cffg.core import LearnableSpec

    t.learnables["transition"] = LearnableSpec(
        "transition", "mniw", ("A", "boff", "Q"), sources=("trans",)
    )
    arrays = {**base, "P0": np.eye(d) * 1e-4}  # x0 pinned per trial via :m
    eng = Engine(compile_template(t, arrays=arrays))
    learnables = Learnables(
        eng, priors={"transition": conj.vague_mniw(d, d)}
    )
    for _ in range(400):
        x0 = rng.normal(scale=2.0)
        x1 = a_true * x0 + rng.normal(scale=np.sqrt(q_true))
        state = eng.init_state()
        params = learnables.write(eng, state.params)
        params = set_params(eng, params, {"prior": {"mean": np.array([x0])}})
        state = _replace(state, params=params)
        state = eng.bind_data(state, {"obs": np.array([x1])})
        result = eng.run(state, 15, tol=1e-13)
        result.require_converged()
        learnables.accumulate(eng, result)
    posterior = conj.MNIW(
        m=learnables.params["transition"]["m"],
        lam=learnables.params["transition"]["lam"],
        psi=learnables.params["transition"]["psi"],
        nu=float(learnables.params["transition"]["nu"]),
    )
    a_cur, b_cur, q_cur = posterior.rollover()
    assert abs(a_cur[0, 0] - a_true) < 0.05
    assert abs(b_cur[0]) < 0.05
    assert abs(q_cur[0, 0] - q_true) < 0.03
