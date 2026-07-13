"""Executable anchors for DERIVATIONS.md D5 (cff-dfpi): hybrid GFE epistemics.

Every proposition in D5 that can be executed is pinned here against
independent oracles (grid quadrature, finite differences, dense simplex
search) — none of the checks reuse the closed form they verify.
"""

import numpy as np
import pytest

from cffg.experiments.hybrid_epistemics import (
    GfeTerms,
    ProbeWorld,
    Predictive,
    collapse,
    discrete_entropy,
    entropy_collapse_upper,
    entropy_conditional_lower,
    entropy_joint_upper,
    entropy_pairwise_lower,
    entropy_quadrature,
    gauss_entropy,
    gfe_exact_quadrature,
    gfe_hat,
    goal_cross_entropy,
    grad_collapse_entropy,
    lgssm_policy_terms,
    mi_regime_bounds,
    mi_regime_hat,
    predictive,
    regime_log_potential,
    run_probe_experiment,
    solve_regime_fixed_point,
    step1_scores,
)


def random_mixture(rng, K=3, spread=2.0):
    """A random 1-D labelled mixture (the quadrature oracle needs d_x = 1)."""
    rho = rng.dirichlet(np.ones(K))
    xbar = rng.normal(0, spread, (K, 1))
    P = rng.uniform(0.05, 1.5, (K, 1, 1))
    return Predictive(rho=rho, xbar=xbar, P=P)


def random_model(rng, K=3, dz=2):
    ms = rng.normal(0, 1, (K, dz))
    Ss = np.stack([_spd(rng, dz) for _ in range(K)])
    As = rng.normal(0, 1, (K, 1, dz))
    bs = rng.normal(0, 2, (K, 1))
    Vs = rng.uniform(0.05, 1.0, (K, 1, 1))
    return ms, Ss, As, bs, Vs


def _spd(rng, d):
    a = rng.normal(0, 1, (d, d))
    return a @ a.T + 0.3 * np.eye(d)


# --- D5.2 Prop. 3: the mixture-entropy sandwich ------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_entropy_sandwich_against_quadrature(seed):
    rng = np.random.default_rng(seed)
    pred = random_mixture(rng, K=2 + seed % 3)
    h = entropy_quadrature(pred)
    assert entropy_pairwise_lower(pred) <= h + 1e-6
    assert entropy_conditional_lower(pred) <= h + 1e-6
    assert h <= entropy_collapse_upper(pred) + 1e-6
    assert h <= entropy_joint_upper(pred) + 1e-6


def test_sandwich_strict_when_separated_tight_when_coincident():
    rho = np.array([0.5, 0.5])
    apart = Predictive(rho, np.array([[-4.0], [4.0]]), np.full((2, 1, 1), 0.2))
    same = Predictive(rho, np.zeros((2, 1)), np.full((2, 1, 1), 0.2))
    # coincident components: every bound collapses onto the exact entropy
    h = entropy_quadrature(same)
    assert np.isclose(entropy_collapse_upper(same), h, atol=1e-6)
    assert np.isclose(entropy_pairwise_lower(same), h, atol=0.35)  # HO is not tight
    assert np.isclose(mi_regime_hat(same), 0.0, atol=1e-12)
    # separated: exact I(x;s) -> H(rho) = log 2, surrogate exceeds it (documented
    # optimism), the H(rho) cap in mi_regime_bounds restores a tight upper bound
    h_apart = entropy_quadrature(apart)
    exact_mi = h_apart - entropy_conditional_lower(apart)
    assert np.isclose(exact_mi, np.log(2), atol=1e-3)
    assert mi_regime_hat(apart) > np.log(2)
    lo, hi = mi_regime_bounds(apart)
    assert lo <= exact_mi <= hi + 1e-9
    assert np.isclose(hi, np.log(2), atol=1e-12)


# --- D5.2 Prop. 1: exact terms and the bound direction of the assembled G ----


@pytest.mark.parametrize("seed", range(3))
def test_goal_cross_entropy_matches_quadrature(seed):
    rng = np.random.default_rng(seed)
    pred = random_mixture(rng)
    g, gamma = rng.normal(), rng.uniform(0.3, 2.0)
    x = np.linspace(-40, 40, 400001)
    dens = np.zeros_like(x)
    for r, m, v in zip(pred.rho, pred.xbar[:, 0], pred.P[:, 0, 0]):
        dens += r * np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
    neglog = 0.5 * (np.log(2 * np.pi * gamma) + (x - g) ** 2 / gamma)
    oracle = np.trapezoid(dens * neglog, x)
    assert np.isclose(goal_cross_entropy(pred, g, gamma), oracle, atol=1e-6)


@pytest.mark.parametrize("seed", range(3))
def test_gfe_hat_lower_bounds_exact_objective(seed):
    rng = np.random.default_rng(10 + seed)
    K = 2 + seed
    rho = rng.dirichlet(np.ones(K))
    ms, Ss, As, bs, Vs = random_model(rng, K=K)
    g, gamma = 0.5, 1.5
    ghat = gfe_hat(rho, ms, Ss, As, bs, Vs, g, gamma)
    gexact = gfe_exact_quadrature(rho, ms, Ss, As, bs, Vs, g, gamma)
    assert ghat.total <= gexact + 1e-6  # optimistic, same stance as S1c/S1d
    # identical components: surrogate exact, regime channel exactly zero
    ms1, Ss1, As1, bs1, Vs1 = (np.repeat(a[:1], K, axis=0) for a in (ms, Ss, As, bs, Vs))
    ghat1 = gfe_hat(rho, ms1, Ss1, As1, bs1, Vs1, g, gamma)
    assert np.isclose(ghat1.mi_regime_hat, 0.0, atol=1e-12)
    assert np.isclose(
        ghat1.total, gfe_exact_quadrature(rho, ms1, Ss1, As1, bs1, Vs1, g, gamma), atol=1e-5
    )


def test_mi_terms_nonnegative():
    rng = np.random.default_rng(42)
    for _ in range(10):
        pred = random_mixture(rng)
        assert mi_regime_hat(pred) >= -1e-12
        lo, hi = mi_regime_bounds(pred)
        assert -1e-12 <= lo <= hi + 1e-9


# --- D5.2 Prop. 4: stationarity on the regime simplex ------------------------


def test_collapse_entropy_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    K = 3
    rho = rng.dirichlet(np.ones(K))
    ms, Ss, As, bs, Vs = random_model(rng, K=K)
    pred = predictive(rho, ms, Ss, As, bs, Vs)
    grad = grad_collapse_entropy(pred)
    eps = 1e-6
    for s in range(K):
        # unconstrained partial: perturb rho_s without renormalising
        up, dn = rho.copy(), rho.copy()
        up[s] += eps
        dn[s] -= eps
        h = lambda r: gauss_entropy(collapse(predictive(r, ms, Ss, As, bs, Vs))[1])
        fd = (h(up) - h(dn)) / (2 * eps)
        assert np.isclose(grad[s], fd, atol=1e-5)


def test_fixed_point_is_constrained_minimum():
    rng = np.random.default_rng(3)
    ms, Ss, As, bs, Vs = random_model(rng, K=2)
    g, gamma = 0.0, 1.0
    mu_e = np.array([0.3, -0.2])
    rho, resid, ok = solve_regime_fixed_point(mu_e, ms, Ss, As, bs, Vs, g, gamma)
    assert ok and resid < 1e-12

    def lagrangian(r):
        rr = np.array([r, 1 - r])
        return gfe_hat(rr, ms, Ss, As, bs, Vs, g, gamma).total - rr @ mu_e

    grid = np.linspace(1e-4, 1 - 1e-4, 4001)
    best = grid[int(np.argmin([lagrangian(r) for r in grid]))]
    assert abs(best - rho[0]) < 1e-3  # dense-search global min == fixed point


def test_fixed_point_satisfies_its_equation():
    rng = np.random.default_rng(11)
    ms, Ss, As, bs, Vs = random_model(rng, K=4)
    mu_e = rng.normal(0, 0.5, 4)
    rho, resid, ok = solve_regime_fixed_point(mu_e, ms, Ss, As, bs, Vs, 0.0, 2.0)
    assert ok
    logits = regime_log_potential(rho, ms, Ss, As, bs, Vs, 0.0, 2.0) + mu_e
    target = np.exp(logits - logits.max())
    assert np.allclose(rho, target / target.sum(), atol=1e-10)


# --- D5.2 Prop. 5: the additive-control negative result ----------------------


def test_additive_lgssm_epistemics_constant_across_policies():
    """Koudahl 2021, GFE form: with additive controls every non-goal term of the
    slice GFE is policy-independent — purposeful exploration is impossible."""
    rng = np.random.default_rng(0)
    dz = 3
    F = 0.9 * _ortho(rng, dz)
    B = rng.normal(0, 1, (dz, 2))
    Q = _spd(rng, dz) * 0.1
    C = rng.normal(0, 1, (1, dz))
    R = np.array([[0.2]])
    m0, S0 = rng.normal(0, 1, dz), _spd(rng, dz)
    g, gamma = np.array([1.0]), np.array([[0.5]])
    ref = None
    for _ in range(5):
        controls = rng.normal(0, 2, (4, 2))  # a random 4-step policy
        terms = lgssm_policy_terms(F, B, Q, C, R, m0, S0, controls, g, gamma)
        vec = np.array(
            [[t.epistemic, t.h_state, t.h_regime, t.mi_regime_hat] for t in terms]
        )
        if ref is None:
            ref = vec
            continue
        assert np.allclose(vec, ref, atol=1e-10)  # every epistemic/entropy term equal
    # ... while the goal term does distinguish policies
    risks = [
        sum(t.goal_risk for t in lgssm_policy_terms(F, B, Q, C, R, m0, S0, u, g, gamma))
        for u in (np.zeros((4, 2)), np.ones((4, 2)))
    ]
    assert abs(risks[0] - risks[1]) > 1e-3


def _ortho(rng, d):
    q, _ = np.linalg.qr(rng.normal(0, 1, (d, d)))
    return q


def test_switching_emission_restores_policy_dependent_epistemics():
    """The positive control: an action-selected emission (probe vs idle) makes
    the epistemic term differ across actions — the door Koudahl 2021 leaves open."""
    world = ProbeWorld()
    rho = np.array([0.5, 0.5])
    ms, Ss = world.state_belief()
    probe = gfe_hat(rho, ms, Ss, *world.emission("probe"), world.g1, world.gamma1)
    idle = gfe_hat(rho, ms, Ss, *world.emission("idle"), world.g1, world.gamma1)
    assert probe.mi_regime_hat > np.log(2) > idle.mi_regime_hat == 0.0
    assert probe.epistemic - idle.epistemic > 1.0  # > 1 nat of epistemic pull
    assert probe.goal_risk > idle.goal_risk  # while risk alone prefers idle


# --- D5.3: behavioural experiment — purposeful exploration + satiation -------


def test_probe_experiment_gfe_explores_and_wins():
    res = run_probe_experiment(n_episodes=200, seed=0, epistemic=True)
    assert res["probe_rate"] == 1.0  # deterministic scorer, ~3 nat margin
    assert res["success_rate"] >= 0.97
    # satiation: after one probe the regime channel has nothing left to earn
    assert max(res["post_probe_mi"]) < 0.05 * np.log(2)


def test_probe_experiment_ablation_never_explores():
    res = run_probe_experiment(n_episodes=200, seed=0, epistemic=False)
    assert res["probe_rate"] == 0.0  # KL-control: probing is strictly dominated
    assert res["success_rate"] < 0.65  # a coin flip on the regime


# --- numerical-robustness contract: float32 + hostile inputs -----------------


def test_float32_smoke():
    rng = np.random.default_rng(1)
    ms, Ss, As, bs, Vs = (a.astype(np.float32) for a in random_model(rng, K=2))
    rho = np.array([0.5, 0.5], dtype=np.float32)
    t = gfe_hat(rho, ms, Ss, As, bs, Vs, np.float32(0.0), np.float32(1.0))
    assert np.isfinite(t.total)
    _, resid, ok = solve_regime_fixed_point(
        np.zeros(2, np.float32), ms, Ss, As, bs, Vs, 0.0, 1.0, tol=1e-5
    )
    assert ok and np.isfinite(resid)


def test_degenerate_regime_belief_is_finite_and_epistemically_flat():
    ms, Ss, As, bs, Vs = random_model(np.random.default_rng(2), K=2)
    t = gfe_hat(np.array([1.0, 0.0]), ms, Ss, As, bs, Vs, 0.0, 1.0)
    assert np.isfinite(t.total)
    assert np.isclose(t.mi_regime_hat, 0.0, atol=1e-12)  # known regime: satiated
    assert np.isclose(t.h_regime, 0.0, atol=1e-12)


def test_improper_component_covariance_is_loud():
    ms, Ss, As, bs, Vs = random_model(np.random.default_rng(4), K=2)
    Vs = Vs.copy()
    Vs[0] = -np.eye(1)  # improper emission noise
    t = gfe_hat(np.array([0.5, 0.5]), ms, Ss, As, bs, Vs, 0.0, 1.0)
    assert not np.isfinite(t.total)  # +/-inf, never a quiet number
