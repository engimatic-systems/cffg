"""The GFE composite: Newton-solved backward state message (Part II Cor. 1).

Ground truth is brute force: for K=2 the local objective
    J(z) = −zᵀ(ρ(z) + log d) − H[z],  ρ(z) = Aᵀ(log c − log Az) − h(A)
is a 1-D function over the simplex; its grid argmin must match the engine's
z*, and the engine's total BFE must equal J(z*) (prior node + z-edge entropy
+ composite term collapse to exactly J — derivation in ticket cff-007q).
"""

import numpy as np

from cffg.compiler import compile_template
from cffg.core import (
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    Template,
    VariableSpec,
)
from cffg.core.ir import BlobRef
from cffg.engine import Engine

K = 2
M = 2


def make_model(A, c, d, p_substitute=True) -> Template:
    t = Template(
        name="gfe_unit",
        variables={"z": VariableSpec("z", K), "x": VariableSpec("x", M)},
        factors={
            "prior": FactorSpec("prior", "categorical", edges={"out": "z"}, param_blobs={"statistic": "d"}),
            "obs": FactorSpec("obs", "matrix_cat", edges={"out": "x", "in": "z"}, param_blobs={"matrix": "A"}),
            "goal": FactorSpec("goal", "categorical", edges={"out": "x"}, param_blobs={"statistic": "c"}),
        },
        blobs={"d": BlobRef("d", "d.npy"), "A": BlobRef("A", "A.npy"), "c": BlobRef("c", "c.npy")},
    )
    if p_substitute:
        t.constraints.append(
            ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "obs", (("substituted_role", "out"),))
        )
    return t


def solve(A, c, d, p_substitute=True, iters=8):
    engine = Engine(compile_template(make_model(A, c, d, p_substitute), arrays={"A": A, "c": c, "d": d}))
    state, deltas = engine.run(engine.init_state(), iters, tol=1e-14)
    assert deltas[-1] < 1e-10
    return engine, state


def rho_np(A, c, zbar):
    h = -(A * np.log(np.maximum(A, 1e-300))).sum(axis=0)
    return A.T @ (np.log(c) - np.log(A @ zbar)) - h


def brute_force_zstar(A, c, d):
    p = np.linspace(1e-9, 1 - 1e-9, 200_001)
    z = np.stack([p, 1 - p])  # [K, N]
    h_col = -(A * np.log(np.maximum(A, 1e-300))).sum(axis=0)
    log_qx = np.log(A @ z)  # [M, N]
    rho = A.T @ (np.log(c)[:, None] - log_qx) - h_col[:, None]  # [K, N]
    j = -(z * (rho + np.log(d)[:, None])).sum(0) + (z * np.log(z)).sum(0)
    i = int(np.argmin(j))
    return z[:, i], float(j[i])


# One informative column, one maximally ambiguous column.
A_EPI = np.array([[0.98, 0.5], [0.02, 0.5]])
UNIF = np.array([0.5, 0.5])


def test_newton_matches_brute_force_argmin():
    engine, state = solve(A_EPI, UNIF, UNIF)
    z_ref, _ = brute_force_zstar(A_EPI, UNIF, UNIF)
    np.testing.assert_allclose(engine.marginals(state)["z"], z_ref, atol=1e-4)


def test_ambiguity_aversion_and_bfe_ablation():
    """Uniform goal, uniform prior: the epistemic term alone must pull the
    agent toward the unambiguous state — and removing the single P-bead
    (Part II §7.1 ablation) must collapse the preference to exactly d."""
    engine, state = solve(A_EPI, UNIF, UNIF)
    q = engine.marginals(state)["z"]
    # The pull is modest for a single node (the quantitative anchor is the
    # brute-force parity test); here we assert its direction and that it is
    # strictly epistemic — the BFE ablation below shows 0.5 exactly.
    assert q[0] > 0.51, f"no ambiguity aversion: {q}"

    engine_b, state_b = solve(A_EPI, UNIF, UNIF, p_substitute=False, iters=10)
    np.testing.assert_allclose(engine_b.marginals(state_b)["z"], UNIF, atol=1e-10)


def test_bfe_equals_local_objective_minimum():
    engine, state = solve(A_EPI, UNIF, UNIF)
    _, j_min = brute_force_zstar(A_EPI, UNIF, UNIF)
    np.testing.assert_allclose(engine.bfe(state), j_min, atol=1e-6)


def test_goal_seeking_with_informative_observations():
    """Biased goal + near-identity A: extrinsic value tilts the state."""
    a = np.array([[0.99, 0.01], [0.01, 0.99]])
    c = np.array([0.9, 0.1])
    engine, state = solve(a, c, UNIF)
    q = engine.marginals(state)["z"]
    z_ref, j_min = brute_force_zstar(a, c, UNIF)
    np.testing.assert_allclose(q, z_ref, atol=1e-4)
    assert q[0] > 0.5
    np.testing.assert_allclose(engine.bfe(state), j_min, atol=1e-6)


def test_uninformative_A_leaves_prior_untouched():
    """If observations carry no state information, no amount of goal utility
    should move the state posterior."""
    a = np.array([[0.5, 0.5], [0.5, 0.5]])
    c = np.array([0.9, 0.1])
    d = np.array([0.3, 0.7])
    engine, state = solve(a, c, d)
    np.testing.assert_allclose(engine.marginals(state)["z"], d, atol=1e-8)


def test_posterior_predictive_on_x():
    engine, state = solve(A_EPI, UNIF, UNIF)
    zbar = engine.marginals(state)["z"]
    np.testing.assert_allclose(engine.marginals(state)["x"], A_EPI @ zbar, atol=1e-8)


def test_learned_composite_recovers_fixed_at_high_counts():
    """Dirichlet mode with enormous counts must converge to the fixed-A
    composite: Ā → A and the closed-form E[A log A] pseudo-entropy → h(A)."""
    from cffg.core import ConstraintKind, ConstraintSpec

    t = make_model(A_EPI, UNIF, UNIF, p_substitute=False)
    t.constraints.append(
        ConstraintSpec(
            ConstraintKind.P_SUBSTITUTION, "obs",
            (("parameter_uncertainty", "dirichlet"), ("substituted_role", "out")),
        )
    )
    counts = A_EPI * 1e7 + 1e-3
    engine = Engine(compile_template(t, arrays={"A": counts, "c": UNIF, "d": UNIF}))
    state, deltas = engine.run(engine.init_state(), 8, tol=1e-14)
    assert deltas[-1] < 1e-10

    fixed_engine, fixed_state = solve(A_EPI, UNIF, UNIF)
    np.testing.assert_allclose(
        engine.marginals(state)["z"], fixed_engine.marginals(fixed_state)["z"], atol=1e-6
    )
    np.testing.assert_allclose(engine.bfe(state), fixed_engine.bfe(fixed_state), atol=1e-5)


def test_learned_composite_novelty_direction():
    """Uncertain-but-maybe-sharp beats confidently-uniform: with a uniform
    goal and prior, the agent should prefer the state whose column is
    unexplored over the one it knows is 50/50."""
    from cffg.core import ConstraintKind, ConstraintSpec

    t = make_model(A_EPI, UNIF, UNIF, p_substitute=False)
    t.constraints.append(
        ConstraintSpec(
            ConstraintKind.P_SUBSTITUTION, "obs",
            (("parameter_uncertainty", "dirichlet"), ("substituted_role", "out")),
        )
    )
    counts = np.array([[50.0, 0.3], [50.0, 0.3]])  # state 0: known 50/50; state 1: unexplored
    engine = Engine(compile_template(t, arrays={"A": counts, "c": UNIF, "d": UNIF}))
    state, _ = engine.run(engine.init_state(), 8, tol=1e-14)
    q = engine.marginals(state)["z"]
    assert q[1] > 0.55, f"no parameter-novelty pull toward the unexplored state: {q}"


# --- sparse support (cff-i45j) -----------------------------------------------


def test_sparse_support_prior_stays_finite():
    """A prior with an exact zero (deterministic transitions upstream produce
    these) used to poison the composite: mu2 = log(zbar) − log_d hits +1e30
    at the impossible state and its logsumexp normalization destroys the
    whole message, cascading to NaN on larger models. The impossible state
    must simply stay impossible, everything finite."""
    d = np.array([1.0, 0.0])
    engine = Engine(
        compile_template(make_model(A_EPI, UNIF, d), arrays={"A": A_EPI, "c": UNIF, "d": d})
    )
    state, _ = engine.run(engine.init_state(), 10)
    assert bool(np.isfinite(state.messages).all()), "messages went non-finite"
    q = engine.marginals(state)
    np.testing.assert_allclose(q["z"], [1.0, 0.0], atol=1e-9)  # z pinned by d
    np.testing.assert_allclose(q["x"], A_EPI[:, 0], atol=1e-6)  # predictive = A col 0
    assert np.isfinite(engine.bfe(state))


def test_run_ignores_nonfinite_deltas():
    """A diverging model must never satisfy tol: non-finite deltas (inf/nan)
    don't count as convergence — run exhausts its budget instead of returning
    a transient as if converged."""
    import jax.numpy as jnp

    from cffg.engine import _replace

    engine, state = solve(A_EPI, UNIF, UNIF)
    poisoned = _replace(state, messages=state.messages.at[0, 0].set(jnp.inf))
    _, deltas = engine.run(poisoned, 5, tol=1e-6)
    assert len(deltas) == 5, f"stopped early on non-finite delta: {deltas}"


def test_sparse_support_messages_stay_bounded_at_scale():
    """K=40 with an exact-zero prior support: pre-fix, mu2 hit +1e30 at the
    impossible states (finite − NEG), and those entries leak probability into
    any downstream update whose log-factor is not also NEG. The rho-form
    message is bounded by the model's own log-scales."""
    rng = np.random.default_rng(0)
    k, m = 40, 8
    a = rng.dirichlet(np.full(m, 0.3), size=k).T
    d = np.zeros(k)
    d[3] = 1.0
    c = np.full(m, 1 / m)
    t = Template(
        "big",
        variables={"z": VariableSpec("z", k), "x": VariableSpec("x", m)},
        factors={
            "prior": FactorSpec("prior", "categorical", edges={"out": "z"}, param_blobs={"statistic": "d"}),
            "obs": FactorSpec("obs", "matrix_cat", edges={"out": "x", "in": "z"}, param_blobs={"matrix": "A"}),
            "goal": FactorSpec("goal", "categorical", edges={"out": "x"}, param_blobs={"statistic": "c"}),
        },
        blobs={n: BlobRef(n, n) for n in ("d", "A", "c")},
    )
    t.constraints.append(
        ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "obs", (("substituted_role", "out"),))
    )
    engine = Engine(compile_template(t, arrays={"A": a, "c": c, "d": d}))
    state, _ = engine.run(engine.init_state(), 10)
    msgs = np.asarray(state.messages)
    finite = msgs[msgs > -1e29]  # NEG padding/zeros are legitimate
    assert np.abs(finite).max() < 1e4, f"message magnitude {np.abs(finite).max():.3g}"
    np.testing.assert_allclose(engine.marginals(state)["z"][3], 1.0, atol=1e-9)
    np.testing.assert_allclose(engine.marginals(state)["x"], a[:, 3], atol=1e-6)
