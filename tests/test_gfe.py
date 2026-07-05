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
