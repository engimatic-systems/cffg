"""GFE-TM composite (cff-fk2u): dynamics curiosity via transition P-substitution.

Pins the derivation in rules.py: the fixed point satisfies its own equation,
the parameter-novelty channel prefers unexplored (z, u) columns with the
state-MI channel held constant, goal information still flows backward through
the substituted node, and the hostile-inputs contract (AGENTS.md) holds.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cffg import rules

NEG = -1.0e30


def norm_log(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / p.sum())


def q_of(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def solve_joint(mu_in, mu_e, mu_down, b_param, learned):
    abar, h = rules._tm_joint_stats(jnp.asarray(b_param), learned)
    log_d = (jnp.asarray(mu_in)[:, None] + jnp.asarray(mu_e)[None, :]).reshape(-1)
    qj = rules._gfe_solve(abar, h, jnp.asarray(mu_down), log_d)
    rho = rules._gfe_rho(abar, h, jnp.asarray(mu_down), qj)
    return np.asarray(qj), np.asarray(rho), (np.asarray(abar), np.asarray(h), np.asarray(log_d))


def test_fixed_point_satisfies_its_equation():
    rng = np.random.default_rng(2)
    k, u = 3, 2
    counts = rng.uniform(0.5, 5.0, size=(u, k, k))
    mu_in = norm_log(rng.dirichlet(np.ones(k)))
    mu_e = norm_log(rng.dirichlet(np.ones(u)))
    mu_down = norm_log(rng.dirichlet(np.ones(k)))
    qj, rho, (_, _, log_d) = solve_joint(mu_in, mu_e, mu_down, counts, learned=True)
    np.testing.assert_allclose(qj, q_of(rho + log_d), atol=1e-8)
    assert abs(qj.sum() - 1.0) < 1e-9


def test_novelty_channel_prefers_unexplored_action():
    # Both actions have the SAME posterior-mean dynamics (uniform columns), so
    # the state-MI channel is identical; only the Dirichlet novelty channel
    # differs. Analytic gap in rho: h_known - h_unknown ≈ 0.255 nats at K=3.
    k, u = 3, 2
    counts = np.stack([
        np.full((k, k), 100.0 / k),  # action 0: well-explored uniform
        np.ones((k, k)),             # action 1: unexplored uniform
    ])
    mu_in = np.full(k, NEG)
    mu_in[0] = 0.0  # current state known: z = 0
    mu_e = norm_log(np.full(u, 1.0 / u))
    mu_down = norm_log(np.full(k, 1.0 / k))

    to_out, to_in, to_sel = rules.gfe_tm_update(
        mu_in[None], mu_e[None], mu_down[None], counts[None], k_max=k, learned=True
    )
    q_u = q_of(np.asarray(to_sel)[0, :u])
    assert q_u[1] > 0.55, f"no curiosity: q(u) = {q_u}"

    # Exploring action 1 (counts grow) must shrink the preference.
    counts_after = counts.copy()
    counts_after[1] *= 100.0
    _, _, to_sel2 = rules.gfe_tm_update(
        mu_in[None], mu_e[None], mu_down[None], counts_after[None], k_max=k, learned=True
    )
    q_u2 = q_of(np.asarray(to_sel2)[0, :u])
    assert q_u2[1] < q_u[1] - 0.04, f"novelty did not decay: {q_u} -> {q_u2}"
    np.testing.assert_allclose(q_u2[1], 0.5, atol=0.02)  # fully explored: no preference


def test_goal_information_flows_backward_through_substituted_node():
    # P-substitution must not sever pragmatic planning: with a goal-peaked
    # downstream message, the action whose dynamics REACH the goal state wins.
    k, u = 2, 2
    log_b = np.log(np.maximum(np.stack([
        np.array([[0.95, 0.95], [0.05, 0.05]]),  # action 0 -> state 0
        np.array([[0.05, 0.05], [0.95, 0.95]]),  # action 1 -> state 1 (goal)
    ]), 1e-12))
    mu_in = norm_log([0.5, 0.5])
    mu_e = norm_log([0.5, 0.5])
    mu_down = norm_log([0.05, 0.95])  # downstream wants state 1

    _, _, to_sel = rules.gfe_tm_update(
        mu_in[None], mu_e[None], mu_down[None], log_b[None], k_max=k, learned=False
    )
    q_u = q_of(np.asarray(to_sel)[0, :u])
    assert q_u[1] > 0.75, f"goal did not propagate: q(u) = {q_u}"


def test_energy_is_stationary_value():
    rng = np.random.default_rng(5)
    k, u = 3, 2
    counts = rng.uniform(0.5, 4.0, size=(u, k, k))
    mu_in = norm_log(rng.dirichlet(np.ones(k)))
    mu_e = norm_log(rng.dirichlet(np.ones(u)))
    mu_down = norm_log(rng.dirichlet(np.ones(k)))
    g = rules.gfe_tm_energy(mu_in[None], mu_e[None], mu_down[None], counts[None], learned=True)
    qj, rho, _ = solve_joint(mu_in, mu_e, mu_down, counts, learned=True)
    ref = -(qj * rho).sum() + (qj * np.log(np.maximum(qj, 1e-300))).sum()
    np.testing.assert_allclose(float(g[0]), ref, atol=1e-8)
    assert np.isfinite(float(g[0]))


# --- hostile-inputs contract (AGENTS.md) -------------------------------------


def test_sparse_support_stays_bounded():
    # Exact log-zeros in every incoming message; deterministic known B.
    k, u = 4, 3
    log_b = np.full((u, k, k), NEG)
    for a in range(u):
        for z in range(k):
            log_b[a, (z + a) % k, z] = 0.0  # deterministic shift dynamics
    mu_in = np.full(k, NEG)
    mu_in[1] = 0.0
    mu_e = np.full(u, NEG)
    mu_e[:2] = np.log(0.5)  # action 2 impossible
    mu_down = np.full(k, NEG)
    mu_down[2] = 0.0

    outs = rules.gfe_tm_update(
        mu_in[None], mu_e[None], mu_down[None], log_b[None], k_max=k, learned=False
    )
    for arr in outs:
        arr = np.asarray(arr)
        finite = arr[arr > -1e29]
        assert np.isfinite(arr).all() and (finite.size == 0 or np.abs(finite).max() < 1e4)
    g = rules.gfe_tm_energy(mu_in[None], mu_e[None], mu_down[None], log_b[None], learned=False)
    assert np.isfinite(float(g[0]))


def test_float32_stays_finite():
    rng = np.random.default_rng(9)
    k, u = 3, 2
    counts = rng.uniform(0.5, 3.0, size=(1, u, k, k))
    mu_in = norm_log(rng.dirichlet(np.ones(k)))[None]
    mu_e = norm_log(rng.dirichlet(np.ones(u)))[None]
    mu_down = norm_log(rng.dirichlet(np.ones(k)))[None]
    with jax.enable_x64(False):
        outs = rules.gfe_tm_update(
            jnp.asarray(mu_in, dtype=jnp.float32),
            jnp.asarray(mu_e, dtype=jnp.float32),
            jnp.asarray(mu_down, dtype=jnp.float32),
            jnp.asarray(counts, dtype=jnp.float32),
            k_max=k, learned=True,
        )
        assert all(bool(jnp.isfinite(o).all()) for o in outs)


def test_uncomfortable_scale_stays_finite():
    # Joint solve at K*U = 120 logits with sparse counts — an order beyond
    # the toy fixtures.
    rng = np.random.default_rng(13)
    k, u = 30, 4
    counts = rng.uniform(0.05, 2.0, size=(1, u, k, k))
    mu_in = norm_log(rng.dirichlet(np.full(k, 0.2)))[None]
    mu_e = norm_log(np.full(u, 1.0 / u))[None]
    mu_down = norm_log(rng.dirichlet(np.full(k, 0.2)))[None]
    to_out, to_in, to_sel = rules.gfe_tm_update(
        mu_in, mu_e, mu_down, counts, k_max=k, learned=True
    )
    for arr in (to_out, to_in, to_sel):
        assert bool(np.isfinite(np.asarray(arr)).all())
    # outputs are normalized log-distributions
    np.testing.assert_allclose(
        float(logsumexp(jnp.asarray(to_sel)[0, :u])), 0.0, atol=1e-6
    )
