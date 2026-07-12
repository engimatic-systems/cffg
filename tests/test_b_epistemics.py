"""GFE-TM composite (cff-fk2u): dynamics curiosity via transition P-substitution.

Pins the derivation in rules.py: the fixed point satisfies its own equation,
the parameter-novelty channel prefers unexplored (z, u) columns with the
state-MI channel held constant, substituted-edge cavities reconstruct q_w,
the ledger contains only node-local G_tm, a two-composite chain matches an
independent constrained optimizer, and the hostile-inputs contract holds.
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


def test_out_edge_cavity_reconstructs_substituted_marginal():
    rng = np.random.default_rng(41)
    k, u = 3, 2
    counts = rng.uniform(0.5, 4.0, size=(u, k, k))
    mu_in = norm_log(rng.dirichlet(np.ones(k)))
    mu_e = norm_log(rng.dirichlet(np.ones(u)))
    mu_down = norm_log([0.72, 0.19, 0.09])

    to_out, _, _ = rules.gfe_tm_update(
        mu_in[None], mu_e[None], mu_down[None], counts[None], k_max=k, learned=True
    )
    qj, _, (abar, _, _) = solve_joint(mu_in, mu_e, mu_down, counts, learned=True)
    qw = abar @ qj
    edge_q = q_of(np.asarray(to_out)[0, :k] + mu_down)
    np.testing.assert_allclose(edge_q, qw, atol=1e-9)

    # A vacuous/uniform facing message is only a constant gauge, so the
    # cavity itself reduces to the substituted predictive marginal.
    vacuous = norm_log(np.ones(k))
    vac_out, _, _ = rules.gfe_tm_update(
        mu_in[None], mu_e[None], vacuous[None], counts[None], k_max=k, learned=True
    )
    vac_qj, _, (vac_abar, _, _) = solve_joint(mu_in, mu_e, vacuous, counts, learned=True)
    np.testing.assert_allclose(q_of(np.asarray(vac_out)[0, :k]), vac_abar @ vac_qj, atol=1e-9)


def test_out_edge_cavity_preserves_exact_zero_support():
    # Every transition ends in state 0. Both q_w and the facing message have
    # exact zero support elsewhere: the masked 0/0 cavity must not resurrect it.
    k, u = 3, 1
    b = np.zeros((u, k, k))
    b[0, 0, :] = 1.0
    log_b = np.where(b > 0, np.log(np.maximum(b, np.finfo(float).tiny)), NEG)
    mu_in, mu_e = norm_log([0.2, 0.3, 0.5]), np.array([0.0])
    mu_down = np.array([0.0, NEG, NEG])
    to_out, _, _ = rules.gfe_tm_update(
        mu_in[None], mu_e[None], mu_down[None], log_b[None], k_max=k, learned=False
    )
    edge_q = q_of(np.asarray(to_out)[0, :k] + mu_down)
    np.testing.assert_array_equal(edge_q, [1.0, 0.0, 0.0])
    assert np.all(np.asarray(to_out)[0, 1:] == NEG)


def test_out_edge_cavity_float32_and_infeasible_support():
    k, u = 2, 2
    b = np.array([
        [[0.8, 0.3], [0.2, 0.7]],
        [[0.45, 0.1], [0.55, 0.9]],
    ])
    with jax.enable_x64(False):
        mi = jnp.asarray(norm_log([0.6, 0.4])[None], dtype=jnp.float32)
        me = jnp.asarray(norm_log([0.35, 0.65])[None], dtype=jnp.float32)
        md = jnp.asarray(norm_log([0.83, 0.17])[None], dtype=jnp.float32)
        lb = jnp.log(jnp.asarray(b[None], dtype=jnp.float32))
        to_out, _, _ = rules.gfe_tm_update(mi, me, md, lb, k_max=k, learned=False)
        abar, _ = rules._tm_joint_stats(lb[0], learned=False)
        qj = rules._gfe_solve(
            abar, rules._tm_joint_stats(lb[0], False)[1], md[0],
            (mi[0, :, None] + me[0, None, :]).reshape(-1),
        )
        edge_q = jax.nn.softmax(to_out[0, :k] + md[0])
        np.testing.assert_allclose(edge_q, abar @ qj, atol=2e-6)

        # A positive predictive marginal cannot be reconstructed when the
        # facing factor assigns exact zero support. Surface infeasibility.
        impossible_md = jnp.asarray([[0.0, NEG]], dtype=jnp.float32)
        bad, _, _ = rules.gfe_tm_update(mi, me, impossible_md, lb, k_max=k, learned=False)
        assert bool(jnp.isnan(bad).all())


def test_energy_is_direct_node_local_gfe():
    rng = np.random.default_rng(5)
    k, u = 3, 2
    counts = rng.uniform(0.5, 4.0, size=(u, k, k))
    mu_in = norm_log(rng.dirichlet(np.ones(k)))
    mu_e = norm_log(rng.dirichlet(np.ones(u)))
    mu_down = norm_log(rng.dirichlet(np.ones(k)))
    g = rules.gfe_tm_energy(mu_in[None], mu_e[None], mu_down[None], counts[None], learned=True)
    qj, _, (abar, h, _) = solve_joint(mu_in, mu_e, mu_down, counts, learned=True)
    qw = abar @ qj
    # Direct enumeration of Proposition 1; deliberately does not restate rho,
    # because rho contains the downstream Lagrange coupling that caused hp3y.
    ref = (qw * np.log(qw)).sum() + qj @ h + (qj * np.log(qj)).sum()
    np.testing.assert_allclose(float(g[0]), ref, atol=1e-8)
    assert np.isfinite(float(g[0]))


def test_two_composite_chain_matches_constrained_objective():
    """Independent SLSQP minimization pins both interior cavities and ledger.

    The optimizer sees only the two direct G_tm objectives and the constraint
    B1 q1 = sum_u q2. It does not call either GFE-TM update or energy rule.
    """
    from scipy.optimize import minimize

    from cffg.compiler import compile_template
    from cffg.core.ir import (
        BlobRef,
        ConstraintKind,
        ConstraintSpec,
        FactorSpec,
        Template,
        VariableSpec,
    )
    from cffg.engine import Engine

    k, u = 2, 2
    b1 = np.array([
        [[0.82, 0.28], [0.18, 0.72]],
        [[0.41, 0.12], [0.59, 0.88]],
    ])
    b2 = np.array([
        [[0.87, 0.24], [0.13, 0.76]],
        [[0.36, 0.19], [0.64, 0.81]],
    ])
    d, e1, e2, c = (
        np.array([0.64, 0.36]),
        np.array([0.57, 0.43]),
        np.array([0.38, 0.62]),
        np.array([0.23, 0.77]),
    )

    names = ("d", "e1", "e2", "c", "B1", "B2")
    t = Template(
        "two_gfe_tm",
        variables={name: VariableSpec(name, card) for name, card in (
            ("z0", k), ("u1", u), ("z1", k), ("u2", u), ("z2", k)
        )},
        factors={
            "prior": FactorSpec("prior", "categorical", {"out": "z0"}, param_blobs={"statistic": "d"}),
            "e1": FactorSpec("e1", "categorical", {"out": "u1"}, param_blobs={"statistic": "e1"}),
            "tm1": FactorSpec("tm1", "transition_mixture", {"out": "z1", "in": "z0", "selector": "u1"}, param_blobs={"matrices": "B1"}),
            "e2": FactorSpec("e2", "categorical", {"out": "u2"}, param_blobs={"statistic": "e2"}),
            "tm2": FactorSpec("tm2", "transition_mixture", {"out": "z2", "in": "z1", "selector": "u2"}, param_blobs={"matrices": "B2"}),
            "goal": FactorSpec("goal", "categorical", {"out": "z2"}, param_blobs={"statistic": "c"}),
        },
        blobs={name: BlobRef(name, f"{name}.npy") for name in names},
        constraints=[
            ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "tm1", (("substituted_role", "out"),)),
            ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "tm2", (("substituted_role", "out"),)),
        ],
    )
    engine = Engine(compile_template(t, arrays=dict(zip(names, (d, e1, e2, c, b1, b2)))))
    state, deltas = engine.run(engine.init_state(), 600, tol=1e-12, damp=0.5)
    assert deltas[-1] < 1e-10

    def stats(b):
        abar = b.transpose(1, 2, 0).reshape(k, k * u)
        h = -(b * np.log(b)).sum(axis=1).T.reshape(k * u)
        return abar, h

    a1, h1 = stats(b1)
    a2, h2 = stats(b2)
    ld, le1, le2, lc = map(np.log, (d, e1, e2, c))

    def split(x):
        return x[: k * u], x[k * u :]

    def terms(q1, q2):
        w1, r, w2 = a1 @ q1, q2.reshape(k, u).sum(1), a2 @ q2
        g1 = (w1 * np.log(w1)).sum() + q1 @ h1 + (q1 * np.log(q1)).sum()
        g2 = (w2 * np.log(w2)).sum() + q2 @ h2 + (q2 * np.log(q2)).sum()
        total = (
            g1 + g2 - (r * np.log(r)).sum()
            - q1.reshape(k, u).sum(1) @ ld - q1.reshape(k, u).sum(0) @ le1
            - q2.reshape(k, u).sum(0) @ le2 - w2 @ lc
        )
        return total, (g1, g2, w1, r, w2)

    q10 = np.full(k * u, 1.0 / (k * u))
    w10 = a1 @ q10
    q20 = (w10[:, None] * np.full((k, u), 1.0 / u)).reshape(-1)
    result = minimize(
        lambda x: terms(*split(x))[0],
        np.concatenate([q10, q20]),
        method="SLSQP",
        bounds=[(np.finfo(float).eps, 1.0)] * (2 * k * u),
        constraints=[
            {"type": "eq", "fun": lambda x: split(x)[0].sum() - 1.0},
            {"type": "eq", "fun": lambda x: split(x)[1].sum() - 1.0},
            {"type": "eq", "fun": lambda x: (a1 @ split(x)[0] - split(x)[1].reshape(k, u).sum(1))[0]},
        ],
        options={"ftol": 1e-13, "maxiter": 2000},
    )
    assert result.success, result.message
    q1, q2 = split(result.x)
    direct_total, (g1, g2, w1, r, w2) = terms(q1, q2)
    np.testing.assert_allclose(w1, r, atol=2e-8)

    plan = engine.arena.gfe_tm_groups[0]
    bp = state.params[plan.name + ":B"]
    runtime_q = {}
    for name in ("tm1", "tm2"):
        i = plan.members.index(name)
        md = state.messages[plan.gather[i, 0], :k]
        mi = state.messages[plan.gather[i, 1], :k]
        me = state.messages[plan.gather[i, 2], :u]
        abar, h = rules._tm_joint_stats(bp[i], learned=False)
        runtime_q[name] = np.asarray(
            rules._gfe_solve(abar, h, md, (mi[:, None] + me[None, :]).reshape(-1))
        )
    np.testing.assert_allclose(runtime_q["tm1"], q1, atol=2e-7)
    np.testing.assert_allclose(runtime_q["tm2"], q2, atol=2e-7)

    marginals = engine.marginals(state)
    np.testing.assert_allclose(marginals["z0"], q1.reshape(k, u).sum(1), atol=2e-7)
    np.testing.assert_allclose(marginals["u1"], q1.reshape(k, u).sum(0), atol=2e-7)
    np.testing.assert_allclose(marginals["z1"], w1, atol=2e-7)
    np.testing.assert_allclose(marginals["u2"], q2.reshape(k, u).sum(0), atol=2e-7)
    np.testing.assert_allclose(marginals["z2"], w2, atol=2e-7)

    # Reconstruct all six composite messages from the optimizer's stationary
    # beliefs, independently of gfe_tm_update, and compare up to log gauge.
    def norm(x):
        return x - np.logaddexp.reduce(x)

    rho2 = (a2.T @ (lc - np.log(w2)) - h2).reshape(k, u)
    back = norm(np.logaddexp.reduce(rho2 + le2[None, :], axis=1))
    forward = norm(np.log(w1) - back)
    rho1 = (a1.T @ (back - np.log(w1)) - h1).reshape(k, u)
    expected = {
        ("tm1", 0): forward,
        ("tm1", 1): norm(np.logaddexp.reduce(rho1 + le1[None, :], axis=1)),
        ("tm1", 2): norm(np.logaddexp.reduce(rho1 + ld[:, None], axis=0)),
        ("tm2", 0): norm(np.log(w2) - lc),
        ("tm2", 1): back,
        ("tm2", 2): norm(np.logaddexp.reduce(rho2 + forward[:, None], axis=0)),
    }
    for (name, role), want in expected.items():
        i = plan.members.index(name)
        got = np.asarray(state.messages[plan.scatter[i, role], : len(want)])
        np.testing.assert_allclose(got, want, atol=3e-7)
    np.testing.assert_allclose(q_of(forward + back), w1, atol=2e-8)

    np.testing.assert_allclose(
        rules.gfe_tm_energy(
            state.messages[plan.gather[:, 1], :k],
            state.messages[plan.gather[:, 2], :u],
            state.messages[plan.gather[:, 0], :k],
            bp,
            learned=False,
        ),
        [g1, g2],
        atol=2e-7,
    )
    np.testing.assert_allclose(engine.bfe(state), direct_total, atol=3e-7)


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
        g = rules.gfe_tm_energy(
            jnp.asarray(mu_in, dtype=jnp.float32),
            jnp.asarray(mu_e, dtype=jnp.float32),
            jnp.asarray(mu_down, dtype=jnp.float32),
            jnp.asarray(counts, dtype=jnp.float32),
            learned=True,
        )
        assert bool(jnp.isfinite(g).all())


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
    g = rules.gfe_tm_energy(mu_in, mu_e, mu_down, counts, learned=True)
    assert bool(np.isfinite(np.asarray(g)).all())
    # outputs are normalized log-distributions
    np.testing.assert_allclose(
        float(logsumexp(jnp.asarray(to_sel)[0, :u])), 0.0, atol=1e-6
    )


def dirichlet_kl(beta, alpha):
    from jax.scipy.special import digamma, gammaln

    b0, a0 = beta.sum(), alpha.sum()
    return float(
        gammaln(b0) - gammaln(a0)
        - (gammaln(jnp.asarray(beta)).sum() - gammaln(jnp.asarray(alpha)).sum())
        + ((beta - alpha) * (digamma(jnp.asarray(beta)) - digamma(b0))).sum()
    )


def test_novelty_identity():
    """Executable proof of the derivation's Prop 3: the novelty term
    I(z'; B | z,u) = H[E[B]-column] − ĥ equals the EXPECTED one-step Bayesian
    update magnitude E_{z'~B̄}[ KL( Dir(α + e_{z'}) ‖ Dir(α) ) ] — exact by
    Dirichlet-categorical conjugacy and the MI = expected-posterior-KL
    identity. This is what pymdp's B-novelty bonus linearizes."""
    rng = np.random.default_rng(21)
    for _ in range(5):
        alpha = rng.uniform(0.3, 8.0, size=4)
        counts = alpha.reshape(1, 4, 1)  # one action, one column: [U=1, K'=4, K=1]
        bbar, h = rules._tm_dirichlet_stats(jnp.asarray(counts))
        bbar, h = np.asarray(bbar)[0, :, 0], float(np.asarray(h)[0, 0])
        novelty = -(bbar * np.log(bbar)).sum() - h  # H[B̄] − ĥ
        expected_kl = sum(
            bbar[i] * dirichlet_kl(alpha + np.eye(4)[i], alpha) for i in range(4)
        )
        np.testing.assert_allclose(novelty, expected_kl, rtol=1e-8)


# --- end-to-end: compiler + engine + agent ------------------------------------

KC, UC = 4, 2


def corridor_arrays(unknown_action=1, known_strength=60.0):
    acc = 0.9
    a = np.full((KC, KC), (1 - acc) / (KC - 1))
    np.fill_diagonal(a, acc)
    counts = np.full((UC, KC, KC), known_strength / KC)
    counts[unknown_action] = 1.0  # same posterior MEAN, different certainty
    return a, counts


def build_curiosity_engine(epistemic: bool):
    from cffg.compiler import compile_template
    from cffg.engine import Engine
    from cffg.models.rollout import build_rollout

    a, counts = corridor_arrays()
    t, arrays = build_rollout(
        "curious", steps=2, past=0,
        d=np.full(KC, 1.0 / KC), A=a, c=np.full(KC, 1.0 / KC),
        e=np.full(UC, 1.0 / UC), Bs=np.full((UC, KC, KC), 1.0 / KC),
        gfe=False, epistemic_b=epistemic,
    )
    arrays["Bc"] = counts
    return Engine(compile_template(t, arrays=arrays))


def test_engine_prefers_unknown_dynamics():
    engine = build_curiosity_engine(epistemic=True)
    state, _ = engine.run(engine.init_state(), 40, tol=1e-11)
    q_u = engine.marginals(state)["u1"]
    assert q_u[1] > 0.55, f"no curiosity through the full engine: {q_u}"
    assert np.isfinite(engine.bfe(state))

    ablation = build_curiosity_engine(epistemic=False)
    st2, _ = ablation.run(ablation.init_state(), 40, tol=1e-11)
    q_u2 = ablation.marginals(st2)["u1"]
    np.testing.assert_allclose(q_u2, 0.5, atol=1e-6)  # plain BP: no preference


def test_agent_curiosity_decays_with_experience():
    """Both actions have IDENTICAL true dynamics (right-shift with slip), so
    once both are learned the state-MI channel is symmetric and only the
    novelty channel separates them: the planner's q(u1 = unknown action)
    starts elevated and must decay toward 0.5 as counts accumulate. The
    ablation (no composite) is flat at 0.5 throughout."""
    from cffg.agent import RollingAgent
    from cffg.models.rollout import build_rollout

    acc = 0.9
    a = np.full((KC, KC), (1 - acc) / (KC - 1))
    np.fill_diagonal(a, acc)
    # Uniform TRUE dynamics for both actions: learned means stay uniform, so
    # the state-MI channel is symmetric throughout and only count-novelty
    # separates the actions.
    true_bs = np.full((UC, KC, KC), 1.0 / KC)

    def builder(**kw):
        return build_rollout(
            "corridor", d=np.full(KC, 1.0 / KC), A=a, c=np.full(KC, 1.0 / KC),
            e=np.full(UC, 1.0 / UC), Bs=np.full((UC, KC, KC), 1.0 / KC),
            gfe=False, **kw,
        )

    counts0 = np.stack([np.full((KC, KC), 25.0 / KC), np.full((KC, KC), 0.3)])
    # action 0: alpha0 = 25 per column (novelty ~0.08 nats); action 1:
    # alpha0 = 1.2 (novelty ~0.78 nats) -> ~0.7-nat initial curiosity gap.

    def run_agent(epistemic):
        rng = np.random.default_rng(4)
        agent = RollingAgent(
            builder, horizon=2, learn_b=True, initial_b_counts=counts0.copy(),
            epistemic_b=epistemic, n_sweeps=50, select="map", seed=4,
        )
        z, q1_trace = int(rng.integers(KC)), []
        for _ in range(50):
            st = agent._with_params(agent.plan)
            st, _ = agent.plan.run(st, agent.n_sweeps, tol=1e-10, damp=agent.damp)
            q = np.asarray(agent.plan.marginals(st)["u1"])
            q1_trace.append(float(q[1] / q.sum()))
            u = int(np.argmax(q))
            z = int(rng.choice(KC, p=true_bs[u, :, z]))
            agent.observe(u, int(rng.choice(KC, p=a[:, z])))
        return np.asarray(q1_trace)

    q1 = run_agent(True)
    assert q1[0] > 0.6, f"no initial curiosity: q1[0] = {q1[0]:.3f}"
    early, late = q1[:10].mean(), q1[-10:].mean()
    assert late < early - 0.05, f"curiosity did not decay: {early:.3f} -> {late:.3f}"
    assert late < 0.57, f"late preference should approach indifference: {late:.3f}"

    q1_flat = run_agent(False)
    # Ablation is never curious — in fact plain BP with E[log B] digamma
    # point estimates is ambiguity-AVERSE (low-count columns are
    # sub-stochastic in expectation and read as evidence against the
    # action), the classic failure the composite exists to correct.
    assert (q1_flat <= 0.501).all(), f"ablation showed curiosity: max {q1_flat.max():.3f}"
    assert q1_flat[0] < 0.5, f"expected ambiguity aversion, got {q1_flat[0]:.3f}"


def test_epistemic_template_renders(viz_gallery):
    import xml.etree.ElementTree as ET

    from cffg.models.rollout import build_rollout
    from cffg.viz import render_svg

    a, _ = corridor_arrays()
    t, _ = build_rollout(
        "curious_viz", steps=2, past=0,
        d=np.full(KC, 1.0 / KC), A=a, c=np.full(KC, 1.0 / KC),
        e=np.full(UC, 1.0 / UC), Bs=np.full((UC, KC, KC), 1.0 / KC),
        gfe=True, epistemic_b=True,
    )
    svg = render_svg(t, level=4)
    ET.fromstring(svg)
    viz_gallery(svg, tag="dynamics-curiosity rollout (P-sub on TM + obs)")
