"""HMM tier: an unrolled+composed chain must reproduce exact forward-backward
smoothing, and its BFE must equal the negative log-likelihood. Plus the colony
contract: vmapped batches equal sequential runs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cffg.compiler import compile_template
from cffg.core.compose import compose, unroll
from cffg.engine import Engine
from conftest import K, M, make_hmm_slice
from test_compose import make_state_prior

rng = np.random.default_rng(11)
T = 4


def col_stochastic(rows, cols):
    m = rng.uniform(0.1, 1.0, size=(rows, cols))
    return m / m.sum(axis=0, keepdims=True)


D = rng.dirichlet(np.ones(K))
B = col_stochastic(K, K)  # p(z_t | z_{t-1})
A = col_stochastic(M, K)  # p(x_t | z_t)


def build_engine():
    rolled = unroll(make_hmm_slice(), T, {"tail": "head"})
    model = compose(
        make_state_prior(), rolled, {"attach": "t0/head"},
        name="hmm", scope_a="prior", scope_b="chain",
    )
    arrays = {"prior/d": D}
    for t in range(T):
        arrays[f"chain/t{t}/A"] = A
        arrays[f"chain/t{t}/B"] = B
    return Engine(compile_template(model, arrays=arrays))


def forward_backward(obs):
    """Exact smoother for z_0..z_T (z_0 prior-only, x_t emitted from z_t, t>=1).
    Returns smoothed marginals [T+1, K] and the log-likelihood."""
    em = [np.ones(K) if o is None else A[o] for o in obs]
    alphas, log_cs = [D.copy()], []
    for t in range(T):
        a = em[t] * (B @ alphas[-1])
        c = a.sum()
        alphas.append(a / c)
        log_cs.append(np.log(c))
    betas = [np.ones(K)]
    for t in reversed(range(T)):
        b = B.T @ (em[t] * betas[0])
        betas.insert(0, b / b.sum())
    smoothed = np.array([a * b for a, b in zip(alphas, betas)])
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return smoothed, float(np.sum(log_cs))


def latent_names():
    yield "prior/z0"  # z_0 (bond kept the prior side's variable)
    for t in range(T):
        yield f"chain/t{t}/z_mid"  # z_{t+1}


def run_case(obs):
    engine = build_engine()
    ports = {f"chain/t{t}/sense": obs[t] for t in range(T)}
    state = engine.bind_data(engine.init_state(), ports)
    state, deltas = engine.run(
        state, 3 * T + 10, tol=1e-14
    ).require_converged_with_trace()
    assert deltas[-1] < 1e-12
    return engine, state


def test_smoothed_marginals_match_forward_backward():
    obs = [1, 0, 1, 1]
    engine, state = run_case(obs)
    q = engine.marginals(state)
    smoothed, _ = forward_backward(obs)
    for k, name in enumerate(latent_names()):
        np.testing.assert_allclose(q[name], smoothed[k], atol=1e-10, err_msg=name)
    # equality arms agree
    np.testing.assert_allclose(q["chain/t2/z_mid"], q["chain/t2/z_obs"], atol=1e-12)
    np.testing.assert_allclose(q["chain/t2/z_mid"], q["chain/t2/z_out"], atol=1e-12)


def test_bfe_equals_negative_loglikelihood():
    obs = [1, 0, 1, 1]
    engine, state = run_case(obs)
    _, loglik = forward_backward(obs)
    np.testing.assert_allclose(engine.bfe(state), -loglik, atol=1e-9)


def test_partial_observations():
    obs = [1, None, None, 0]
    engine, state = run_case(obs)
    q = engine.marginals(state)
    smoothed, loglik = forward_backward(obs)
    for k, name in enumerate(latent_names()):
        np.testing.assert_allclose(q[name], smoothed[k], atol=1e-10, err_msg=name)
    # predicted missing observation = A @ smoothed latent
    pred = A @ smoothed[2]
    np.testing.assert_allclose(q["chain/t1/x"], pred / pred.sum(), atol=1e-10)
    np.testing.assert_allclose(engine.bfe(state), -loglik, atol=1e-9)


def test_vmap_batch_equals_sequential():
    engine = build_engine()
    sequences = [[1, 0, 1, 1], [0, 0, None, 1], [1, 1, 1, None]]
    states = [
        engine.bind_data(engine.init_state(), {f"chain/t{t}/sense": s[t] for t in range(T)})
        for s in sequences
    ]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

    step_v = eqx.filter_vmap(
        lambda st: engine._step(
            st.messages, st.g_eta, st.g_lam, st.g_obs, st.g_mask,
            st.ef_eta, st.ef_obs, st.ef_mask, st.params,
        )[0]
    )
    msgs = batched.messages
    for _ in range(3 * T + 10):
        batched = eqx.tree_at(lambda s: s.messages, batched, step_v(batched))
    msgs = batched.messages

    for b, s in enumerate(sequences):
        solo = engine.run(states[b], 3 * T + 10).state
        np.testing.assert_allclose(np.asarray(msgs[b]), np.asarray(solo.messages), atol=1e-10)
