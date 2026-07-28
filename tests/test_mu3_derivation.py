"""Independent executable anchors for DERIVATIONS D6b's mu3 contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp, xlogy
from scipy.special import betaln, roots_legendre


MC_TOL = 1.5e-2  # preregistered binary-oracle tolerance at 2**15 particles


def _mu3_log(a, abar, z, c):
    """Equation D6b.2, vectorized over any leading sample dimensions."""
    a = jnp.asarray(a)
    abar = jnp.asarray(abar, dtype=a.dtype)
    z = jnp.asarray(z, dtype=a.dtype)
    c = jnp.asarray(c, dtype=a.dtype)
    sample_predictive = jnp.einsum("...mk,k->...m", a, z)
    marginal_predictive = abar @ z
    entropy_part = jnp.sum(xlogy(a * z, a), axis=(-2, -1))
    ratio_part = jnp.sum(
        xlogy(sample_predictive, c)
        - xlogy(sample_predictive, marginal_predictive),
        axis=-1,
    )
    return entropy_part + ratio_part


def _column_entropy(a):
    return -jnp.sum(xlogy(a, a), axis=-2)


def _draw_base(key, alpha, sample_count):
    alpha = jnp.asarray(alpha)
    if (
        not bool(jnp.all(jnp.isfinite(alpha)))
        or not bool(jnp.all(alpha > 0))
    ):
        raise ValueError("Dirichlet base concentrations must be finite and > 0")
    next_key, sample_key = jax.random.split(key)
    column_keys = jax.random.split(sample_key, alpha.shape[1])
    columns = [
        jax.random.dirichlet(column_keys[k], alpha[:, k], shape=(sample_count,))
        for k in range(alpha.shape[1])
    ]
    return next_key, jnp.stack(columns, axis=-1)


def _normalize_log_weights(log_weights, min_ess_fraction=0.0):
    log_weights = jnp.asarray(log_weights)
    if bool(jnp.any(jnp.isnan(log_weights) | jnp.isposinf(log_weights))):
        raise RuntimeError("mu3 importance weights diverged")
    finite = jnp.isfinite(log_weights)
    if not bool(jnp.any(finite)):
        raise RuntimeError("all mu3 importance weights are non-finite")
    log_norm = logsumexp(log_weights)
    normalized = jnp.where(finite, jnp.exp(log_weights - log_norm), 0.0)
    ess = 1.0 / jnp.sum(normalized**2)
    status = (
        "converged"
        if float(ess / log_weights.size) >= min_ess_fraction
        else "budget_exhausted"
    )
    log_z = log_norm - jnp.log(jnp.asarray(log_weights.size, log_weights.dtype))
    return normalized, log_z, ess, status


def _importance_step(samples, abar, zs, cs, min_ess_fraction=0.0):
    per_slice = jnp.stack(
        [_mu3_log(samples, abar, z, c) for z, c in zip(zs, cs)], axis=0
    )
    combined = jnp.sum(per_slice, axis=0)
    weights, log_z, ess, status = _normalize_log_weights(
        combined, min_ess_fraction
    )
    next_abar = jnp.einsum("s,smk->mk", weights, samples)
    hbar = jnp.einsum("s,sk->k", weights, _column_entropy(samples))
    rhos = jnp.stack(
        [
            next_abar.T @ (jnp.log(c) - jnp.log(next_abar @ z)) - hbar
            for z, c in zip(zs, cs)
        ]
    )
    return {
        "per_slice": per_slice,
        "combined": combined,
        "weights": weights,
        "log_z": log_z,
        "ess": ess,
        "status": status,
        "abar": next_abar,
        "hbar": hbar,
        "rhos": rhos,
        "expected_phi": jnp.stack(
            [z @ rho for z, rho in zip(zs, rhos)]
        ),
    }


def _binary_grid(alpha, z, c, order=96):
    """Product Gauss-Legendre oracle over two independent Beta columns."""
    nodes, quad_weights = roots_legendre(order)
    values = (nodes + 1.0) / 2.0
    quad_weights = quad_weights / 2.0
    left, right = np.meshgrid(values, values, indexing="ij")
    a = np.stack(
        [
            np.stack([left, right], axis=-1),
            np.stack([1.0 - left, 1.0 - right], axis=-1),
        ],
        axis=-2,
    )
    log_base = (
        (alpha[0, 0] - 1) * np.log(left)
        + (alpha[1, 0] - 1) * np.log1p(-left)
        - betaln(alpha[0, 0], alpha[1, 0])
        + (alpha[0, 1] - 1) * np.log(right)
        + (alpha[1, 1] - 1) * np.log1p(-right)
        - betaln(alpha[0, 1], alpha[1, 1])
    )
    base_weight = np.outer(quad_weights, quad_weights) * np.exp(log_base)
    base_weight /= base_weight.sum()

    # Solve the empirical/quadrature version of Abar = E_q*[A].
    abar = alpha / alpha.sum(axis=0, keepdims=True)
    for _ in range(200):
        phi = np.asarray(_mu3_log(a, abar, z, c))
        log_weight = np.log(base_weight) + phi
        weight = np.exp(log_weight - np.max(log_weight))
        weight /= weight.sum()
        nxt = np.einsum("ij,ijmk->mk", weight, a)
        if np.max(np.abs(nxt - abar)) < 1e-14:
            abar = nxt
            break
        abar = nxt
    else:  # pragma: no cover - oracle failure is intentionally loud
        raise AssertionError("binary quadrature fixed point did not converge")

    phi = np.asarray(_mu3_log(a, abar, z, c))
    log_weight = np.log(base_weight) + phi
    log_z = np.log(np.exp(log_weight - np.max(log_weight)).sum()) + np.max(
        log_weight
    )
    weight = np.exp(log_weight - log_z)
    hbar = np.einsum(
        "ij,ijk->k", weight, np.asarray(_column_entropy(a))
    )
    rho = abar.T @ (np.log(c) - np.log(abar @ z)) - hbar
    return {
        "abar": abar,
        "hbar": hbar,
        "rho": rho,
        "expected_phi": z @ rho,
        "log_z": log_z,
    }


def test_mu3_log_density_matches_expanded_formula_and_is_not_dirichlet():
    abar = np.array([[0.65, 0.25], [0.35, 0.75]])
    z = np.array([0.4, 0.6])
    c = np.array([0.8, 0.2])
    a = np.array([[0.2, 0.9], [0.8, 0.1]])
    h = -np.sum(np.where(a > 0, a * np.log(a), 0.0), axis=0)
    r = np.log(c) - np.log(abar @ z)
    formal = z @ (a.T @ r - h)
    np.testing.assert_allclose(_mu3_log(a, abar, z, c), formal, atol=1e-14)

    # No constant plus fixed coefficients on log A fits the functional score
    # on an open product of binary simplexes.
    values = np.array(
        [[0.10, 0.20], [0.20, 0.35], [0.35, 0.15], [0.45, 0.70],
         [0.60, 0.25], [0.70, 0.55], [0.80, 0.40], [0.90, 0.75]]
    )
    matrices = np.stack(
        [np.stack([values, 1.0 - values], axis=1)[:, :, k] for k in range(2)],
        axis=-1,
    )
    design = np.column_stack(
        [np.ones(len(matrices)), np.log(matrices).reshape(len(matrices), -1)]
    )
    score = np.asarray(_mu3_log(matrices, abar, z, c))
    _, residuals, _, _ = np.linalg.lstsq(design, score, rcond=None)
    assert residuals.item() > 1e-5


def test_mu3_importance_sampling_matches_binary_quadrature():
    alpha = np.array([[2.2, 1.5], [1.8, 2.5]])
    z = np.array([0.35, 0.65])
    c = np.array([0.75, 0.25])
    oracle = _binary_grid(alpha, z, c)

    _, samples = _draw_base(
        jax.random.PRNGKey(19), jnp.asarray(alpha), 2**15
    )
    abar = jnp.asarray(alpha / alpha.sum(axis=0, keepdims=True))
    result = None
    for _ in range(100):
        result = _importance_step(samples, abar, (z,), (c,))
        if float(jnp.max(jnp.abs(result["abar"] - abar))) < 1e-10:
            break
        abar = result["abar"]
    assert result is not None
    np.testing.assert_allclose(result["abar"], oracle["abar"], atol=MC_TOL)
    np.testing.assert_allclose(result["hbar"], oracle["hbar"], atol=MC_TOL)
    np.testing.assert_allclose(result["rhos"][0], oracle["rho"], atol=MC_TOL)
    np.testing.assert_allclose(
        result["expected_phi"][0], oracle["expected_phi"], atol=MC_TOL
    )
    np.testing.assert_allclose(result["log_z"], oracle["log_z"], atol=MC_TOL)


def test_mu3_replays_from_explicit_key_and_combines_slices():
    alpha = jnp.asarray([[1.2, 0.8], [0.7, 1.6]])
    key = jax.random.PRNGKey(71)
    next_a, samples_a = _draw_base(key, alpha, 2048)
    next_b, samples_b = _draw_base(key, alpha, 2048)
    np.testing.assert_array_equal(next_a, next_b)
    np.testing.assert_array_equal(samples_a, samples_b)

    abar = alpha / alpha.sum(axis=0, keepdims=True)
    zs = (jnp.asarray([0.8, 0.2]), jnp.asarray([0.25, 0.75]))
    cs = (jnp.asarray([0.9, 0.1]), jnp.asarray([0.3, 0.7]))
    result_a = _importance_step(samples_a, abar, zs, cs)
    result_b = _importance_step(samples_b, abar, zs, cs)
    for name in (
        "combined", "weights", "abar", "hbar", "rhos", "expected_phi", "log_z"
    ):
        np.testing.assert_array_equal(result_a[name], result_b[name])
    np.testing.assert_allclose(
        result_a["combined"], result_a["per_slice"].sum(axis=0), atol=0, rtol=0
    )


def test_mu3_hostile_support_weight_collapse_float32_and_scale():
    boundary = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    # A zero multiplier makes the zero/zero goal-predictive coordinate inert.
    assert float(
        _mu3_log(
            boundary,
            boundary,
            jnp.asarray([1.0, 0.0]),
            jnp.asarray([1.0, 0.0]),
        )
    ) == 0.0
    # Positive predictive mass outside goal support has zero functional weight.
    assert np.isneginf(
        float(
            _mu3_log(
                boundary,
                boundary,
                jnp.asarray([0.5, 0.5]),
                jnp.asarray([1.0, 0.0]),
            )
        )
    )

    with pytest.raises(ValueError, match="finite and > 0"):
        _draw_base(jax.random.PRNGKey(0), jnp.asarray([[0.0], [1.0]]), 8)
    with pytest.raises(RuntimeError, match="all mu3"):
        _normalize_log_weights(jnp.full((8,), -jnp.inf))
    _, _, ess, status = _normalize_log_weights(
        jnp.asarray([0.0, *([-100.0] * 15)]), min_ess_fraction=0.25
    )
    assert float(ess) == pytest.approx(1.0)
    assert status == "budget_exhausted"

    # 20 observations × 40 states is 10–20x the binary oracle dimensions.
    # Sparse positive f32 proposal columns, scores, weights and required
    # moments remain finite with no absolute probability floor.
    with jax.enable_x64(False):
        m, k, s = 20, 40, 256
        alpha = jnp.full((m, k), 0.05, dtype=jnp.float32)
        _, samples = _draw_base(jax.random.PRNGKey(5), alpha, s)
        z = jnp.full((k,), 1 / k, dtype=jnp.float32)
        c = jnp.full((m,), 1 / m, dtype=jnp.float32)
        abar = alpha / alpha.sum(axis=0, keepdims=True)
        result = _importance_step(samples, abar, (z,), (c,))
        assert result["abar"].dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(result["combined"])))
        assert bool(jnp.all(jnp.isfinite(result["weights"])))
        assert bool(jnp.all(jnp.isfinite(result["abar"])))
        assert bool(jnp.all(jnp.isfinite(result["hbar"])))
        assert np.isfinite(float(result["log_z"]))
