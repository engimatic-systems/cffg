"""The released AXIOM readout and its effects on match versus birth."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma

from cffg.experiments.axiom_evidence import initialized_view, release_view
from cffg.mixed_learning import MixedBank, MixedBatch, score


def posterior(psi, mu, dtype):
    k, d = np.shape(mu)
    return MixedBank(
        jnp.asarray(mu, dtype),
        jnp.full(k, 1.0001, dtype),
        jnp.asarray(psi, dtype),
        jnp.full(k, 2 + 2 * d, dtype),
        jnp.ones(k, dtype),
        (),
        jnp.ones(k, bool),
    )


def observation(x):
    d = x.shape[-1]
    return MixedBatch(x[None], jnp.zeros((1, d, d), x.dtype), jnp.ones(1, bool), ())


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_default_interaction_scale_changes_novelty_decision(dtype):
    d = 7
    bank = posterior(np.eye(d)[None] / 75**2, np.zeros((1, d)), dtype)
    point = observation(jnp.zeros(d, dtype).at[0].set(0.03))
    effective = jax.jit(release_view)(bank)
    for model, scale in ((bank, 75**-2), (effective, 0.001)):
        ell, _, ok = score(model, point)
        assert ok
        expected = 0.5 * (
            np.sum(digamma((16 - np.arange(d)) / 2))
            + d * np.log(2)
            - d * np.log(2 * np.pi * scale)
            - d / 1.0001
            - 16 * 0.03**2 / scale
        )
        np.testing.assert_allclose(ell[0, 0], expected, atol=1e-5)
        assert bool(ell[0, 0] > -10) == (scale == 0.001)
    np.testing.assert_allclose(bank.psi[0], np.eye(d) / 75**2, atol=1e-10)


def test_scale_projection_from_raw_natural_parameters_and_zero_boundary():
    mean = [[99.0, 1.0, 10.0], [0.0, 0.0, 0.0]]
    matrices = np.array(
        [
            [[1.0, 0.0002, -0.007], [0.0002, 0.1, 0.1], [-0.007, 0.1, 0.5]],
            [[1.0, 0.001, -0.0009], [0.001, 1.0, -0.0015], [-0.0009, -0.0015, 0.2]],
        ]
    )
    b = posterior(matrices, mean, jnp.float64)
    ref = []
    for m, p, k in zip(b.mu, b.psi, b.kappa, strict=True):
        m, p, k = np.asarray(m), np.asarray(p), float(k)
        eta = -0.5 * (p + k * np.outer(m, m))
        scale = np.sqrt(np.maximum(abs(np.diag(eta)), 1))
        norm = np.outer(scale, scale)
        normalized = (-2 * eta - k * np.outer(m, m)) / norm
        np.fill_diagonal(normalized, np.maximum(normalized.diagonal(), 1e-3))
        normalized[abs(normalized) < 1e-3] = 0
        ref.append(normalized * norm)
    out = release_view(b)
    np.testing.assert_allclose(out.psi, ref, rtol=5e-13, atol=1e-13)
    assert out.psi[1, 0, 1] == 0.001


def test_sparsifying_an_otherwise_proper_scale_can_invalidate_the_released_score():
    b = posterior(
        [[[0.01, 0.0009, 0.0072], [0.0009, 0.01, 0.0072], [0.0072, 0.0072, 0.01]]],
        [[0, 0, 0]],
        jnp.float64,
    )
    point = observation(b.mu[0])
    assert score(b, point)[2]
    assert not score(release_view(b), point)[2]


def test_virgin_overlay_is_separate_from_retired_rows_and_raw_reset_priors():
    bank = posterior(np.tile(np.eye(2), (3, 1, 1)), np.zeros((3, 2)), jnp.float32)
    bank = bank._replace(active=jnp.array([False, False, True]))
    dormant = bank._replace(mu=jnp.ones_like(bank.mu))
    for generations in ((0, 2**35, 1), jnp.array([[0, 0], [8, 0], [0, 1]], jnp.uint32)):
        effective = initialized_view(bank, generations, dormant)
        np.testing.assert_array_equal(effective.mu, [[1, 1], [0, 0], [0, 0]])
        np.testing.assert_array_equal(effective.active, bank.active)
    np.testing.assert_array_equal(bank.mu, 0)
