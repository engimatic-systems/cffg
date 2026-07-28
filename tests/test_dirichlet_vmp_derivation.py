"""Executable anchors for DERIVATIONS D6a's Dirichlet VMP contract.

These tests deliberately build the sufficient statistics and VFE from direct
Dirichlet algebra.  They exercise the existing out-of-graph family; the
in-graph arena path belongs to cff-fr5c.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import digamma, gammaln

from cffg.learn import Dirichlet


FAMILY = Dirichlet()


def _log_beta(alpha):
    alpha = jnp.asarray(alpha)
    return jnp.sum(gammaln(alpha), axis=0) - gammaln(jnp.sum(alpha, axis=0))


def _expected_log(alpha):
    alpha = jnp.asarray(alpha)
    return digamma(alpha) - digamma(jnp.sum(alpha, axis=0, keepdims=True))


def _negative_entropy(alpha):
    alpha = jnp.asarray(alpha)
    elog = _expected_log(alpha)
    return -_log_beta(alpha) + jnp.sum((alpha - 1) * elog, axis=0)


def _parameter_block_vfe(alpha0, counts):
    """Eq. D6a.3 without the ordinary categorical-edge entropies."""
    alpha0 = jnp.asarray(alpha0)
    counts = jnp.asarray(counts)
    alpha = alpha0 + counts
    elog = _expected_log(alpha)
    q_term = _negative_entropy(alpha)
    prior_energy = _log_beta(alpha0) - jnp.sum((alpha0 - 1) * elog, axis=0)
    likelihood_energy = -jnp.sum(counts * elog, axis=0)
    return jnp.sum(q_term + prior_energy + likelihood_energy)


def _state_negative_entropy(q):
    q = jnp.asarray(q)
    return jnp.sum(jnp.where(q > 0, q * jnp.log(q), 0))


def _assert_fold_and_vfe(ftype, alpha0, readout, expected_counts, state_qs):
    stats = FAMILY.stats(ftype, readout)
    np.testing.assert_allclose(stats["n"], expected_counts, rtol=1e-15, atol=0)

    posterior = FAMILY.fold({"alpha": np.asarray(alpha0)}, stats)["alpha"]
    np.testing.assert_allclose(
        posterior, alpha0 + expected_counts, rtol=1e-15, atol=0
    )

    direct = _parameter_block_vfe(alpha0, expected_counts)
    collapsed = jnp.sum(_log_beta(alpha0) - _log_beta(posterior))
    np.testing.assert_allclose(direct, collapsed, rtol=1e-12, atol=1e-12)

    # Complete local mean-field block: every ordinary edge contributes q log q
    # once, while the parameter part reduces to the beta-function ratio.
    complete = direct + sum(_state_negative_entropy(q) for q in state_qs)
    independently_collapsed = collapsed + sum(
        jnp.sum(jnp.where(jnp.asarray(q) > 0, q * jnp.log(q), 0)) for q in state_qs
    )
    np.testing.assert_allclose(complete, independently_collapsed, rtol=1e-12, atol=1e-12)


def test_categorical_fractional_counts_and_total_vfe():
    q_out = np.array([0.0, 0.25, 0.75])
    alpha0 = np.array([0.4, 1.2, 2.5])
    _assert_fold_and_vfe("categorical", alpha0, q_out, q_out, (q_out,))


def test_matrix_cat_fractional_counts_and_total_vfe():
    q_out = np.array([0.1, 0.6, 0.3])
    q_in = np.array([0.8, 0.2])
    counts = q_out[:, None] * q_in[None, :]
    alpha0 = np.array([[0.2, 1.1], [0.7, 0.4], [1.5, 0.3]])
    _assert_fold_and_vfe(
        "matrix_cat", alpha0, counts, counts, (q_out, q_in)
    )


def test_transition_mixture_fractional_counts_and_total_vfe():
    q_out = np.array([0.3, 0.7])
    q_in = np.array([0.2, 0.5, 0.3])
    q_sel = np.array([0.6, 0.0, 0.4])
    counts = np.einsum("u,o,i->uoi", q_sel, q_out, q_in)
    node_belief = np.einsum("o,i,u->oiu", q_out, q_in, q_sel)
    alpha0 = np.linspace(0.2, 2.0, counts.size).reshape(counts.shape)
    _assert_fold_and_vfe(
        "transition_mixture",
        alpha0,
        node_belief,
        counts,
        (q_out, q_in, q_sel),
    )


def test_exact_bp_parameter_message_is_not_dirichlet_closed():
    """A weighted sum of two table cells cannot be one Dirichlet monomial."""
    theta = np.array(
        [
            [0.10, 0.20, 0.70],
            [0.20, 0.30, 0.50],
            [0.40, 0.10, 0.50],
            [0.25, 0.60, 0.15],
            [0.55, 0.35, 0.10],
            [0.70, 0.20, 0.10],
            [0.15, 0.75, 0.10],
            [0.45, 0.45, 0.10],
        ]
    )
    design = np.column_stack([np.ones(len(theta)), np.log(theta)])

    latent_bp = theta @ np.array([0.35, 0.65, 0.0])
    _, residuals, _, _ = np.linalg.lstsq(design, np.log(latent_bp), rcond=None)
    assert residuals.item() > 1e-4

    # The hard-assignment exception is exactly the monomial theta[1].
    hard_bp = theta[:, 1]
    coefficients, residuals, _, _ = np.linalg.lstsq(
        design, np.log(hard_bp), rcond=None
    )
    np.testing.assert_allclose(coefficients, [0.0, 0.0, 1.0, 0.0], atol=1e-12)
    assert residuals.item() < 1e-25


def test_dirichlet_vmp_hostile_inputs_and_scale():
    # A missing or improper base is an error, never a vacuous message.
    with pytest.raises(ValueError, match="posterior is"):
        FAMILY.validate_prior("A", {}, {"A": (2, 2)})
    for bad in (
        np.array([[0.0, 1.0], [1.0, 1.0]]),
        np.array([[-1.0, 1.0], [1.0, 1.0]]),
        np.array([[np.nan, 1.0], [1.0, 1.0]]),
    ):
        with pytest.raises(ValueError, match="must be > 0"):
            FAMILY.validate_prior("A", {"alpha": bad}, {"A": (2, 2)})

    # Exact-zero mass survives the outer product as an exact-zero count.
    q_out = np.array([1.0, 0.0, 0.0])
    q_in = np.array([0.25, 0.0, 0.75])
    counts = FAMILY.stats("matrix_cat", q_out[:, None] * q_in[None, :])["n"]
    np.testing.assert_array_equal(counts[1:], 0.0)
    np.testing.assert_array_equal(counts[:, 1], 0.0)

    # 20 actions × 64 outputs × 48 inputs is 20–32x the tiny anchors on every
    # categorical axis. Sparse positive f32 concentrations and their expected
    # logs remain finite without a literal floor.
    with jax.enable_x64(False):
        alpha = np.full((20, 64, 48), 1e-3, dtype=np.float32)
        alpha[:, 0, :] = 2.0
        elog = FAMILY.rollover({"alpha": alpha}, "transition_mixture")["matrices"]
        assert elog.dtype == np.float32
        assert elog.shape == alpha.shape
        assert np.all(np.isfinite(elog))

        q_out = np.full(64, 1 / 64, dtype=np.float32)
        q_in = np.full(48, 1 / 48, dtype=np.float32)
        q_sel = np.zeros(20, dtype=np.float32)
        q_sel[::3] = 1
        q_sel /= q_sel.sum()
        belief = np.einsum("o,i,u->oiu", q_out, q_in, q_sel)
        large_counts = FAMILY.stats("transition_mixture", belief)["n"]
        assert large_counts.shape == alpha.shape
        assert np.all(np.isfinite(large_counts))
        np.testing.assert_array_equal(large_counts[q_sel == 0], 0.0)
