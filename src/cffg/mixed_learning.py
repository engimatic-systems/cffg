"""Online conjugate learning for a bank of Gaussian and categorical components."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg import mixed_scoring


from cffg.mixed_state import MixedBank, MixedCarry


class MixedBatch(NamedTuple):
    """Conditional feature moments and known category indices for N records.

    mean [N,d] and covariance [N,d,d] describe the same continuous field.
    gaussian_observed [N] selects observed rows. A point has zero covariance.
    Each integer category array has shape [N]; -1 denotes a missing field.
    """

    mean: jax.Array
    covariance: jax.Array
    gaussian_observed: jax.Array
    categories: tuple[jax.Array, ...]


def score(bank: MixedBank, batch: MixedBatch, *, include_mixing: bool = False):
    """Return absolute observed log-likelihood scores, responsibilities and validity.

    Only the normalized responsibilities incorporate optional expected log
    mixing weights. Unsupported components receive score -inf and zero mass.
    """
    return mixed_scoring.score(bank, batch, include_mixing=include_mixing)


def update(bank: MixedBank, batch: MixedBatch, weights: jax.Array) -> tuple[MixedBank, jax.Array]:
    """Add a batch with fixed responsibilities; return (new bank, valid).

    Every observed field uses the same [N,K] probability matrix. A partially
    observed record counts once toward mixture frequencies, and a record with
    no observed field contributes nothing. Invalid numerical inputs or an
    unrepresentable posterior return the input bank with valid=False.
    """
    valid = mixed_scoring.validate_inputs(bank, batch)
    n = batch.mean.shape[0]
    k = len(bank.mu)
    if np.shape(weights) != (n, k) or getattr(weights, "dtype", None) != bank.mu.dtype:
        raise ValueError("weights must have shape [N,K] and the bank's floating dtype")
    bank, batch, weights = jax.tree.map(jnp.asarray, (bank, batch, weights))
    # A tree sum of K normalized probabilities can accumulate rounding error.
    tolerance = (8 + k.bit_length()) * jnp.finfo(weights.dtype).eps
    valid &= (jnp.all(jnp.isfinite(weights)) & jnp.all(weights >= 0)
              & jnp.all(jnp.where(bank.active[None, :], True, weights == 0))
              & jnp.all(jnp.abs(weights.sum(axis=1) - 1) <= tolerance))
    if n == 0:
        return bank, valid

    def accumulate():
        observed = batch.gaussian_observed
        gaussian_weights = jnp.where(observed[:, None], weights, 0)
        mass = gaussian_weights.sum(axis=0)
        kn = bank.kappa + mass
        mean = jnp.where(observed[:, None], batch.mean, 0)
        cov = jnp.where(observed[:, None, None], batch.covariance, 0)
        cov = 0.5 * cov + 0.5 * jnp.swapaxes(cov, -1, -2)
        # Center where the evidence is largest, before forming any square.
        # Keeping the offset in local coordinates also preserves scatter when
        # the final absolute posterior mean rounds to the prior.
        anchor = jnp.where((mass >= bank.kappa)[:, None],
                           mean[jnp.argmax(gaussian_weights, axis=0)], bank.mu)
        delta = jnp.where(gaussian_weights[:, :, None] > 0, mean[:, None, :] - anchor, 0)
        prior_delta = bank.mu - anchor
        precision = jax.lax.Precision.HIGHEST
        center = ((bank.kappa / kn)[:, None] * prior_delta
                  + jnp.einsum("nk,nkd->kd", gaussian_weights / kn, delta, precision=precision))
        residual = jnp.where(gaussian_weights[:, :, None] > 0, delta - center, 0)
        scaled = jnp.sqrt(gaussian_weights)[:, :, None] * residual
        prior_residual = jnp.sqrt(bank.kappa)[:, None] * (prior_delta - center)
        within = jnp.einsum("nk,nij->kij", gaussian_weights, cov, precision=precision)
        psi = (bank.psi + within + jnp.einsum("nki,nkj->kij", scaled, scaled, precision=precision)
               + prior_residual[:, :, None] * prior_residual[:, None, :])
        present = observed
        heads = []
        for index, alpha in zip(batch.categories, bank.categories, strict=True):
            index = mixed_scoring.category_indices(index)
            observed_category = index >= 0
            present = present | observed_category
            hot = jax.nn.one_hot(index, alpha.shape[1], dtype=weights.dtype)
            hot = jnp.where(observed_category[:, None], hot, 0)
            counts = jnp.matmul(weights.T, hot, precision=precision)
            heads.append(jnp.where(counts > 0, alpha + counts, alpha))
        seen_mass = jnp.where(present[:, None], weights, 0).sum(axis=0)
        changed = mass > 0
        candidate = MixedBank(
            jnp.where(changed[:, None], anchor + center, bank.mu),
            jnp.where(changed, kn, bank.kappa),
            jnp.where(changed[:, None, None], psi, bank.psi),
            jnp.where(changed, bank.nu + mass, bank.nu),
            jnp.where(seen_mass > 0, bank.alpha + seen_mass, bank.alpha),
            tuple(heads), bank.active,
        )
        proper = mixed_scoring.validate_bank(candidate)
        return jax.tree.map(lambda new, old: jnp.where(proper, new, old), candidate, bank), proper

    return jax.lax.cond(valid, accumulate, lambda: (bank, jnp.asarray(False)))
