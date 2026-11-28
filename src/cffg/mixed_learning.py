"""Online conjugate learning for a bank of Gaussian and categorical components."""

from __future__ import annotations

from typing import NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from cffg import _compensated, mixed_scoring


from cffg.mixed_state import MixedBank, MixedCarry, RoundedBank


_Bank = TypeVar("_Bank", MixedBank, RoundedBank)


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


def score(bank: MixedBank | RoundedBank, batch: MixedBatch, *, include_mixing: bool = False):
    """Return absolute observed log-likelihood scores, responsibilities and validity.

    Only the normalized responsibilities incorporate optional expected log
    mixing weights. Unsupported components receive score -inf and zero mass.
    A RoundedBank scores only the primary fields.
    """
    return mixed_scoring.score(bank, batch, include_mixing=include_mixing)


def _add_count(value, low, count):
    if low is None:
        return jnp.where(count > 0, value + count, value), None, jnp.asarray(True)
    return _compensated.add(value, low, count, count=True)


def update(bank: _Bank, batch: MixedBatch, weights: jax.Array) -> tuple[_Bank, jax.Array]:
    """Add a batch with fixed responsibilities; return (new bank, valid).

    Every observed field uses the same [N,K] probability matrix. A partially
    observed record counts once toward mixture frequencies, and a record with
    no observed field contributes nothing. Invalid numerical inputs or an
    unrepresentable posterior return the input bank with valid=False.

    Batch products and reductions use the input dtype. High and low words
    retain their rounded increments across calls; neither word extends the
    dtype's exponent range. Counts report exhaustion if an increment loses
    mass and the low word can no longer resolve even a whole observation.
    A RoundedBank instead rounds each update to a single word. A fractional
    change can disappear at a large mean or count while validity remains true.
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
        carry = None if isinstance(bank, RoundedBank) else bank.carry
        kn, kappa_low, ok = _add_count(bank.kappa, None if carry is None else carry.kappa, mass)
        mean = jnp.where(observed[:, None], batch.mean, 0)
        cov = jnp.where(observed[:, None, None], batch.covariance, 0)
        cov = 0.5 * cov + 0.5 * jnp.swapaxes(cov, -1, -2)
        # Center where the evidence is largest, before forming any square.
        # Keeping the offset in local coordinates also preserves scatter when
        # the final absolute posterior mean rounds to the prior.
        use_data = (mass >= bank.kappa)[:, None]
        anchor = jnp.where(use_data,
                           mean[jnp.argmax(gaussian_weights, axis=0)], bank.mu)
        diff = mean[:, None, :] - anchor
        prior_delta = bank.mu - anchor
        precision = jax.lax.Precision.HIGHEST
        if carry is None:
            delta = jnp.where(gaussian_weights[:, :, None] > 0, diff, 0)
            center = ((bank.kappa / kn)[:, None] * prior_delta
                      + jnp.einsum("nk,nkd->kd", gaussian_weights / kn, delta, precision=precision))
            mu = jnp.where(mass[:, None] > 0, anchor + center, bank.mu)
        else:
            anchor_low = jnp.where(use_data, 0, carry.mu)
            delta = jnp.where(gaussian_weights[:, :, None] > 0, diff - anchor_low, 0)
            prior_delta += carry.mu - anchor_low
            data_sum = jnp.einsum("nk,nkd->kd", gaussian_weights, delta, precision=precision)
            center = (bank.kappa / kn)[:, None] * prior_delta + data_sum / kn[:, None]
            mu, mu_low, representable = _compensated.add(anchor, anchor_low, center)
            ok &= representable
        residual = jnp.where(gaussian_weights[:, :, None] > 0, delta - center, 0)
        scaled = jnp.sqrt(gaussian_weights)[:, :, None] * residual
        prior_residual = jnp.sqrt(bank.kappa)[:, None] * (prior_delta - center)
        within = jnp.einsum("nk,nij->kij", gaussian_weights, cov, precision=precision)
        scatter = jnp.einsum("nki,nkj->kij", scaled, scaled, precision=precision)
        prior_scatter = prior_residual[:, :, None] * prior_residual[:, None, :]
        if carry is None:
            psi = jnp.where(mass[:, None, None] > 0,
                            bank.psi + within + scatter + prior_scatter, bank.psi)
        else:
            increment = within + scatter + prior_scatter
            psi, psi_low, representable = _compensated.add(bank.psi, carry.psi, increment)
            ok &= representable
        present = observed
        heads, head_lows = [], []
        for i, (index, alpha) in enumerate(zip(batch.categories, bank.categories, strict=True)):
            index = mixed_scoring.category_indices(index)
            observed_category = index >= 0
            present = present | observed_category
            hot = jax.nn.one_hot(index, alpha.shape[1], dtype=weights.dtype)
            hot = jnp.where(observed_category[:, None], hot, 0)
            counts = jnp.matmul(weights.T, hot, precision=precision)
            head, head_low, representable = _add_count(alpha, None if carry is None else carry.categories[i], counts)
            ok &= representable
            heads.append(head)
            head_lows.append(head_low)
        seen_mass = jnp.where(present[:, None], weights, 0).sum(axis=0)
        alpha, alpha_low, representable = _add_count(bank.alpha, None if carry is None else carry.alpha, seen_mass)
        ok &= representable
        nu, nu_low, representable = _add_count(bank.nu, None if carry is None else carry.nu, mass)
        ok &= representable
        fields = mu, kn, psi, nu, alpha, tuple(heads), bank.active
        if carry is None:
            candidate = RoundedBank(*fields)
        else:
            candidate = MixedBank(*fields, MixedCarry(mu_low, kappa_low, psi_low, nu_low,
                                                      alpha_low, tuple(head_lows)))
        proper = ok & mixed_scoring.validate_bank(candidate)
        return jax.tree.map(lambda new, old: jnp.where(proper, new, old), candidate, bank), proper

    return jax.lax.cond(valid, accumulate, lambda: (bank, jnp.asarray(False)))
