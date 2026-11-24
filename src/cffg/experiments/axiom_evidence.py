"""Gaussian likelihood parameterization of the released AXIOM mixture model."""

import jax
import jax.numpy as jnp

from cffg import _compensated
from cffg.mixed_state import MixedBank


def _project_scale(bank):
    psi = bank.psi
    mean = bank.mu + bank.carry.mu
    kappa = bank.kappa + bank.carry.kappa
    diag = jnp.diagonal(psi + bank.carry.psi, axis1=-2, axis2=-1)
    magnitude = jnp.maximum(1, 0.5 * (diag + kappa[:, None] * mean**2))
    scale = jnp.sqrt(magnitude)
    norm = scale[:, :, None] * scale[:, None, :]
    cutoff = norm * jnp.asarray(1e-3, psi.dtype)
    eye = jnp.eye(psi.shape[-1], dtype=bool)[None]
    # Compare while keeping the low word at the boundary. If a field survives
    # the projection its exact outgoing high/low pair remains available.
    low = bank.carry.psi
    diff = (jnp.abs(psi) - cutoff) + jnp.sign(psi) * low
    replace_diag = eye & ((psi - cutoff) + low < 0)
    drop_cross = ~eye & (diff < 0)
    high = jnp.where(replace_diag, cutoff, jnp.where(drop_cross, 0, psi))
    low = jnp.where(replace_diag | drop_cross, 0, low)
    # Propagate range failure from the normalization to the public score check.
    high = jnp.where(jnp.isfinite(cutoff), high, jnp.nan)
    return bank._replace(psi=high, carry=bank.carry._replace(psi=low)), norm


def sampling_view(bank: MixedBank) -> MixedBank:
    """Return the projected scale used in the release's expected covariance.

    The released sampler divides this scale by nu-d-1 before applying
    a separate definiteness repair to each component's covariance matrix.
    """
    projected, _ = _project_scale(bank)
    return projected


def _minima(projected, norm):
    normalized = (projected.psi + projected.carry.psi) / norm
    d = normalized.shape[-1]
    diag = jnp.diagonal(normalized, axis1=-2, axis2=-1)
    off_sum = jnp.sum(jnp.where(jnp.eye(d, dtype=bool), 0, jnp.abs(normalized)), axis=-1)
    eps = jnp.finfo(normalized.dtype).eps
    # Strict diagonal dominance certifies positivity cheaply for initial
    # banks. Allow for roundoff in the row sums and in the eigensolver.
    margin = 8 * d * eps * jnp.max(jnp.abs(diag) + off_sum, axis=-1, keepdims=True)
    positive = jnp.all(diag - off_sum > margin)

    def eigenvalues():
        return jnp.minimum(jnp.linalg.eigvalsh(normalized)[:, 0], 0)

    return jax.lax.cond(positive, lambda: jnp.zeros_like(diag[:, 0]), eigenvalues)


def release_minima(bank: MixedBank) -> jax.Array:
    """Return each row's smallest normalized projected eigenvalue, capped at zero."""
    return _minima(*_project_scale(bank))


def release_view(bank: MixedBank, *, minimum=None) -> MixedBank:
    """Return the likelihood scale using the full bank's normalized eigenvalue repair.

    AXIOM thresholds the normalized scale at 0.001 and repairs the precision
    and log determinant using one minimum across all rows, including dormant
    ones. When scoring selected rows, supply min(release_minima(full_bank))
    from the same model revision. Keep the raw posterior for learning.
    """
    projected, norm = _project_scale(bank)
    d = bank.mu.shape[-1]
    if minimum is None:
        minimum = jnp.min(_minima(projected, norm))
    else:
        minimum = jnp.asarray(minimum)
        if minimum.shape != () or minimum.dtype != bank.mu.dtype:
            raise ValueError("minimum must be a scalar of the bank's floating dtype")

    def repair():
        # The shared shift is in normalized coordinates. Convert it back to
        # each component's physical units, preserving stored off-diagonal lows.
        shift = 2 * (jnp.finfo(bank.psi.dtype).eps - minimum)
        delta = shift * jnp.diagonal(norm, axis1=-2, axis2=-1)
        idx = jnp.arange(d)
        high, low, _ = _compensated.add(
            projected.psi[:, idx, idx], projected.carry.psi[:, idx, idx], delta
        )
        return projected._replace(
            psi=projected.psi.at[:, idx, idx].set(high),
            carry=projected.carry._replace(psi=projected.carry.psi.at[:, idx, idx].set(low)),
        )

    return jax.lax.cond(jnp.isfinite(minimum) & (minimum >= 0), lambda: projected, repair)


def initialized_view(posterior, generations, dormant_initial):
    """Expose seeded unused rows to unmasked AXIOM queries until their first birth."""
    if dormant_initial is None:
        return posterior
    if isinstance(generations, (tuple, list)):
        # Compare host IDs before transfer when JAX 64-bit integers are disabled.
        generations = tuple(int(n != 0) for n in generations)
    generations = jnp.asarray(generations)
    k = len(posterior.alpha)
    if generations.shape == (k, 2) and generations.dtype == jnp.uint32:
        unused = jnp.all(generations == 0, axis=1)
    elif generations.shape == (k,) and jnp.issubdtype(generations.dtype, jnp.integer):
        unused = generations == 0
    else:
        raise ValueError("expected one generation per component")
    initial = unused & ~posterior.active

    def choose(current, original):
        if current.shape != original.shape or current.dtype != original.dtype:
            raise ValueError("seeded components must match the learned bank layout")
        return jnp.where(initial.reshape((k,) + (1,) * (current.ndim - 1)), original, current)

    return jax.tree.map(choose, posterior, dormant_initial)._replace(active=posterior.active)
