"""Gaussian likelihood parameterization of the released AXIOM mixture model."""

import jax
import jax.numpy as jnp

from cffg.mixed_state import MixedBank


def release_view(bank: MixedBank) -> MixedBank:
    """Compute the released model's scoring scale from the unmodified posterior.

    In normalized natural-parameter coordinates the model sets diagonal
    entries below 0.001 to 0.001 and removes smaller off-diagonals. This
    threshold belongs to the model, including on float64. The original
    posterior carries the sufficient statistics for the next observation.
    """
    psi = bank.psi
    mean = bank.mu + bank.carry.mu
    kappa = bank.kappa + bank.carry.kappa
    diag = jnp.diagonal(psi + bank.carry.psi, axis1=-2, axis2=-1)
    magnitude = jnp.maximum(1, 0.5 * (diag + kappa[:, None] * mean**2))
    scale = jnp.sqrt(magnitude)
    cutoff = (scale[:, :, None] * scale[:, None, :]) * jnp.asarray(1e-3, psi.dtype)
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
    return bank._replace(psi=high, carry=bank.carry._replace(psi=low))


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
