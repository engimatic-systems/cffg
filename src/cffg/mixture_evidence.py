"""Gaussian-mixture joint construction and component evidence."""

import jax.numpy as jnp

from cffg import gauss

_LOG2PI = 1.8378770664093453


def log_z(eta, precision):
    """Absolute Gaussian log partition; callers own propriety admission."""
    solution = gauss._pd_solve(precision, eta[..., None])[..., 0]
    return (.5 * (eta * solution).sum(-1) + .5 * precision.shape[-1] * _LOG2PI
            - .5 * gauss._pd_logdet(precision))


def joint_blocks(eta_o, precision_o, eta_i, precision_i, matrix, offset, noise_precision):
    """Per-component joint information over (out, in)."""
    weighted = jnp.einsum("kij,kjl->kil", noise_precision, matrix)
    top = jnp.concatenate([noise_precision + precision_o[None], -weighted], axis=-1)
    bottom = jnp.concatenate([
        -weighted.swapaxes(-1, -2),
        jnp.einsum("kji,kjl->kil", matrix, weighted) + precision_i[None],
    ], axis=-1)
    precision = jnp.concatenate([top, bottom], axis=-2)
    weighted_offset = jnp.einsum("kij,kj->ki", noise_precision, offset)
    eta = jnp.concatenate([
        weighted_offset + eta_o[None],
        -jnp.einsum("kji,kj->ki", matrix, weighted_offset) + eta_i[None],
    ], axis=-1)
    return eta, precision


def node_constant(offset, noise_precision):
    """Component-dependent constant in a normalized Gaussian transition."""
    return (-.5 * jnp.einsum("ki,kij,kj->k", offset, noise_precision, offset)
            + .5 * gauss._pd_logdet(noise_precision) - .5 * offset.shape[-1] * _LOG2PI)
