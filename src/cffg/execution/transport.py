"""Shape-local gathers and scatters shared by runtime consumers."""

import jax
import jax.numpy as jnp
import numpy as np

from cffg.compiled import NEG, Arena


def _disc_rows(
    arena: Arena,
    messages: jax.Array,
    slots: np.ndarray,
    cards: tuple[int, ...],
) -> jax.Array:
    """Gather logical discrete slots into one shape-local padded block.

    Padding exists only in this transient rule input and only to the maximum
    role cardinality of one homogeneous factor group. Persistent storage is
    exactly segmented, so an unrelated model-wide outlier cannot widen it.
    """
    width = max(cards)
    blocks = []
    for role, card in enumerate(cards):
        block = messages[arena.message_indices(slots[..., role], card)]
        blocks.append(
            jnp.pad(
                block,
                [(0, 0)] * (block.ndim - 1) + [(0, width - card)],
                constant_values=NEG,
            )
        )
    return jnp.stack(blocks, axis=-2)


def _disc_both(
    arena: Arena,
    messages: jax.Array,
    slots: np.ndarray,
    cards: tuple[int, ...],
) -> jax.Array:
    """Gather both logical slots per VMP edge as ``[M, R, 2, local_k]``."""
    width = max(cards)
    blocks = []
    for role, card in enumerate(cards):
        block = messages[arena.message_indices(slots[:, role, :], card)]
        blocks.append(
            jnp.pad(block, ((0, 0), (0, 0), (0, width - card)), constant_values=NEG)
        )
    return jnp.stack(blocks, axis=1)


def _set_disc_rows(
    arena: Arena,
    messages: jax.Array,
    slots: np.ndarray,
    cards: tuple[int, ...],
    rows: jax.Array,
) -> jax.Array:
    """Scatter a shape-local ``[..., R, local_k]`` rule output exactly."""
    out = messages
    for role, card in enumerate(cards):
        idx = arena.message_indices(slots[..., role], card)
        out = out.at[idx].set(rows[..., role, :card])
    return out


def _set_disc_block(
    arena: Arena,
    messages: jax.Array,
    slots: np.ndarray,
    card: int,
    rows: jax.Array,
) -> jax.Array:
    return messages.at[arena.message_indices(slots, card)].set(rows[..., :card])


def _gauss_rows(
    arena: Arena,
    eta: jax.Array,
    lam: jax.Array,
    slots: np.ndarray,
    dims: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Gather Gaussian segments into one shape-local padded rule block."""
    width = max(dims)
    eta_blocks, lam_blocks = [], []
    for role, dim in enumerate(dims):
        e = eta[arena.g_eta_indices(slots[..., role], dim)]
        l = lam[arena.g_lam_indices(slots[..., role], dim)]
        eta_blocks.append(jnp.pad(e, [(0, 0)] * (e.ndim - 1) + [(0, width - dim)]))
        lam_blocks.append(
            jnp.pad(l, [(0, 0)] * (l.ndim - 2) + [(0, width - dim), (0, width - dim)])
        )
    return jnp.stack(eta_blocks, axis=-2), jnp.stack(lam_blocks, axis=-3)


def _set_gauss_block(
    arena: Arena,
    eta: jax.Array,
    lam: jax.Array,
    slots: np.ndarray,
    dim: int,
    eta_rows: jax.Array,
    lam_rows: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return (
        eta.at[arena.g_eta_indices(slots, dim)].set(eta_rows[..., :dim]),
        lam.at[arena.g_lam_indices(slots, dim)].set(lam_rows[..., :dim, :dim]),
    )


def _obs_rows(
    arena: Arena, observations: jax.Array, rows: np.ndarray, dim: int
) -> jax.Array:
    return observations[arena.g_obs_indices(rows, dim)]
