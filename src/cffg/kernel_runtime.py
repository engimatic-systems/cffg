"""Local kernel calls shared by flooding, schedules, and public readouts.

These adapters never gather from a global message arena. In particular a
registerized scan can call them with its existing small local carry blocks.
"""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cffg import efrules
from cffg.rules import bp_equality, energy_equality


def parameters(group, values, rows=None):
    return {
        key: values[path] if rows is None else values[path][rows]
        for key, path in group.parameter_paths
    }


def discrete_input(group, incoming):
    if group.mode == "bp":
        return incoming
    return tuple(
        efrules.CATEGORICAL.moments(
            efrules.CATEGORICAL.product(
                incoming[:, role, 0, :card], incoming[:, role, 1, :card]
            )
        )
        for role, card in enumerate(group.cards)
    )


def discrete_message(group, incoming, params, reduction):
    width = max(group.cards)
    if group.kernel is None:
        out = bp_equality(incoming, None, group.cards, width, reduction)
    else:
        binding = group.kernel.binding
        inputs = discrete_input(group, incoming)
        if group.mode == "vmp":
            out = binding.vmp.message(inputs, params, group.cards, width)
        else:
            out = binding.bp.message(inputs, params, group.cards, width, reduction)
    expected = (incoming.shape[0], len(group.cards), width)
    if out.shape != expected:
        raise ValueError(
            f"group {group.name!r}: outgoing message shape {out.shape} != {expected}"
        )
    return out


def discrete_energy(group, incoming, params):
    if group.kernel is None:
        out = energy_equality(incoming, jnp.zeros(()), group.cards)
    else:
        mode = (
            group.kernel.binding.vmp if group.mode == "vmp" else group.kernel.binding.bp
        )
        out = mode.energy(discrete_input(group, incoming), params, group.cards)
    if out.shape != (incoming.shape[0],):
        raise ValueError(
            f"group {group.name!r}: node energy must have one value per member"
        )
    return out


def discrete_potential(group, params):
    out = group.kernel.binding.potential(params, group.cards)
    expected = (len(group.members), *group.cards)
    if out.shape != expected:
        raise ValueError(
            f"group {group.name!r}: log-potential shape {out.shape} != {expected}"
        )
    return out


def discrete_readout(group, incoming, params):
    mode = group.kernel.binding.vmp if group.mode == "vmp" else group.kernel.binding.bp
    values = mode.readout(discrete_input(group, incoming), params, group.cards)
    for key, value in values.items():
        if not value.shape or value.shape[0] != incoming.shape[0]:
            raise ValueError(
                f"group {group.name!r}: readout {key!r} needs a member axis"
            )
    return values


def validate_discrete_shapes(group, packed, semiring):
    """Abstract evaluation at compilation; no callback sees concrete data.

    Structural declarations and host packing remain JAX-independent. This
    numerical check is imported only when compiling an executable arena.
    It validates every declared mode, not only the model's current bead.
    """
    params = {
        key: jax.ShapeDtypeStruct(
            value.shape, jax.dtypes.canonicalize_dtype(value.dtype)
        )
        for key, value in packed.items()
    }
    dtype = jax.dtypes.canonicalize_dtype(np.float64)
    m, r, width = len(group.members), len(group.cards), max(group.cards)
    jax.eval_shape(lambda p: discrete_potential(group, p), params)
    modes = ("bp", "vmp") if group.kernel.binding.vmp is not None else ("bp",)
    for mode in modes:
        plan = replace(group, mode=mode)
        shape = (m, r, width) if mode == "bp" else (m, r, 2, width)
        inc = jax.ShapeDtypeStruct(shape, dtype)
        reductions = group.kernel.binding.semirings if mode == "bp" else (semiring,)
        for reduction in reductions:
            op = jnp.max if reduction == "max" else logsumexp
            jax.eval_shape(partial(discrete_message, plan, reduction=op), inc, params)
        jax.eval_shape(partial(discrete_energy, plan), inc, params)
        callbacks = plan.kernel.binding.vmp if mode == "vmp" else plan.kernel.binding.bp
        if callbacks.readout is not None:
            jax.eval_shape(partial(discrete_readout, plan), inc, params)


def gaussian_message(group, eta, precision, params, observation=None, mask=None):
    if group.kernel is None:
        from cffg import gauss

        d = group.dims[0]
        e, p = gauss.equality_messages(eta[..., :d], precision[..., :d, :d])
        outputs = tuple((e[:, r], p[:, r]) for r in range(len(group.roles)))
    else:
        outputs = group.kernel.binding.message(
            eta,
            precision,
            params,
            group.dims,
            group.observed_role,
            observation,
            mask,
        )
    if len(outputs) != len(group.roles):
        raise ValueError(
            f"group {group.name!r}: outgoing message needs one pair per role"
        )
    for role, dim, pair in zip(group.roles, group.dims, outputs):
        e, p = pair
        if e.shape != (eta.shape[0], dim) or p.shape != (eta.shape[0], dim, dim):
            raise ValueError(
                f"group {group.name!r}: outgoing message shape for {role!r} must be [M,{dim}] and [M,{dim},{dim}]"
            )
    return outputs


def gaussian_energy(group, eta, precision, params, observation=None, mask=None):
    if group.kernel is None:
        from cffg import gauss

        d = group.dims[0]
        out = gauss.equality_energy(eta[..., :d], precision[..., :d, :d])
    else:
        out = group.kernel.binding.energy(
            eta,
            precision,
            params,
            group.dims,
            group.observed_role,
            observation,
            mask,
        )
    if out.shape != (eta.shape[0],):
        raise ValueError(
            f"group {group.name!r}: node energy must have one value per member"
        )
    return out


def gaussian_readout(
    group, eta, precision, params, observation=None, mask=None, *, statistics=False
):
    binding = group.kernel.binding
    callback = binding.statistics if statistics else binding.readout
    values = callback(
        eta, precision, params, group.dims, group.observed_role, observation, mask
    )
    for key, value in values.items():
        if not value.shape or value.shape[0] != eta.shape[0]:
            raise ValueError(
                f"group {group.name!r}: readout {key!r} needs a member axis"
            )
    return values


def validate_gaussian_shapes(group, packed):
    params = {
        key: jax.ShapeDtypeStruct(
            value.shape, jax.dtypes.canonicalize_dtype(value.dtype)
        )
        for key, value in packed.items()
    }
    dtype = jax.dtypes.canonicalize_dtype(np.float64)
    m, r, width = len(group.members), len(group.roles), max(group.dims)
    eta = jax.ShapeDtypeStruct((m, r, width), dtype)
    precision = jax.ShapeDtypeStruct((m, r, width, width), dtype)
    observation = mask = None
    if group.observed_role is not None:
        observation = jax.ShapeDtypeStruct((m, group.observed_dim), dtype)
        mask = jax.ShapeDtypeStruct((m,), dtype)
    args = (eta, precision, params, observation, mask)
    jax.eval_shape(partial(gaussian_message, group), *args)
    jax.eval_shape(partial(gaussian_energy, group), *args)
