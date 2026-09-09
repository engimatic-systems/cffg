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
