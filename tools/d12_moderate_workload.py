"""D12 workload construction, imported only after native float32 is configured."""

import jax
import jax.numpy as jnp

from cffg import (
    hybrid_joint as joint,
    hybrid_joint_solver as solver,
    hybrid_joint_storage as storage,
)
from cffg.hybrid_natural import Natural
from cffg.hybrid_separator import TransitionParameters
from d12_compile_scaling import checked_arrays
from d12_moderate_plan import MODES


def raw_coordinate(coordinate, xi):
    return (
        jnp.zeros((coordinate.size,), coordinate.origin.dtype)
        .at[coordinate.message_size :]
        .set(xi)
    )


def initialize(coordinate, parameters, sites, xi, mode):
    """Three frozen starts; mode is dynamic so one executable serves all six."""
    return jax.lax.cond(
        mode == MODES.index("raw-zero"),
        lambda _: solver.initial_state_from_coordinate(
            coordinate, parameters, sites, raw_coordinate(coordinate, xi)
        ),
        lambda _: solver.initial_state(
            coordinate, parameters, sites, xi, perturb=mode == MODES.index("perturbed")
        ),
        operand=None,
    )


def output_values(coordinate, state):
    """Full State plus natural, physical belief and physical learning readouts."""
    messages, xi = coordinate.unpack(state.coordinate)
    result = dict(
        state=state, messages=dict(left=messages[0], right=messages[1]), xi=xi
    )
    for name in ("regions", "separators"):
        belief = getattr(state.evaluation.local, name)
        result[name] = dict(
            mean=belief.mean,
            second=belief.second,
            full_covariance=belief.full_covariance,
        )
    learning = state.evaluation.auxiliary
    result["learning"] = dict(
        physical_linear=jax.vmap(lambda a: a.physical_linear)(learning),
        physical_precision=jax.vmap(lambda a: a.physical_precision)(learning),
    )
    return result


def wide_context(t=100, k=20, d=10):
    """Shape-only placeholder context. These are not additional numerical cases."""
    sites = Natural(
        jnp.zeros((t, k), jnp.float32),
        jnp.zeros((t, k, d), jnp.float32),
        jnp.broadcast_to(jnp.eye(d, dtype=jnp.float32), (t, k, d, d)),
        jnp.zeros(t, jnp.int32),
        jnp.zeros((t, k, d), jnp.float32),
    )
    shapes = (
        (t - 1, k, k - 1, d),
        (t - 1, k, k - 1),
        (t - 1, k, k - 1),
        (t - 1, k, d, d),
        (t - 1, k, d),
        (t - 1, k, d, d),
    )
    parameters = TransitionParameters(
        *(jax.ShapeDtypeStruct(s, jnp.float32) for s in shapes)
    )
    return joint.Coordinate.from_sites(sites), parameters, sites


def logical_inventory(coordinate, parameters, sites):
    """Trace actual enclosing buffer trees; no held evaluation or optimizer call."""
    vector = jax.ShapeDtypeStruct((coordinate.size,), coordinate.origin.dtype)
    shapes = jax.eval_shape(
        lambda p, s, x: storage.buffer_values(coordinate, p, s, x),
        parameters,
        sites,
        vector,
    )
    groups = {name: checked_arrays(value) for name, value in shapes.items()}
    return dict(
        N=coordinate.size,
        M=coordinate.residual_size,
        groups=groups,
        meaning="Abstract logical leaf inventory, not executed buffers or physical peak; aliases may repeat",
    )


def compile_arguments(coordinate, parameters, sites, program):
    vector = jax.ShapeDtypeStruct((coordinate.size,), coordinate.origin.dtype)
    if program == "solve_from_coordinate":
        return parameters, sites, vector
    if program == "iteration":
        state = jax.eval_shape(
            lambda p, s, x: solver.initial_state_from_coordinate(coordinate, p, s, x),
            parameters,
            sites,
            vector,
        )
        return parameters, sites, state
    raise ValueError(f"unknown complete storage program: {program}")
