"""Budgeted candidate sweeps on an established feasible support face."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg.compiled import NEG
from cffg.outer_chart import OuterArrays


class Candidate(NamedTuple):
    current: OuterArrays
    mapped: OuterArrays
    inner: jax.Array
    count: jax.Array
    maps: jax.Array
    deltas: jax.Array
    advance: jax.Array


def array_delta(left, right):
    return jnp.max(jnp.stack([
        jnp.max(jnp.abs(a - b), initial=0.) for a, b in zip(left, right, strict=True)
    ]))


def finite(arrays):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in arrays]))


def make_candidate(program, capacity):
    def run(current, mapped, inner, context, budget, tolerance):
        # Feasibility only depends on discrete supports and the fixed factor parameters.
        # Include eliminated slots in the reconstructed support witness.
        base_support = current.messages != NEG
        rebuilt = program.reconstruct_arrays(current, context)[0]
        feasible_support = rebuilt.messages != NEG

        def step(row):
            trial = OuterArrays(*(0.5 * a + 0.5 * b for a, b in zip(row.current, row.mapped, strict=True)))
            before = program.reconstruct_arrays(trial, context)[0]
            following, residual, code, _ = program.map_arrays(before, context)
            after = program.reconstruct_arrays(following, context)[0]
            same_face = jnp.all((following.messages != NEG) == base_support)
            feasible = (jnp.all((before.messages != NEG) == feasible_support)
                        & jnp.all((after.messages != NEG) == feasible_support))
            valid = (same_face & feasible & finite(trial) & finite(following)
                     & jnp.isfinite(residual) & (code == 0))
            # A boundary is re-evaluated and classified by the complete host policy.
            accepted = Candidate(trial, following, residual, row.count + 1, row.maps + 1,
                                 row.deltas.at[row.count].set(array_delta(row.current, trial)), True)
            return jax.lax.cond(valid, lambda: accepted,
                                lambda: row._replace(maps=row.maps + 1, advance=False))

        def keep(row):
            return row.advance & (row.count < budget) & (array_delta(row.current, row.mapped) > tolerance)

        start = Candidate(current, mapped, inner, jnp.int32(0), jnp.int32(0),
                          jnp.zeros(capacity, current.messages.dtype), True)
        return jax.lax.while_loop(keep, step, start)
    return run
