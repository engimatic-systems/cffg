"""Typed outputs and numerical facts at the local-update boundary."""

from typing import NamedTuple

import jax


class ExecutionDelta[T](NamedTuple):
    """One plan's outgoing blocks and its existing local solver facts.

    Numerical facts stay arrays for JAX transformations. ``None`` means the
    rule supplies no such fact, not that a solver converged. Exact support
    remains encoded in the outgoing blocks; a successful solver status does
    not replace downstream support checks. The output structure is static for
    each compiled plan, including rules that return no solver diagnostics.
    """

    outgoing: T
    residual: jax.Array | None = None
    status: jax.Array | None = None
