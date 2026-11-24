"""Source-precision high and low words for persistent sufficient statistics."""

import jax
import jax.numpy as jnp


def _two_sum(a, b):
    """Knuth's error-free sum for finite inputs and a representable result."""
    # Each intermediate must round once, including under XLA compilation.
    rounded = jax.lax.optimization_barrier
    high = rounded(a + b)
    b_part = rounded(high - a)
    a_part = rounded(high - b_part)
    return high, rounded(a - a_part) + rounded(b - b_part)


def add(high, low, increment, *, count=False):
    """Accumulate one rounded increment; counts must still resolve a whole trial."""
    increment = jax.lax.optimization_barrier(increment)
    total, error = _two_sum(high, increment)
    tail, lost = _two_sum(low, error)
    new_high, new_low = _two_sum(total, tail)
    # After cancellation or a carry to the high word there may be room for
    # the discarded bits. Renormalize once more, including half-ULP ties.
    tail, lost = _two_sum(new_low, lost)
    new_high, new_low = _two_sum(new_high, tail)
    ok = jnp.asarray(True)
    if count:
        # Tiny responsibility tails can round at the low word in a healthy
        # accumulator. Reject lost mass if even one whole trial is unresolved.
        spacing = jnp.nextafter(new_low, jnp.full_like(new_low, jnp.inf)) - new_low
        ok = jnp.all((lost == 0) | (spacing <= 1))
    return (jnp.where(increment != 0, new_high, high),
            jnp.where(increment != 0, new_low, low), ok)


def canonical(high, low):
    """The high word is a nearest float; either endpoint represents a midpoint."""
    direction = jnp.copysign(jnp.full_like(high, jnp.inf), low)
    neighbor = jnp.nextafter(high, direction)
    gap = jnp.abs(neighbor - high)
    # At the largest finite value the outward ULP equals the inward one.
    gap = jnp.where(jnp.isinf(neighbor), jnp.abs(high - jnp.nextafter(high, jnp.zeros_like(high))), gap)
    return jnp.all(jnp.isfinite(high + low) & (jnp.abs(low) <= gap * 0.5))
