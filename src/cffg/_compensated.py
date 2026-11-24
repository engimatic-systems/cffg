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
        # At +2**p the next float is two units away. At -2**p the
        # positive neighbor is still one unit away (p includes the hidden bit).
        unit_limit = float(2**(jnp.finfo(new_low.dtype).nmant + 1))
        resolves_unit = (new_low >= -unit_limit) & (new_low < unit_limit)
        ok = jnp.all((lost == 0) | resolves_unit)
    return (jnp.where(increment != 0, new_high, high),
            jnp.where(increment != 0, new_low, low), ok)


def canonical(high, low):
    """The high word is a nearest float; either endpoint represents a midpoint."""
    rounded = jax.lax.optimization_barrier(high + low)
    # Round-to-nearest either returns high or chooses the other endpoint
    # of an exact midpoint. In that case their difference is exactly 2*low.
    displacement = jax.lax.optimization_barrier(rounded - high)
    return jnp.all(jnp.isfinite(rounded) & ((rounded == high) | (0.5 * displacement == low)))
