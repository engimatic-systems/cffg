"""Bounded device messages for independent continuous interval potentials (D13a).

Each coordinate has a sorted prefix of disjoint half-open intervals with
finite log heights. Its density is zero elsewhere. Infinite tail endpoints
are allowed. Unused array cells carry no information. Apply jax.vmap to an
operation for independent instances; coordinates within one message are a
product, so one failing coordinate invalidates the whole operation.
"""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class IntervalStatus(IntEnum):
    OK = 0
    INFEASIBLE = 1
    IMPROPER = 2
    CAPACITY = 3
    NUMERICAL = 4
    INVALID = 5


class IntervalMessage(NamedTuple):
    lower: jax.Array  # (dimension, capacity)
    upper: jax.Array
    log_height: jax.Array
    count: jax.Array  # (dimension,)


class IntervalResult(NamedTuple):
    message: IntervalMessage
    status: jax.Array


class IntervalIntegral(NamedTuple):
    log_mass: jax.Array
    entropy: jax.Array
    status: jax.Array


def unit(dimension, capacity, dtype):
    """Construct a unit likelihood on the whole real line in each coordinate."""
    if (
        dimension < 1
        or capacity < 1
        or jnp.dtype(dtype) not in (jnp.float32, jnp.float64)
    ):
        raise ValueError(
            "interval messages require positive sizes and a floating dtype"
        )
    zero = jnp.zeros((dimension, capacity), dtype=dtype)
    return IntervalMessage(
        zero.at[:, 0].set(-jnp.inf),
        zero.at[:, 0].set(jnp.inf),
        zero,
        jnp.ones(dimension, dtype=jnp.int32),
    )


def _mask(message):
    lower, upper, heights, count = message
    if (
        lower.ndim != 2
        or min(lower.shape) < 1
        or upper.shape != lower.shape
        or heights.shape != lower.shape
        or count.shape != (lower.shape[0],)
        or lower.dtype not in (jnp.float32, jnp.float64)
        or upper.dtype != lower.dtype
        or heights.dtype != lower.dtype
        or count.dtype != jnp.int32
    ):
        raise ValueError("incompatible interval array dimensions or dtypes")
    return jnp.arange(lower.shape[1]) < count[:, None]


def _valid(message, live):
    lower, upper, heights, count = message
    ordered = jnp.all(~live[:, 1:] | (lower[:, 1:] >= upper[:, :-1]))
    return (
        ordered
        & jnp.all((count >= 0) & (count <= lower.shape[1]))
        & jnp.all(~live | ((lower < upper) & jnp.isfinite(heights)))
    )


def _finish(message, status):
    # Failure payloads are empty sentinels. A caller commits only OK results.
    ok = status == IntervalStatus.OK
    return IntervalResult(jax.tree.map(lambda x: jnp.where(ok, x, 0), message), status)


def _compress(lower, upper, heights, active, capacity):
    indices = jnp.nonzero(active, size=capacity, fill_value=0)[0]
    count = jnp.sum(active, dtype=jnp.int32)
    mask = jnp.arange(capacity) < count
    return IntervalMessage(
        *(jnp.where(mask, a[indices], 0) for a in (lower, upper, heights)), count
    )


def _transform(message, knot, offset, left, right, *, forward):
    live = _mask(message)
    lower, upper, height, _ = message
    params = tuple(jnp.asarray(x) for x in (knot, offset, left, right))
    if any(x.shape != lower.shape[:1] or x.dtype != lower.dtype for x in params):
        raise ValueError(
            "activation parameters must match coordinate dimension and dtype"
        )
    knot, offset, left, right = (a[:, None] for a in params)
    valid = (
        _valid(message, live)
        & jnp.all(jnp.isfinite(jnp.stack(params)))
        & jnp.all((left > 0) & (right > 0))
    )
    src, dst = (knot, offset) if forward else (offset, knot)

    def endpoint(x):
        slope = jnp.where(x < src, left, right)
        delta = x - src
        moved = dst + (delta * slope if forward else delta / slope)
        return jnp.where(jnp.isinf(x), x, moved)

    lo, hi = endpoint(lower), endpoint(upper)
    numeric = jnp.all(
        ~live
        | (
            (lo < hi)
            & (~jnp.isfinite(lower) | jnp.isfinite(lo))
            & (~jnp.isfinite(upper) | jnp.isfinite(hi))
        )
    ) & jnp.all(
        ~live[:, 1:] | (lower[:, 1:] == upper[:, :-1]) | (lo[:, 1:] > hi[:, :-1])
    )
    # For a bend, a distinct endpoint rounded onto its knot can erase a
    # positive-width side even if the whole transformed cell is nonempty.
    for x, moved in ((lower, lo), (upper, hi)):
        same_side = ((x < src) == (moved < dst)) & ((x > src) == (moved > dst))
        numeric &= jnp.all(~live | (left == right) | same_side)
    cut = live & (lower < src) & (upper > src) & (left != right)
    if forward:
        # One occupied interval may split into two different Jacobian heights.
        numeric &= jnp.all(~cut | ((lo < dst) & (dst < hi)))
        first = height - jnp.log(jnp.where(lower < src, left, right))
        second = height - jnp.log(right)
        numeric &= jnp.all(~live | jnp.isfinite(first)) & jnp.all(
            ~cut | jnp.isfinite(second)
        )

        def pairs(a, b):
            return jnp.stack(jnp.broadcast_arrays(a, b), axis=-1).reshape(
                a.shape[0], -1
            )

        packed = jax.vmap(_compress, in_axes=(0, 0, 0, 0, None))(
            pairs(lo, dst),
            pairs(jnp.where(cut, dst, hi), hi),
            pairs(first, second),
            pairs(live, cut),
            lower.shape[1],
        )
    else:
        # A constant likelihood stays constant across the bend.
        packed = IntervalMessage(
            jnp.where(live, lo, 0),
            jnp.where(live, hi, 0),
            jnp.where(live, height, 0),
            message.count,
        )
    status = jnp.select(
        [~valid, ~numeric, jnp.any(packed.count > lower.shape[1])],
        [IntervalStatus.INVALID, IntervalStatus.NUMERICAL, IntervalStatus.CAPACITY],
        default=IntervalStatus.OK,
    )
    return _finish(packed, status)


def pushforward(message, knot, offset, left, right):
    """Transport a density, dividing the height of each mapped cell by its slope."""
    return _transform(message, knot, offset, left, right, forward=True)


def pullback(message, knot, offset, left, right):
    """Compose an output likelihood with the activation on the input axis."""
    return _transform(message, knot, offset, left, right, forward=False)


def _intersect_coordinate(a, b):
    capacity = a.lower.shape[0]
    zero = jnp.zeros_like(a.lower)
    empty = IntervalMessage(zero, zero, zero, jnp.int32(0))

    def more(carry):
        i, j, _, _ = carry
        return (i < a.count) & (j < b.count)

    def advance(carry):
        i, j, out, valid = carry
        lo, hi = (
            jnp.maximum(a.lower[i], b.lower[j]),
            jnp.minimum(a.upper[i], b.upper[j]),
        )
        log_h = a.log_height[i] + b.log_height[j]
        keep = lo < hi

        def append(out):
            n = out.count
            # A count beyond capacity is still recorded for the atomic failure.
            return IntervalMessage(
                out.lower.at[n].set(lo, mode="drop"),
                out.upper.at[n].set(hi, mode="drop"),
                out.log_height.at[n].set(log_h, mode="drop"),
                n + 1,
            )

        out = jax.lax.cond(keep, append, lambda x: x, out)
        ax, bx = a.upper[i], b.upper[j]
        return (
            i + (ax <= bx),
            j + (bx <= ax),
            out,
            valid & (~keep | jnp.isfinite(log_h)),
        )

    # Clamp invalid counts before using them for loop termination. The public
    # entry point separately rejects malformed inputs, including NaN endpoints.
    a = a._replace(count=jnp.clip(a.count, 0, capacity))
    b = b._replace(count=jnp.clip(b.count, 0, capacity))

    def bounded(_, carry):
        return jax.lax.cond(more(carry), advance, lambda x: x, carry)

    _, _, result, numeric = jax.lax.fori_loop(
        0, 2 * capacity, bounded, (jnp.int32(0), jnp.int32(0), empty, jnp.asarray(True))
    )
    return result, numeric


def product(a, b):
    """All pairwise intersections, in order, or an explicit capacity failure."""
    ma, mb = _mask(a), _mask(b)
    if a.lower.shape != b.lower.shape or a.lower.dtype != b.lower.dtype:
        raise ValueError("interval products require matching shape and dtype")
    out, numeric = jax.vmap(_intersect_coordinate)(a, b)
    valid = _valid(a, ma) & _valid(b, mb)
    status = jnp.select(
        [~valid, ~jnp.all(numeric), jnp.any(out.count > a.lower.shape[1])],
        [IntervalStatus.INVALID, IntervalStatus.NUMERICAL, IntervalStatus.CAPACITY],
        default=IntervalStatus.OK,
    )
    return _finish(out, status)


def integral(message):
    """Total log mass and differential entropy of the normalized product belief."""
    live = _mask(message)
    valid = _valid(message, live)
    low, high, height, count = message
    finite = live & jnp.isfinite(low) & jnp.isfinite(high)
    lo, hi = jnp.where(finite, low, 0), jnp.where(finite, high, 1)
    # On an interval crossing zero the physical width can exceed float max.
    crosses = (lo < 0) & (hi > 0)
    log_width = jnp.where(
        crosses, jnp.logaddexp(jnp.log(-lo), jnp.log(hi)), jnp.log(hi - lo)
    )
    peak = jnp.max(jnp.where(finite, height, -jnp.inf), axis=1, keepdims=True)
    peak = jnp.where(jnp.any(finite, axis=1, keepdims=True), peak, 0)
    terms = jnp.where(finite, (height - peak) + log_width, -jnp.inf)
    relative = logsumexp(terms, axis=1, keepdims=True)
    log_p = jnp.where(finite, terms - relative, 0)
    entropy = jnp.sum(jnp.where(finite, jnp.exp(log_p) * (log_width - log_p), 0))
    log_mass = jnp.sum(peak + relative)
    impossible = jnp.any(count == 0)
    improper = jnp.any(live & ~finite)
    numeric = (
        jnp.all(~finite | jnp.isfinite(terms))
        & jnp.isfinite(log_mass)
        & jnp.isfinite(entropy)
    )
    status = jnp.select(
        [~valid, impossible, improper, ~numeric],
        [
            IntervalStatus.INVALID,
            IntervalStatus.INFEASIBLE,
            IntervalStatus.IMPROPER,
            IntervalStatus.NUMERICAL,
        ],
        default=IntervalStatus.OK,
    )
    ok = status == IntervalStatus.OK
    log_mass = jnp.select(
        [ok, status == IntervalStatus.INFEASIBLE, status == IntervalStatus.IMPROPER],
        [log_mass, -jnp.inf, jnp.inf],
        default=jnp.nan,
    )
    return IntervalIntegral(log_mass, jnp.where(ok, entropy, jnp.nan), status)
