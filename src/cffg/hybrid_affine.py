"""Same-dtype compensated affine arithmetic for fixed D12 origins.

TwoSum, integer-significand TwoProduct and compensated dot accumulation
retain product roundoff before large affine terms cancel. Addition rounding
boundaries have optimization barriers. Integer product decomposition avoids
compiler FMA contraction and splitter overflow.

This is compensated precision, not arbitrary precision or universal exactness.
Unrepresentable products, subnormal addends/results and overflowing partial sums
fail with NaN so the owning hybrid kernel reports NONFINITE. No precision floor is
inserted. Derivatives are those of the declared real affine expression, not
derivatives of its floating-point decomposition.

The compensated summation follows Ogita, Rump and Oishi, "Accurate Sum and
Dot Product", https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf. The two-limb
integer product below supplies the exact product error without floating FMA
assumptions; its bounds are stated beside the implementation.
"""

import jax
import jax.numpy as jnp


def _checked_sum(left, right, *, subtract=False):
    result = jax.lax.optimization_barrier(left - right if subtract else left + right)
    info = jnp.finfo(result.dtype)
    width = 32 if info.nmant == 23 else 64
    unsigned = jnp.uint32 if width == 32 else jnp.uint64
    sign = jnp.asarray(1 << (width - 1), unsigned)
    smallest_normal = jnp.asarray(1 << info.nmant, unsigned)
    left_bits, right_bits, result_bits = (jax.lax.bitcast_convert_type(x, unsigned)
                                          for x in (left, right, result))
    left_magnitude, right_magnitude, result_magnitude = (x & ~sign for x in
                                                        (left_bits, right_bits, result_bits))
    subnormal_input = ((left_magnitude != 0) & (left_magnitude < smallest_normal)
                       | (right_magnitude != 0) & (right_magnitude < smallest_normal))
    different_signs = ((left_bits ^ right_bits) & sign) != 0
    cancellation = ~different_signs if subtract else different_signs
    # Unequal opposing normal magnitudes cannot sum exactly to zero. Inspect
    # bits because floating comparisons can flush a subnormal result/addend.
    lost = cancellation & (left_magnitude != right_magnitude) & (result_magnitude < smallest_normal)
    return jnp.where(subnormal_input | lost, jnp.nan, result)


def _add(left, right):
    return _checked_sum(left, right)


def _subtract(left, right):
    return _checked_sum(left, right, subtract=True)


def _two_sum(left, right):
    total = _add(left, right)
    virtual = _subtract(total, left)
    error = _add(_subtract(left, _subtract(total, virtual)), _subtract(right, virtual))
    return total, error


def _two_product(left, right):
    """Round an exact two-limb significand product and retain its remainder.

    Floating multiply/subtract expansions can contract after HLO optimization
    barriers are removed. Integer significands avoid that dependency entirely.
    A p-bit product rounds to p bits; its signed remainder has at most p bits,
    so both words convert exactly to the original float dtype before scaling.
    """
    info = jnp.finfo(left.dtype)
    precision = info.nmant + 1
    half = (precision + 1) // 2
    limb = 2 * half
    width = 32 if precision == 24 else 64
    unsigned = jnp.uint32 if width == 32 else jnp.uint64
    signed = jnp.int32 if width == 32 else jnp.int64
    one = jnp.asarray(1, unsigned)
    mask = (one << half) - one
    limb_mask = (one << limb) - one
    exponent_mask = (1 << info.iexp) - 1
    bias = (1 << (info.iexp - 1)) - 1

    def decode(value):
        bits = jax.lax.bitcast_convert_type(value, unsigned)
        exponent = ((bits >> info.nmant) & exponent_mask).astype(jnp.int32)
        mantissa = (bits & ((one << info.nmant) - one))
        mantissa |= jnp.where(exponent > 0, one << info.nmant, jnp.asarray(0, unsigned))
        power = jnp.maximum(exponent, 1) - bias - info.nmant
        return mantissa, power, bits >> (width - 1)

    a, a_power, a_sign = decode(left)
    b, b_power, b_sign = decode(right)
    cross = (a >> half) * (b & mask) + (a & mask) * (b >> half)
    low = (a & mask) * (b & mask) + ((cross & mask) << half)
    high = (a >> half) * (b >> half) + (cross >> half) + (low >> limb)
    low &= limb_mask
    length = jnp.where(high != 0, width - jax.lax.clz(high) + limb, width - jax.lax.clz(low))
    shift = jnp.maximum(length.astype(jnp.int32) - precision, 0).astype(unsigned)
    top = (high << (limb - shift)) | (low >> shift)
    remainder = low & ((one << shift) - one)
    midpoint = one << jnp.maximum(shift.astype(jnp.int32) - 1, 0).astype(unsigned)
    up = (shift > 0) & ((remainder > midpoint) | ((remainder == midpoint) & ((top & one) != 0)))
    rounded = top + up.astype(unsigned)
    remainder = remainder.astype(signed) - jnp.where(up, one << shift, jnp.asarray(0, unsigned)).astype(signed)
    power = a_power + b_power
    product = jnp.ldexp(rounded.astype(left.dtype), power + shift.astype(jnp.int32))
    error = jnp.ldexp(remainder.astype(left.dtype), power)
    product = jnp.where((a_sign ^ b_sign) != 0, -product, product)
    error = jnp.where((a_sign ^ b_sign) != 0, -error, error)
    # Subnormal scaling can double-round or flush a word; fail rather than
    # claim an error-free product when the dtype cannot retain its two words.
    normal_product = (rounded == 0) | (abs(product) >= info.tiny)
    normal_error = (remainder == 0) | (abs(error) >= info.tiny)
    valid = jnp.isfinite(left) & jnp.isfinite(right) & jnp.isfinite(product) & jnp.isfinite(error)
    valid &= normal_product & normal_error
    return jnp.where(valid, product, jnp.nan), jnp.where(valid, error, jnp.nan)


@jax.custom_jvp
def affine(weights, values, *offsets):
    """Evaluate sum(weights*values, axis=-1) plus uncombined offsets.

    Inputs broadcast over leading dimensions and keep float32 or float64.
    Supply each additive term separately: computing b-c first could discard
    information before this routine sees it. The storage is linear in the
    local dot-product dimension; it creates no trajectory or label cube.
    """
    if weights.dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise ValueError("affine inputs must use float32 or float64")
    if any(value.dtype != weights.dtype for value in (values, *offsets)):
        raise ValueError("affine input dtypes disagree")
    weights, values = jnp.broadcast_arrays(weights, values)
    product, error = _two_product(weights, values)
    initial = (jnp.zeros(product.shape[:-1], product.dtype),) * 2

    def accumulate(state, term):
        total, correction = state
        high, low = term
        total, remainder = _two_sum(total, high)
        correction = _add(correction, _add(remainder, low))
        return (total, correction), None

    (total, correction), _ = jax.lax.scan(
        accumulate, initial, (jnp.moveaxis(product, -1, 0), jnp.moveaxis(error, -1, 0)))
    for offset in offsets:
        total, remainder = _two_sum(total, jnp.broadcast_to(offset, total.shape))
        correction = _add(correction, remainder)
    result = _add(total, correction)
    return jnp.where(jnp.isfinite(result), result, jnp.nan)


@affine.defjvp
def _affine_jvp(primals, tangents):
    weights, values, *offsets = primals
    weight_tangent, value_tangent, *offset_tangents = tangents
    tangent = jnp.sum(weight_tangent * values + weights * value_tangent, axis=-1)
    for offset in offset_tangents:
        tangent += offset
    return affine(weights, values, *offsets), tangent


def centered_affine(weights, displacement, center, *offsets):
    """Evaluate an affine function of center+displacement without adding them."""
    weights, displacement, center = jnp.broadcast_arrays(weights, displacement, center)
    return affine(jnp.concatenate((weights, weights), axis=-1),
                  jnp.concatenate((center, displacement), axis=-1), *offsets)
