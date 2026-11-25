"""Device comparisons against the ordinary controller's Python-float tolerance."""

from fractions import Fraction
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Tolerance(NamedTuple):
    delta: np.floating
    energy_high: np.floating
    energy_low: np.floating
    halve_energy: bool


def _fraction(value):
    return Fraction.from_float(float(value))


def _float32_floor(bound, *, inclusive=True):
    """Largest float32 whose exact value is within the rational bound."""
    rounded = np.float32(float(bound))
    exact = _fraction(rounded)
    if exact > bound or (exact == bound and not inclusive):
        return np.nextafter(rounded, np.float32(-np.inf))
    return rounded


def _energy_cutoff(limit):
    # The host first rounds the difference to double. Its inclusive upper
    # bound therefore extends halfway to the next double, with ties to even.
    next_limit = math.nextafter(limit, math.inf)
    midpoint = (_fraction(limit) + _fraction(next_limit)) / 2
    even = not (int(np.float64(limit).view(np.uint64)) & 1)
    halve = limit >= float(np.finfo(np.float32).max)
    if halve:
        midpoint /= 2

    lower = _float32_floor(midpoint)
    higher = np.nextafter(lower, np.float32(np.inf))
    # The precision pair from FastTwoSum is normalized to nearest float32.
    # Reproduce that rounding directly on the exact threshold midpoint.
    adjacent_midpoint = (_fraction(lower) + _fraction(higher)) / 2
    lower_is_odd = bool(int(lower.view(np.uint32)) & 1)
    round_up = midpoint > adjacent_midpoint or (midpoint == adjacent_midpoint and lower_is_odd)
    high = higher if round_up else lower
    return high, _float32_floor(midpoint - _fraction(high), inclusive=even), halve


def tolerance(dtype, tol):
    """Prepare scalar controls for the previous host convergence criteria."""
    limit = -math.inf
    if tol is not None and not math.isnan(float(tol)):
        limit = max(float(tol), 64 * float(np.finfo(dtype).eps))
    if jax.config.x64_enabled:
        return Tolerance(np.float64(limit), np.float64(limit), np.float64(0), False)

    maximum = float(np.finfo(np.float32).max)
    if not math.isfinite(limit) or limit >= 2 * maximum:
        disabled_or_unbounded = np.float32(-np.inf if limit < 0 else np.inf)
        return Tolerance(disabled_or_unbounded, disabled_or_unbounded, np.float32(0), False)
    delta = _float32_floor(_fraction(limit)) if limit <= maximum else np.float32(np.inf)
    high, low, halve = _energy_cutoff(limit)
    return Tolerance(delta, high, low, halve)


def energy_within(fe, previous, controls):
    """Check the host's rounded absolute energy difference on the device."""
    if jax.config.x64_enabled:
        difference = jnp.abs(fe.astype(jnp.float64) - previous.astype(jnp.float64))
        return jnp.isfinite(fe) & jnp.isfinite(previous) & (difference <= controls.energy_high)

    # If large f32 differences can qualify, halving keeps the sum of both
    # magnitudes finite. Any lost subnormal is below half a double ULP there.
    scale = jnp.where(controls.halve_energy, np.float32(.5), np.float32(1))
    larger = jnp.maximum(fe, previous) * scale
    negative_smaller = -jnp.minimum(fe, previous) * scale
    swap = jnp.abs(larger) < jnp.abs(negative_smaller)
    first = jnp.where(swap, negative_smaller, larger)
    second = jnp.where(swap, larger, negative_smaller)
    # FastTwoSum. The barriers retain both rounding steps under fusion.
    high = jax.lax.optimization_barrier(first + second)
    remainder = jax.lax.optimization_barrier(high - first)
    low = second - remainder
    inside = (high < controls.energy_high) | (
        (high == controls.energy_high) & (low <= controls.energy_low)
    )
    unbounded = controls.energy_high == np.float32(np.inf)
    return jnp.isfinite(fe) & jnp.isfinite(previous) & (inside | unbounded)
