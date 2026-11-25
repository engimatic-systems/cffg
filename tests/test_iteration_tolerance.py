"""Threshold decisions match the existing Python-float convergence policy."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.iteration_tolerance import energy_within, tolerance


@pytest.mark.parametrize("x64", [False, True])
def test_compiled_energy_bound_matches_host_rounding_and_overflow(x64):
    largest = float(np.finfo(np.float32).max)
    midpoint32 = 1. + 2.**-24
    cases = [
        # Float32 subtraction rounds these to the same returned high word.
        (.25, -.25 * 2.**-26, .25),
        (.25, -2.**-55, .25),  # even double accepts the exact halfway value
        (.25, -2.**-54, .25),
        (.25, -3. * 2.**-55, math.nextafter(.25, math.inf)),  # odd double rejects halfway
        (.25, -2.**-54, math.nextafter(.25, math.inf)),
        (.25, 2.**-54, math.nextafter(.25, 0.)),
        (1., -2.**-24, midpoint32),
        (1., -2.**-24, math.nextafter(midpoint32, 0.)),
        (largest, -largest, largest),
        (largest, -largest, math.nextafter(2 * largest, 0.)),
        (largest, -largest, 2 * largest),
        (largest, -largest, math.inf),
        (.25, .25, None),
        (.25, .25, math.nan),
        (np.inf, largest, math.inf),
    ]
    with jax.enable_x64(x64):
        check = jax.jit(jax.vmap(energy_within, in_axes=(0, 0, 0)))
        caps = jax.tree.map(
            lambda *values: np.asarray(values),
            *(tolerance(np.float32, limit) for _, _, limit in cases),
        )
        left = np.asarray([a for a, _, _ in cases], np.float32)
        right = np.asarray([b for _, b, _ in cases], np.float32)
        expected = [
            limit is not None and np.isfinite(a) and np.isfinite(b)
            and abs(float(a) - float(b)) <= max(limit, 64 * float(np.finfo(np.float32).eps))
            for a, b, (_, _, limit) in zip(left, right, cases, strict=True)
        ]
        np.testing.assert_array_equal(check(left, right, caps), expected)
        np.testing.assert_array_equal(check(right, left, caps), expected)


@pytest.mark.parametrize("x64", [False, True])
def test_delta_comparison_does_not_round_up_tolerance(x64):
    with jax.enable_x64(x64):
        compare = jax.jit(lambda delta, caps: delta <= caps.delta)
        requested = math.nextafter(.25, 0.)
        delta = jnp.float32(.25)
        assert not compare(delta, tolerance(np.float32, requested))
        assert compare(delta, tolerance(np.float32, .25))
        assert not compare(jnp.float32(0), tolerance(np.float32, None))
        assert not compare(jnp.float32(0), tolerance(np.float32, math.nan))
        minimum = jnp.float32(64 * np.finfo(np.float32).eps)
        assert compare(minimum, tolerance(np.float32, 0))


def test_float64_uses_the_host_rounded_difference():
    with jax.enable_x64(True):
        limits = tolerance(np.float64, .25)
        first = jnp.asarray(.25, dtype=jnp.float64)
        evaluate = jax.jit(energy_within)
        assert evaluate(first, jnp.asarray(-2.**-55), limits)
        assert not evaluate(first, jnp.asarray(-2.**-54), limits)
