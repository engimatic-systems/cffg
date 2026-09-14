"""Exact stored-input references for compensated D12 affine coordinates."""

from fractions import Fraction
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.hybrid_affine import affine, _two_product
from cffg import hybrid_natural as hn, hybrid_separator as hs
from hybrid_kernel_cases import tolerance
from test_hybrid_centering import local


CASES = ((np.float32, 10000, 1e-4, [6.7, 1.3], -80000),
         (np.float64, 1e8, 1, [.3, .7], -1e8),
         (np.float32, 10000, 1e-4, [6.7], -67000),
         (np.float64, 1e8, 1, [.7], -7e7),
         (np.float32, 1e8, 1e-4, [.7], -7e7))


def exact_affine(weights, values, *offsets):
    return sum((Fraction(float(w)) * Fraction(float(x)) for w, x in zip(weights, values)),
               sum((Fraction(float(b)) for b in offsets), Fraction(0)))


def case(dtype, origin, variance, weights, offset):
    dim = len(weights)
    center = jnp.full((2, dim), origin, dtype)
    covariance = jnp.broadcast_to(jnp.eye(dim, dtype=dtype) * variance, (2, dim, dim))
    left = hn.gaussian_site(jnp.log(jnp.array([.4, .6], dtype)), center, covariance)
    right = hn.Natural(jnp.zeros(2, dtype), jnp.zeros((2, dim), dtype), jnp.zeros((2, dim, dim), dtype))
    params = hs.TransitionParameters(jnp.broadcast_to(jnp.asarray(weights, dtype), (2, 1, dim)),
                                    jnp.full((2, 1), offset, dtype), jnp.zeros((2, 1), dtype),
                                    jnp.zeros((2, dim, dim), dtype), jnp.zeros((2, dim), dtype),
                                    jnp.broadcast_to(jnp.eye(dim, dtype=dtype), (2, dim, dim)))
    return params, left, right


@pytest.mark.parametrize("dtype,origin,variance,weights,offset", CASES)
def test_nonbinary_local_paths_match_exact_stored_input_translation(dtype, origin, variance, weights, offset):
    weights = np.asarray(weights, dtype)
    translated_offset = dtype(exact_affine(weights, np.full(len(weights), origin, dtype), dtype(offset)))
    inputs = case(dtype, origin, variance, weights, offset)
    for evaluate in (local, jax.jit(local)):
        baseline = evaluate(*case(dtype, 0, variance, weights, translated_offset))
        result = evaluate(*inputs)
        region, messages, energy, status, aux = result
        hn.raise_for_status(region.status | status | aux.status)
        for field in ("probability", "displacement", "covariance", "log_normalizer"):
            np.testing.assert_allclose(getattr(region, field), getattr(baseline[0], field),
                                       atol=tolerance(dtype), rtol=tolerance(dtype))
        assert energy == pytest.approx(float(baseline[2]), abs=tolerance(dtype))
        for actual, reference in zip(messages, baseline[1]):
            for a, b in zip(actual.arrays(), reference.arrays()):
                np.testing.assert_allclose(a, b, atol=tolerance(dtype), rtol=tolerance(dtype))
        for a, b in zip(aux[:5], baseline[4][:5]):
            np.testing.assert_allclose(a, b, atol=tolerance(dtype), rtol=tolerance(dtype))

        # Independent scalar Gaussian/JJ identity, using the exact stored
        # affine residual before rounding it once to the declared dtype.
        squared_weight = np.square(weights.astype(float)).sum()
        denominator = 1 + variance * squared_weight / 4
        logit = float(translated_offset) / denominator
        destination = 1 / (1 + np.exp(-logit))
        expected = np.array([.4, .6])[:, None] * [destination, 1 - destination]
        np.testing.assert_allclose(region.probability, expected, atol=tolerance(dtype), rtol=0)
        expected_logz = (-np.log(denominator) / 2 - float(translated_offset) ** 2 / (8 * denominator)
                        + variance * squared_weight / (8 * denominator)
                        + np.logaddexp(logit / 2, -logit / 2) - np.log(2))
        assert region.log_normalizer == pytest.approx(expected_logz, abs=tolerance(dtype))


@pytest.mark.parametrize("dtype,origin", ((np.float32, 10000), (np.float64, 1e8)))
def test_channel_bias_uses_separate_additive_terms_and_preserves_owned_energy(dtype, origin):
    params, left, right = case(dtype, origin, 1e-4 if dtype == np.float32 else 1, [0, 0], 0)
    matrix = np.array([[[6.7, 1.3], [1.3, 6.7]], [[.3, .7], [.7, .3]]], dtype)
    bias = np.array([[-8 * origin, -8 * origin], [-origin, -origin]], dtype)
    exact_bias = np.array([[float(exact_affine(row, [dtype(origin)] * 2, b)) for row, b in zip(a, v)]
                          for a, v in zip(matrix, bias)], dtype)
    params = params._replace(matrix=jnp.asarray(matrix), bias=jnp.asarray(bias))
    right = hn.gaussian_site(jnp.log(jnp.array([.55, .45], dtype)), right.linear,
                              jnp.broadcast_to(jnp.eye(2, dtype=dtype), (2, 2, 2)))
    centered_left = left._replace(center=jnp.zeros_like(left.center))
    reference = local(params._replace(bias=jnp.asarray(exact_bias)), centered_left, right)
    for evaluate in (local, jax.jit(local)):
        result = evaluate(params, left, right)
        hn.raise_for_status(result[0].status | result[3] | result[4].status)
        for field in ("probability", "displacement", "covariance", "log_normalizer"):
            np.testing.assert_allclose(getattr(result[0], field), getattr(reference[0], field),
                                       atol=tolerance(dtype), rtol=tolerance(dtype))
        assert result[2] == pytest.approx(float(reference[2]), abs=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_integer_product_words_sum_to_exact_stored_product_and_affine_batches_match(dtype):
    rng = np.random.default_rng(812)
    scale = 40 if dtype == np.float32 else 400
    a = np.ldexp(rng.uniform(-1, 1, 64), rng.integers(-scale, scale, 64)).astype(dtype)
    b = np.ldexp(rng.uniform(-1, 1, 64), rng.integers(-scale, scale, 64)).astype(dtype)
    for evaluate in (_two_product, jax.jit(_two_product)):
        high, low = map(np.asarray, evaluate(jnp.asarray(a), jnp.asarray(b)))
        for w, x, h, l in zip(a, b, high, low):
            assert Fraction(float(w)) * Fraction(float(x)) == Fraction(float(h)) + Fraction(float(l))
    weights = jnp.asarray([[6.7, 1.3], [.3, .7]], dtype)
    values = jnp.full((2, 2), 10000, dtype)
    offsets = jnp.asarray([-80000, -10000], dtype)
    expected = np.array([float(exact_affine(w, x, o)) for w, x, o in zip(weights, values, offsets)], dtype)
    for evaluate in (affine, jax.jit(affine), jax.vmap(affine), jax.jit(jax.vmap(affine))):
        np.testing.assert_array_equal(evaluate(weights, values, offsets), expected)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_compensated_affine_has_analytic_forward_reverse_and_finite_difference_derivatives(dtype):
    w, x, offset = jnp.array([6.7, 1.3], dtype), jnp.full(2, 10000, dtype), jnp.array(-80000, dtype)
    for derivative in (jax.jacfwd(affine, argnums=(0, 1, 2)), jax.jacrev(affine, argnums=(0, 1, 2))):
        actual = jax.jit(derivative)(w, x, offset)
        for a, b in zip(actual, (x, w, jnp.ones((), dtype))):
            np.testing.assert_array_equal(a, b)
    step = np.finfo(dtype).eps ** (1 / 3)
    difference = (affine(w, x, offset, jnp.asarray(step, dtype))
                  - affine(w, x, offset, jnp.asarray(-step, dtype))) / (2 * step)
    assert difference == pytest.approx(1, abs=tolerance(dtype))
    params, left, right = case(dtype, 10000, 1e-4, [6.7, 1.3], -80000)
    region, _, _, _, auxiliary = local(params, left, right)
    derivative = jax.grad(lambda xi: -hs.transition_expectation(region, params._replace(xi=xi))[0])(params.xi)
    np.testing.assert_allclose(derivative, auxiliary.exposure * auxiliary.residual / 96,
                               atol=64 * np.finfo(dtype).eps * float(auxiliary.target.max()), rtol=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_unrepresentable_product_words_and_partial_sum_overflow_fail_explicitly(dtype):
    info = np.finfo(dtype)
    cases = (([info.max], [2], 0), ([info.tiny], [.1], 0), ([np.inf], [0], 0),
             ([info.max, info.max], [1, 1], -info.max))
    for weights, values, offset in cases:
        for evaluate in (affine, jax.jit(affine)):
            value = evaluate(jnp.array(weights, dtype), jnp.array(values, dtype), jnp.array(offset, dtype))
            assert np.isnan(value)


def test_float32_consumer_mode_without_x64_preserves_complete_review_reproduction():
    source = textwrap.dedent("""
        import jax
        jax.config.update('jax_enable_x64', False)
        import jax.numpy as jnp
        from cffg.hybrid_affine import affine
        from cffg import hybrid_natural as hn, hybrid_separator as hs
        w = jnp.array([6.7, 1.3], jnp.float32)
        x = jnp.full(2, 10000, jnp.float32)
        o = jnp.array(-80000, jnp.float32)
        exact = jnp.array(-0.002384185791015625, jnp.float32)
        for evaluate in (affine, jax.jit(affine)):
            value = evaluate(w, x, o)
            assert value.dtype == jnp.float32 and value == exact
        def tilt(center, offset):
            left = hn.gaussian_site(jnp.log(jnp.array([.4, .6], jnp.float32)),
                                    jnp.full((2, 2), center, jnp.float32),
                                    jnp.broadcast_to(jnp.eye(2, dtype=jnp.float32) * 1e-4, (2, 2, 2)))
            right = hn.Natural(jnp.zeros(2, jnp.float32), jnp.zeros((2, 2), jnp.float32),
                                jnp.zeros((2, 2, 2), jnp.float32))
            params = hs.TransitionParameters(jnp.broadcast_to(w, (2, 1, 2)), jnp.full((2, 1), offset),
                                            jnp.zeros((2, 1), jnp.float32), jnp.zeros((2, 2, 2), jnp.float32),
                                            jnp.zeros((2, 2), jnp.float32),
                                            jnp.broadcast_to(jnp.eye(2, dtype=jnp.float32), (2, 2, 2)))
            factor = hs.transition(*params, left_center=left.origin, right_center=right.origin)
            belief = hn.normalize(factor + hn.lift(left, right))
            energy, status = hs.transition_expectation(belief, params)
            return belief.probability, energy, belief.status | status
        for evaluate in (tilt, jax.jit(tilt)):
            a, b = evaluate(jnp.array(10000, jnp.float32), o), evaluate(jnp.array(0, jnp.float32), exact)
            hn.raise_for_status(a[2] | b[2])
            assert a[0].dtype == jnp.float32 and a[1].dtype == jnp.float32
            assert jnp.max(abs(a[0] - b[0])) <= 512 * 4 * jnp.finfo(jnp.float32).eps
            assert abs(a[1] - b[1]) <= 512 * 4 * jnp.finfo(jnp.float32).eps
        assert not jax.config.x64_enabled
    """)
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
