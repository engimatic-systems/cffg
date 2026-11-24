"""Source-precision continuation and relative scores from sub-ULP evidence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit

from cffg.mixed_learning import MixedBank, MixedBatch, score, update


def scalar_bank(dtype, mu, kappa, psi, count=3, size=1):
    def fill(value, shape):
        return jnp.full(shape, value, dtype=dtype)
    return MixedBank(fill(mu, (size, 1)), fill(kappa, (size,)), fill(psi, (size, 1, 1)),
                     fill(count, (size,)), fill(count, (size,)), (), jnp.ones(size, bool))


def one(dtype, mean, variance):
    return MixedBatch(jnp.array([[mean]], dtype), jnp.array([[[variance]]], dtype),
                      jnp.array([True]), ())


def stream(prior, record, weights, length):
    def step(state, _):
        bank, valid = state
        posterior, ok = update(bank, record, weights)
        return (posterior, valid & ok), None
    return jax.lax.scan(step, (prior, jnp.array(True)), None, length=length)[0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_translated_stream_keeps_local_mean_and_uncertainty(dtype):
    with jax.enable_x64(dtype == np.float64):
        bank = scalar_bank(dtype, 10000, 4096, 3)
        x = one(dtype, 10001, .125)
        w = jnp.ones((1, 1), dtype)
        posterior, valid = stream(bank, x, w, 1024)
        assert valid and posterior.mu.dtype == dtype
        eps = np.finfo(dtype).eps
        offset = (posterior.mu.item() - 10000.) + posterior.carry.mu.item()
        assert posterior.mu.item() == dtype(10000.2)
        np.testing.assert_allclose(offset, .2, atol=0, rtol=8*eps)
        np.testing.assert_allclose(posterior.psi.item(), 950.2, atol=0, rtol=8*eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_two_records_at_one_ulp_survive_exact_restart(dtype):
    with jax.enable_x64(dtype == np.float64):
        h = np.spacing(dtype(4096))
        bank = scalar_bank(dtype, 4096, 1, h*h)
        x = one(dtype, 4096 + h, 0)
        w = jnp.ones((1, 1), dtype)
        compiled = jax.jit(update)
        half, valid = update(bank, x, w)
        assert valid and half.mu.item() == 4096 and half.carry.mu.item() == h/2
        restored = MixedBank.from_state_dict(half.state_dict(exact=True))
        split, valid = compiled(restored, x, w)
        assert valid
        doubled = jax.tree.map(lambda row: jnp.repeat(row, 2, axis=0), x)
        full, valid = compiled(bank, doubled, jnp.repeat(w, 2, axis=0))
        assert valid
        for result in (split, full):
            offset = (result.mu.item() - 4096) + result.carry.mu.item()
            np.testing.assert_allclose(offset / h, 2/3, rtol=4*np.finfo(dtype).eps)
            np.testing.assert_allclose(result.psi.item() / h**2, 5/3, rtol=4*np.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fractional_counts_accumulate_for_observed_fields_only(dtype):
    with jax.enable_x64(dtype == np.float64):
        count = dtype(2.**(np.finfo(dtype).nmant - 1))
        bank = scalar_bank(dtype, 0, count, 1, count=count, size=3)
        head = jnp.full((3, 2), count, dtype)
        bank = bank._replace(categories=(head, head), active=jnp.array([True, True, False]))
        x = one(dtype, 0, 0)._replace(categories=(jnp.array([1]), jnp.array([-1])))
        result, valid = stream(bank, x, jnp.array([[.25, .75, 0]], dtype), 4)
        assert valid
        for primary in (result.alpha, result.kappa, result.nu, result.categories[0][:, 1]):
            np.testing.assert_array_equal(primary - count, [1, 3, 0])
        np.testing.assert_array_equal(result.categories[0][:, 0], count)
        np.testing.assert_array_equal(result.categories[1], count)
        for low in jax.tree.leaves(result.carry):
            np.testing.assert_array_equal(low, 0)


def test_scale_accumulates_fractional_evidence_across_calls():
    with jax.enable_x64(False):
        bank = scalar_bank(np.float32, 0, 1, 2**24)
        record = one(np.float32, 0, .25)
        result, valid = stream(bank, record, jnp.ones((1, 1), jnp.float32), 16)
        assert valid and result.psi.item() == 2**24 + 4


def test_small_soft_assignment_does_not_reject_an_entire_record():
    # This concentration and its tiny assignment occur on an observed color patch.
    dtype = jnp.float32
    bank = scalar_bank(dtype, 0, 1, 1, size=2)
    high = jnp.array([[1.0009524], [1]], dtype)
    low = jnp.array([[-5.854682e-8], [0]], dtype)
    bank = bank._replace(categories=(high,))
    bank = bank._replace(carry=bank.carry._replace(categories=(low,)))
    x = one(dtype, 0, 0)._replace(categories=(jnp.array([0]),))
    w = jnp.array([[4.162068e-12, 1]], dtype)
    post, ok = jax.jit(update)(bank, x, w)
    assert ok
    change = ((post.categories[0][0, 0].item() - high[0, 0].item())
              + (post.carry.categories[0][0, 0].item() - low[0, 0].item()))
    low_ulp = abs(np.spacing(np.float32(low[0, 0])))
    np.testing.assert_allclose(change, w[0, 0], atol=low_ulp / 2, rtol=0)
    assert post.categories[0][1, 0] == 2


def test_exact_midpoint_accepts_either_nearest_primary():
    # A compiled covariance contraction can put a midpoint on the odd endpoint.
    dtype = np.float32
    high = dtype(-.25234035)
    low = np.spacing(high) / dtype(2)
    bank = scalar_bank(dtype, high, 1, 1)
    bank = bank._replace(carry=bank.carry._replace(mu=jnp.array([[low]], dtype)))
    record = one(dtype, high, .25)
    result, ok = jax.jit(update)(bank, record, jnp.ones((1, 1), dtype))
    assert ok and result.kappa[0] == 2
    beyond = np.nextafter(low, dtype(-np.inf))
    invalid = bank._replace(carry=bank.carry._replace(mu=jnp.array([[beyond]], dtype)))
    _, ok = jax.jit(update)(invalid, record, jnp.ones((1, 1), dtype))
    assert not ok


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_second_word_exhaustion_is_an_atomic_failure(dtype):
    with jax.enable_x64(dtype == np.float64):
        bits = np.finfo(dtype).nmant + 1
        bank = scalar_bank(dtype, 0, 1, 1, count=2.**(3*bits))
        # A unit observation is smaller than one ULP even of this legal low word.
        bank = bank._replace(carry=bank.carry._replace(alpha=jnp.array([2.**(bits+5)], dtype)))
        result, valid = jax.jit(update)(bank, one(dtype, 10, .5), jnp.ones((1, 1), dtype))
        assert not valid
        for got, original in zip(jax.tree.leaves(result), jax.tree.leaves(bank), strict=True):
            np.testing.assert_array_equal(got, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scores_retain_gaussian_and_categorical_low_word_odds(dtype):
    with jax.enable_x64(dtype == np.float64):
        eps = np.finfo(dtype).eps
        bank = scalar_bank(dtype, 0, 1, 1, count=1, size=2)
        fraction = eps/4
        carry = bank.carry._replace(psi=jnp.array([[[fraction]], [[-fraction]]], dtype))
        remote = 2.**(np.finfo(dtype).nmant // 2 + 1)
        scores, weights, valid = jax.jit(score)(bank._replace(carry=carry), one(dtype, remote, 0))
        assert valid
        odds = remote**2 * fraction / (1 - fraction**2) - np.arctanh(fraction)
        np.testing.assert_allclose(weights[0, 0], expit(odds), rtol=4*eps)
        assert scores.dtype == dtype

        # With category concentrations (a,1), observing zero scores exactly -1/a.
        a = dtype(eps/2)
        counts = jnp.array([[a, 1], [a, 1]], dtype)
        bank = bank._replace(categories=(counts,))
        low = jnp.array([[a*fraction, 0], [-a*fraction, 0]], dtype)
        bank = bank._replace(carry=bank.carry._replace(categories=(low,)))
        missing = one(dtype, np.nan, np.nan)._replace(gaussian_observed=jnp.array([False]),
                                                     categories=(jnp.array([0]),))
        _, weights, valid = jax.jit(score)(bank, missing)
        assert valid
        np.testing.assert_allclose(weights[0, 0], expit(2*fraction/(a*(1-fraction**2))), rtol=8*eps)
