"""Hostile inputs and transformed execution for the local D12 kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import hybrid_natural as hn
from cffg import hybrid_separator as hs
from hybrid_kernel_cases import build, case, messages, stack, tolerance, unstack


def site(dtype=np.float64, card=2, dim=1):
    return hn.Natural(jnp.zeros(card, dtype), jnp.zeros((card, dim), dtype),
                      jnp.broadcast_to(jnp.eye(dim, dtype=dtype), (card, dim, dim)), jnp.int32(0))


@pytest.mark.parametrize("dtype,shift", ((np.float32, 1e8), (np.float64, 1e16)))
@pytest.mark.parametrize("precisions", ((1, 1), (1, 4)))
def test_large_common_gauge_preserves_normalized_mass_and_gaussian_scales(dtype, shift, precisions):
    value = site(dtype)._replace(precision=jnp.asarray(precisions, dtype)[:, None, None])
    reference = hn.normalize(value)
    shifted = value._replace(constant=jnp.full(2, shift, dtype))
    expected = np.reciprocal(np.sqrt(np.asarray(precisions, dtype)))
    expected /= expected.sum()
    budget = 32 * np.finfo(dtype).eps
    for normalize in (hn.normalize, jax.jit(hn.normalize)):
        result = normalize(shifted)
        hn.raise_for_status(result.status)
        np.testing.assert_allclose(result.probability, expected, atol=budget, rtol=budget)
        np.testing.assert_allclose(result.log_probability, reference.log_probability, atol=budget, rtol=budget)
        assert result.probability.sum() == pytest.approx(1, abs=budget)
        assert result.entropy() == pytest.approx(float(reference.entropy()), abs=budget)
        assert result.log_normalizer == dtype(shift + float(reference.log_normalizer))

        def log_normalizer(constant):
            return normalize(shifted._replace(constant=constant)).log_normalizer

        derivative = jax.grad(log_normalizer)(shifted.constant)
        np.testing.assert_allclose(derivative, expected, atol=budget, rtol=budget)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("kind,flag", (("empty", hn.Status.EMPTY_SUPPORT), ("singular", hn.Status.IMPROPER),
                                      ("negative", hn.Status.IMPROPER), ("nan", hn.Status.NONFINITE),
                                      ("infinite", hn.Status.NONFINITE), ("asymmetric", hn.Status.INVALID),
                                      ("underflow", hn.Status.UNDERFLOW)))
def test_invalid_beliefs_fail_identically_on_host_and_jit(dtype, kind, flag):
    value = site(dtype, dim=2)
    if kind == "empty":
        value = value._replace(constant=jnp.full(2, -jnp.inf, dtype))
    elif kind in ("singular", "negative"):
        value = value._replace(precision=value.precision.at[0, 0, 0].set(0 if kind == "singular" else -1))
    elif kind in ("nan", "infinite"):
        value = value._replace(linear=value.linear.at[0, 0].set(jnp.nan if kind == "nan" else jnp.inf))
    elif kind == "asymmetric":
        value = value._replace(precision=value.precision.at[0, 0, 1].set(0.1))
    else:
        value = value._replace(constant=value.constant.at[1].set(2 * np.log(np.finfo(dtype).tiny)))
    for normalize in (hn.normalize, jax.jit(hn.normalize)):
        result = normalize(value)
        assert int(result.status) & flag
        assert not np.isfinite(result.log_normalizer)
        with pytest.raises(ValueError, match=flag.name.lower()):
            hn.raise_for_status(result.status)
        if kind == "underflow":
            assert np.isfinite(result.log_probability[1])


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_spectral_resolution_scales_with_dtype_and_units(dtype):
    dim = 2
    threshold = 32 * dim * np.finfo(dtype).eps
    for scale in (1e-6, 1e6):
        value = site(dtype, dim=dim)
        matrix = jnp.diag(jnp.asarray([threshold / 2, 1], dtype)) * scale
        bad = value._replace(precision=jnp.broadcast_to(matrix, value.precision.shape))
        assert hn.normalize(bad).status & hn.Status.IMPROPER
        matrix = matrix.at[0, 0].set(2 * threshold * scale)
        good = value._replace(precision=jnp.broadcast_to(matrix, value.precision.shape))
        hn.raise_for_status(hn.normalize(good).status)


def test_literal_zeros_and_changed_support_require_fresh_message_inventory():
    cavity = site()._replace(constant=jnp.array([0.0, -jnp.inf]))
    belief = hn.normalize(cavity)
    np.testing.assert_array_equal(belief.probability, [1, 0])
    np.testing.assert_array_equal(belief.mean[1], 0)
    np.testing.assert_array_equal(belief.covariance[1], 0)
    quotient = hn.quotient(belief, cavity)
    assert np.isneginf(quotient.constant[1])
    np.testing.assert_array_equal(quotient.precision[1], 0)
    revived = hn.normalize(site())
    invalid = jax.jit(hn.quotient)(revived, cavity)
    assert invalid.status & hn.Status.SUPPORT_CHANGED
    _, status = hs.natural_residual(quotient, hn.quotient(revived, site()))
    assert status & hn.Status.SUPPORT_CHANGED


def test_signed_and_vacuous_cavities_can_have_proper_tilts_without_clamping():
    factors, sites = map(unstack, build(case(2)))
    for precision in (-0.01, 0.0):
        cavity = sites[1]._replace(precision=jnp.full((2, 1, 1), precision))
        assert hn.normalize(cavity).status & hn.Status.IMPROPER
        region = hn.normalize(factors[0] + hn.lift(sites[0], cavity))
        hn.raise_for_status(region.status)
        assert np.isfinite(region.entropy())
    belief = hn.from_moments(jnp.array([0.4, 0.6]), jnp.array([[2.0], [-1.0]]), jnp.array([[[0.5]], [[3.0]]]))
    cavity = site()._replace(precision=jnp.full((2, 1, 1), 4.0))
    output = hn.quotient(belief, cavity)
    assert np.all(output.precision < 0)
    reconstructed = hn.normalize(cavity + output)
    np.testing.assert_allclose(reconstructed.probability, belief.probability)
    np.testing.assert_allclose(reconstructed.mean, belief.mean)
    np.testing.assert_allclose(reconstructed.covariance, belief.covariance)


def test_inactive_auxiliaries_and_learning_are_zero_without_exposure_division():
    factors, sites = map(unstack, build(case(2)))
    sites[0] = sites[0]._replace(constant=jnp.array([0.0, -jnp.inf]))
    region = hn.normalize(factors[0] + hn.lift(*sites))
    weights, offsets = jnp.ones((2, 1, 1)), jnp.zeros((2, 1))
    aux = jax.jit(hs.auxiliary_statistics)(region, weights, offsets, jnp.zeros((2, 1)))
    hn.raise_for_status(aux.status)
    for value in (aux.target, aux.residual, aux.exposure, aux.linear, aux.precision):
        np.testing.assert_array_equal(value[1], 0)
    stale = hs.auxiliary_statistics(region, weights, offsets, jnp.ones((2, 1)))
    assert stale.residual[1, 0] == 1
    assert stale.target[1, 0] == 0


def test_nonfinite_objective_and_residual_never_report_success():
    factors, sites = build(case(3))
    outgoing = messages(sites)
    outgoing = (outgoing[0]._replace(linear=outgoing[0].linear.at[0, 0, 0].set(jnp.nan)), outgoing[1])
    result = jax.jit(hs.evaluate)(factors, sites, outgoing)
    assert result.status & hn.Status.NONFINITE
    for value in (result.free_energy, result.dual_energy, result.moment_residual, result.quotient_residual):
        assert not np.isfinite(value)
    good = hn.normalize(site())
    value, status = hs.moment_residual(good, good._replace(mean=good.mean.at[0, 0].set(jnp.nan)))
    assert status & hn.Status.NONFINITE
    assert not np.isfinite(value)


def test_tiny_active_label_cannot_hide_conditional_covariance_error():
    first = hn.from_moments(jnp.array([1 - 1e-20, 1e-20]), jnp.zeros((2, 1)), jnp.ones((2, 1, 1)))
    second = first._replace(covariance=first.covariance.at[1, 0, 0].set(2))
    value, status = hs.moment_residual(first, second)
    hn.raise_for_status(status)
    assert value == 0.5


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_nonfinite_second_moment_is_rejected_even_when_mean_is_finite(dtype):
    mean = jnp.full((2, 1), 2 * np.sqrt(np.finfo(dtype).max), dtype)
    result = hn.from_moments(jnp.full(2, 0.5, dtype), mean, jnp.ones((2, 1, 1), dtype))
    assert result.status & hn.Status.NONFINITE


def test_adapter_dimensions_cannot_broadcast_to_a_different_gaussian_port():
    with pytest.raises(ValueError, match="dimensions"):
        hn.gaussian_adapter(site(dim=2), hn.Natural(jnp.array(0.0), jnp.zeros(1), jnp.ones((1, 1))))


@pytest.mark.parametrize("failure", ("shape", "dtype", "channel", "noise", "negative_xi", "aux_shape"))
def test_invalid_transition_coordinates_have_explicit_contract(failure):
    weights, offsets, xi = jnp.zeros((2, 1, 1)), jnp.zeros((2, 1)), jnp.ones((2, 1))
    matrix, bias, noise = jnp.ones((2, 1, 1)), jnp.zeros((2, 1)), jnp.ones((2, 1, 1))
    if failure == "shape":
        with pytest.raises(ValueError, match="shapes"):
            hs.transition(weights, offsets, xi, matrix[0], bias, noise)
    elif failure == "dtype":
        with pytest.raises(ValueError, match="dtypes"):
            hs.transition(weights.astype(jnp.float32), offsets, xi, matrix, bias, noise)
    elif failure == "channel":
        for channel in ("source", "pair", "nonlinear"):
            with pytest.raises(ValueError, match="destination-selected"):
                hs.transition(weights, offsets, xi, matrix, bias, noise, channel=channel)
    elif failure == "noise":
        result = jax.jit(hs.transition)(weights, offsets, xi, matrix, bias, -noise)
        assert result.status & hn.Status.IMPROPER
    elif failure == "negative_xi":
        result = hs.transition(weights, offsets, -xi, matrix, bias, noise)
        assert result.status & hn.Status.INVALID
    else:
        with pytest.raises(ValueError, match="shapes"):
            hs.transition(weights, offsets, xi[:, 0], matrix, bias, noise)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_single_vmap_jit_agree_for_all_local_outputs(dtype):
    factors, sites = build(case(3, dtype))
    left = jax.tree.map(lambda x: x[:-1], sites)
    right = jax.tree.map(lambda x: x[1:], sites)

    def local(factor, l, r):
        region = hn.normalize(factor + hn.lift(l, r))
        aux = hs.auxiliary_statistics(region, jnp.full((2, 1, 1), 0.3, dtype),
                                      jnp.zeros((2, 1), dtype), jnp.ones((2, 1), dtype))
        return region, hn.quotient(hn.marginal(region, 0), l), hn.quotient(hn.marginal(region, 1), r), aux

    individual = stack([local(f, l, r) for f, l, r in zip(unstack(factors), unstack(left), unstack(right))])
    for result in (jax.vmap(local)(factors, left, right), jax.jit(jax.vmap(local))(factors, left, right)):
        for actual, expected in zip(jax.tree.leaves(result), jax.tree.leaves(individual)):
            np.testing.assert_allclose(actual, expected, atol=tolerance(dtype), rtol=tolerance(dtype))
    eager = hs.evaluate(factors, sites, messages(sites))
    compiled = jax.jit(hs.evaluate)(factors, sites, messages(sites))
    for actual, expected in zip(jax.tree.leaves(compiled), jax.tree.leaves(eager)):
        np.testing.assert_allclose(actual, expected, atol=tolerance(dtype), rtol=tolerance(dtype))


@pytest.mark.parametrize("card,dim", ((20, 1), (2, 10)))
def test_float32_polynomial_local_inventory(card, dim):
    dtype = jnp.float32
    weights, offsets = jnp.full((card, card - 1, dim), 0.02, dtype), jnp.zeros((card, card - 1), dtype)
    noise = jnp.broadcast_to(jnp.eye(dim, dtype=dtype), (card, dim, dim))
    factor = jax.jit(hs.transition)(weights, offsets, jnp.ones_like(offsets), noise / 2,
                                  jnp.zeros((card, dim), dtype), noise)
    cavity = site(dtype, card, dim)
    region = jax.jit(hn.normalize)(factor + hn.lift(cavity, cavity))
    aux = jax.jit(hs.auxiliary_statistics)(region, weights, offsets, jnp.ones_like(offsets))
    hn.raise_for_status(region.status | aux.status)
    assert region.covariance.shape == (card, card, 2 * dim, 2 * dim)
    assert aux.precision.shape == (card, card - 1, dim + 1, dim + 1)
    assert np.isfinite(region.entropy())
    assert sum(x.size for x in jax.tree.leaves((factor, region, aux))) < 50 * card * card * (dim + 1) ** 2


def test_hundred_slice_float32_evaluation_stores_only_local_covariances():
    factors, sites = build(case(100, np.float32))
    result = jax.jit(hs.evaluate)(factors, sites, messages(sites))
    hn.raise_for_status(result.status)
    assert result.regions.covariance.shape == (99, 2, 2, 2, 2)
    assert result.separators.covariance.shape == (100, 2, 1, 1)
    assert np.isfinite(result.free_energy)
    total_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(result))
    assert total_bytes < 100 * 2 ** 2 * 2 ** 2 * 4 * 6
