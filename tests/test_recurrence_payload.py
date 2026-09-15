"""Held R(i,j,x) oracles for D12.8 statistics and stable JJ expectation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import hybrid_natural as hn, hybrid_separator as hs
from hybrid_kernel_cases import stack


STATISTICS = jax.jit(hs.recurrence_auxiliary_statistics)
EXPECTATION = jax.jit(hs.recurrence_expectation)


def held(dtype=np.float64, card=4, dim=2):
    rng = np.random.default_rng(921)
    probability = np.arange(1, card * card + 1, dtype=dtype).reshape(card, card)
    probability[1] = probability[:, 1] = probability[:, -1] = 0
    probability[2 % card, 1:] = 0
    probability /= probability.sum()
    displacement = rng.normal(size=(card, card, dim)).astype(dtype) / 4
    center = rng.normal(size=(card, card, dim)).astype(dtype) * 2
    root = rng.normal(size=(card, card, dim, dim)).astype(dtype) / (2 * dim)
    covariance = root @ root.swapaxes(-1, -2) + np.eye(dim, dtype=dtype)
    correction = covariance * (np.finfo(dtype).eps * .75)
    region = hn.from_centered_moments(*(jnp.asarray(x) for x in
        (probability, displacement, covariance, center)), covariance_correction=jnp.asarray(correction))
    hn.raise_for_status(region.status)
    weights = rng.normal(size=(card, card - 1, dim)).astype(dtype) / dim
    offsets = rng.normal(size=(card, card - 1)).astype(dtype) / 3
    xi = np.linspace(0, 2, card * (card - 1), dtype=dtype).reshape(card, card - 1)
    return region, *(jnp.asarray(x) for x in (weights, offsets, xi))


def reference(region, weights, offsets, xi):
    """Enumerate each supported pair/stick in extended precision, without scans.

    Evaluate the affine logit before its square and add within-pair variance.
    This independent held-belief calculation uses neither expanded potentials
    nor the candidate's conditional suffix-mixture moments.
    """
    p, r, covariance, center = (np.asarray(x, np.longdouble) for x in
        (region.probability, region.displacement, region.full_covariance, region.center))
    w, o, xi = (np.asarray(x, np.longdouble) for x in (weights, offsets, xi))
    card, _, dim = w.shape
    exposure, target = np.zeros_like(xi), np.zeros_like(xi)
    linear = np.zeros((*xi.shape, dim + 1), np.longdouble)
    precision = np.zeros((*xi.shape, dim + 1, dim + 1), np.longdouble)
    energy = np.longdouble(0)
    for i in range(card):
        for stick in range(card - 1):
            c = np.sqrt(xi[i, stick])
            omega = np.longdouble(.25) if c == 0 else np.tanh(c / 2) / (2 * c)
            base = -np.log(2 * np.cosh(c / 2)) + omega * xi[i, stick] / 2
            for j in range(stick, card):
                if p[i, j] == 0:
                    continue
                feature = np.r_[r[i, j] + center[i, j] - center[i, 0], np.longdouble(1)]
                second = np.outer(feature, feature)
                second[:dim, :dim] += covariance[i, j]
                logit = w[i, stick] @ (r[i, j] + center[i, j]) + o[i, stick]
                square = logit ** 2 + w[i, stick] @ covariance[i, j] @ w[i, stick]
                sign = np.longdouble(.5 if j == stick else -.5)
                exposure[i, stick] += p[i, j]
                target[i, stick] += p[i, j] * square
                linear[i, stick] += p[i, j] * sign * feature
                precision[i, stick] += p[i, j] * omega * second
                energy += p[i, j] * (base + sign * logit - omega * square / 2)
    target = np.divide(target, exposure, out=np.zeros_like(target), where=exposure > 0)
    return dict(target=target, residual=xi - target, exposure=exposure,
                linear=linear, precision=precision, center=center[:, 0], energy=energy)


def assert_reference(result, expected, dtype, terms=4):
    auxiliary, (value, status) = result
    hn.raise_for_status(status | auxiliary.status)
    budget = 64 * terms * np.finfo(dtype).eps
    for name in auxiliary._fields:
        if name != "status":
            np.testing.assert_allclose(getattr(auxiliary, name), expected[name], atol=budget, rtol=budget)
    np.testing.assert_array_equal(auxiliary.active, expected["exposure"] > 0)
    np.testing.assert_allclose(value, expected["energy"], atol=budget, rtol=budget)
    assert value.dtype == dtype
    assert status.dtype == jnp.int32


def payload(*args):
    return hs.recurrence_auxiliary_statistics(*args), hs.recurrence_expectation(*args)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_separate_recurrence_matches_held_pair_oracle_and_transforms(dtype):
    args = held(dtype)
    expected = reference(*args)
    eager = payload(*args)
    assert_reference(eager, expected, dtype)
    assert_reference((STATISTICS(*args), EXPECTATION(*args)), expected, dtype)
    batch = stack((args, (args[0], args[1] / 2, args[2], args[3])))
    individual = stack((eager, payload(args[0], args[1] / 2, args[2], args[3])))
    for transformed in (jax.vmap(payload), jax.jit(jax.vmap(payload))):
        result = transformed(*batch)
        for actual, wanted in zip(jax.tree.leaves(result), jax.tree.leaves(individual)):
            np.testing.assert_allclose(actual, wanted, atol=256 * np.finfo(dtype).eps, rtol=256 * np.finfo(dtype).eps)
    auxiliary = eager[0]
    inactive = ~np.asarray(auxiliary.active)
    assert inactive.any()
    for value in (auxiliary.target, auxiliary.exposure, auxiliary.linear, auxiliary.precision):
        np.testing.assert_array_equal(np.asarray(value)[inactive], 0)
    np.testing.assert_array_equal(np.asarray(auxiliary.residual)[inactive], np.asarray(args[3])[inactive])
    assert np.isneginf(np.asarray(args[0].log_probability)[1]).all()


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_held_energy_parameter_gradients_equal_learning_and_xi_equations(dtype):
    region, weights, offsets, xi = held(dtype)
    xi = jnp.zeros_like(xi)
    truth = reference(region, weights, offsets, xi)
    center = truth["center"]
    coefficient = np.concatenate((np.asarray(weights),
        (np.asarray(offsets) + np.einsum("ild,id->il", weights, center))[..., None]), axis=-1)
    gradient = truth["linear"] - np.einsum("ilab,ilb->ila", truth["precision"], coefficient)
    expected_weights = gradient[..., :-1] + center[:, None] * gradient[..., -1:]
    budget = 512 * np.finfo(dtype).eps
    derivative = jax.jit(jax.grad(lambda w, o, x: hs.recurrence_expectation(region, w, o, x)[0],
                                   argnums=(0, 1, 2)))(weights, offsets, xi)
    for actual, expected in zip(derivative, (expected_weights, gradient[..., -1],
                                -truth["exposure"] * truth["residual"] / 96)):
        np.testing.assert_allclose(actual, expected, atol=budget, rtol=budget)
    assert np.max(abs(derivative[2])) > 1e-3
    auxiliary = STATISTICS(region, weights, offsets, xi)
    physical = auxiliary.physical_linear - jnp.einsum("ilab,ilb->ila", auxiliary.physical_precision,
                                                     jnp.concatenate((weights, offsets[..., None]), axis=-1))
    np.testing.assert_allclose(physical[..., :-1], expected_weights, atol=budget, rtol=budget)


def test_held_moment_jvp_and_vjp_match_independent_difference():
    region, weights, offsets, xi = held()
    direction = jnp.cos(jnp.arange(region.displacement.size)).reshape(region.displacement.shape) / 4

    def outputs(displacement):
        moved = region._replace(displacement=displacement)
        auxiliary, (value, _) = payload(moved, weights, offsets, xi)
        return jnp.concatenate((auxiliary.target.ravel(), jnp.reshape(value, (1,))))

    _, jvp = jax.jvp(outputs, (region.displacement,), (direction,))
    step = np.finfo(np.float64).eps ** (1 / 3)
    references = []
    for sign in (-1, 1):
        moved = region._replace(displacement=region.displacement + sign * step * direction)
        result = reference(moved, weights, offsets, xi)
        references.append(np.r_[result["target"].ravel(), result["energy"]])
    np.testing.assert_allclose(jvp, (references[1] - references[0]) / (2 * step), atol=64 * step ** 2, rtol=64 * step ** 2)
    cotangent = jnp.sin(jnp.arange(jvp.size))
    _, transpose = jax.vjp(outputs, region.displacement)
    np.testing.assert_allclose(jnp.vdot(jvp, cotangent), jnp.vdot(direction, transpose(cotangent)[0]), atol=1e-12)


def grouped(region):
    """Independent unit Gaussian destination, retaining source correction words."""
    dim = region.displacement.shape[-1]
    pad = ((0, 0), (0, 0), (0, dim))
    matrices = (*pad, (0, dim))
    return region._replace(displacement=jnp.pad(region.displacement, pad), center=jnp.pad(region.center, pad),
        covariance=jnp.pad(region.covariance, matrices).at[..., dim:, dim:].set(jnp.eye(dim, dtype=region.covariance.dtype)),
        covariance_correction=jnp.pad(region.correction, matrices))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_grouped_interfaces_preserve_complete_statistics_and_original_energy(dtype):
    region, weights, offsets, xi = held(dtype)
    group = grouped(region)
    auxiliary = STATISTICS(region, weights, offsets, xi)
    params = hs.TransitionParameters(weights, offsets, xi, jnp.zeros((4, 2, 2), dtype),
                                     jnp.zeros((4, 2), dtype), jnp.broadcast_to(jnp.eye(2, dtype=dtype), (4, 2, 2)))
    for calculate in (hs.auxiliary_statistics, jax.jit(hs.auxiliary_statistics)):
        actual = calculate(group, weights, offsets, xi)
        for a, b in zip(actual, auxiliary):
            np.testing.assert_allclose(a, b, atol=8 * np.finfo(dtype).eps, rtol=8 * np.finfo(dtype).eps)
    expected = reference(region, weights, offsets, xi)["energy"] - (1 + np.log(2 * np.pi))
    for calculate in (hs.transition_expectation, jax.jit(hs.transition_expectation)):
        value, status = calculate(group, params)
        hn.raise_for_status(status)
        np.testing.assert_allclose(value, expected, atol=512 * np.finfo(dtype).eps, rtol=512 * np.finfo(dtype).eps)
    # A malformed destination correction must survive reduction to R status.
    invalid = group._replace(covariance_correction=group.correction.at[..., -1, -1].set(.01))
    assert hs.auxiliary_statistics(invalid, weights, offsets, xi).status & hn.Status.INVALID
    assert hs.transition_expectation(invalid, params)[1] == hn.Status.INVALID | hn.Status.NONFINITE


@pytest.mark.parametrize("kind,flag", (("improper", hn.Status.IMPROPER), ("empty", hn.Status.EMPTY_SUPPORT),
    ("negative_probability", hn.Status.INVALID), ("nonfinite", hn.Status.NONFINITE), ("negative_xi", hn.Status.INVALID)))
def test_failed_checked_beliefs_and_parameters_have_explicit_status(kind, flag):
    region, weights, offsets, xi = held(np.float32)
    if kind in ("improper", "empty"):
        potential = hn.Natural(jnp.full((4, 4), -jnp.inf if kind == "empty" else 0, jnp.float32),
                                jnp.zeros((4, 4, 2), jnp.float32), jnp.zeros((4, 4, 2, 2), jnp.float32))
        region = hn.normalize(potential)
    elif kind == "negative_probability":
        region = hn.from_moments(region.probability.at[0, 0].set(-1), region.mean, region.full_covariance)
    elif kind == "nonfinite":
        offsets = offsets.at[0, 0].set(jnp.inf)
    else:
        xi = xi.at[0, 0].set(-1)
    for calculate in (payload, jax.jit(payload)):
        auxiliary, (value, status) = calculate(region, weights, offsets, xi)
        assert auxiliary.status & flag and status & flag
        assert np.isnan(value)
        assert all(np.isnan(x).all() for x in auxiliary[:5])


@pytest.mark.parametrize("precision", (-.1, 0.))
def test_signed_or_vacuous_cavity_is_lawful_when_recurrence_tilt_is_proper(precision):
    dtype = jnp.float32
    weights, offsets, xi = jnp.ones((2, 1, 1), dtype), jnp.zeros((2, 1), dtype), jnp.zeros((2, 1), dtype)
    cavity = hn.Natural(jnp.zeros((2, 2), dtype), jnp.zeros((2, 2, 1), dtype), jnp.full((2, 2, 1, 1), precision, dtype))
    assert hn.normalize(cavity).status & hn.Status.IMPROPER
    region = hn.normalize(hs.recurrence_potential(weights, offsets, xi) + cavity)
    assert_reference((STATISTICS(region, weights, offsets, xi), EXPECTATION(region, weights, offsets, xi)),
                     reference(region, weights, offsets, xi), np.float32)


@pytest.mark.parametrize("card,dim", ((20, 1), (2, 10)))
def test_native_float32_independent_label_and_state_scale(card, dim):
    with jax.enable_x64(False):
        args = held(np.float32, card, dim)
        result = STATISTICS(*args), EXPECTATION(*args)
        assert_reference(result, reference(*args), np.float32, terms=max(card, dim))
        assert all(x.dtype in (jnp.float32, jnp.int32) for x in jax.tree.leaves(result))
        assert result[0].precision.shape == (card, card - 1, dim + 1, dim + 1)
        assert sum(x.size for x in jax.tree.leaves((args[0], result))) < 20 * card * card * (dim + 1) ** 2


@pytest.mark.parametrize("dtype,origin,variance", ((np.float32, 10000, 1.), (np.float64, 1e8, 1.)))
def test_original_parameters_retain_canceled_logit_variance_and_xi_derivative(dtype, origin, variance):
    region = hn.from_centered_moments(jnp.full((2, 2), .25, dtype), jnp.zeros((2, 2, 1), dtype),
        jnp.full((2, 2, 1, 1), variance, dtype), jnp.full((2, 2, 1), origin, dtype))
    weights, offsets, xi = jnp.ones((2, 1, 1), dtype), jnp.full((2, 1), -origin, dtype), jnp.zeros((2, 1), dtype)
    result = STATISTICS(region, weights, offsets, xi), EXPECTATION(region, weights, offsets, xi)
    assert_reference(result, reference(region, weights, offsets, xi), dtype)
    np.testing.assert_allclose(result[0].target, variance, atol=0, rtol=64 * np.finfo(dtype).eps)
    derivative = jax.grad(lambda x: -hs.recurrence_expectation(region, weights, offsets, x)[0])(xi)
    np.testing.assert_allclose(derivative, -variance / 192, atol=0, rtol=64 * np.finfo(dtype).eps)


def test_recurrence_statistics_retain_covariance_low_word_and_require_actual_scope():
    dtype = jnp.float32
    probability = jnp.array([[1, 0], [0, 0]], dtype)
    covariance = jnp.ones((2, 2, 1, 1), dtype)
    mean = jnp.zeros((2, 2, 1), dtype)
    low = covariance * (.75 * jnp.finfo(dtype).eps)
    region = hn.from_centered_moments(probability, mean, covariance, mean, covariance_correction=low)
    weights, offsets, xi = jnp.ones((2, 1, 1), dtype), jnp.zeros((2, 1), dtype), jnp.zeros((2, 1), dtype)
    result = STATISTICS(region, weights, offsets, xi)
    uncorrected = STATISTICS(region._replace(covariance_correction=None), weights, offsets, xi)
    assert result.target[0, 0] > uncorrected.target[0, 0]
    assert result.target[0, 0] == region.full_covariance[0, 0, 0, 0]
    assert result.precision[0, 0, 0, 0] == region.full_covariance[0, 0, 0, 0] / 4
    with pytest.raises(ValueError, match="belief shapes or dtypes"):
        hs.recurrence_auxiliary_statistics(grouped(region), weights, offsets, xi)
