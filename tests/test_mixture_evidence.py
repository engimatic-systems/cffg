"""Independent Gaussian-overlap oracles for relative mixture evidence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import softmax

from cffg import mixg
from cffg.compiled import NEG
from cffg.mixture_evidence import linear_evidence, predictive_evidence, prior_evidence


def overlap(means, covariances, observation, noise, projection=None):
    projection = np.eye(len(observation)) if projection is None else projection
    scores = []
    for mean, covariance in zip(means, covariances):
        innovation = noise + projection @ covariance @ projection.T
        residual = observation - projection @ mean
        scores.append(-.5 * (residual @ np.linalg.solve(innovation, residual)
                              + np.linalg.slogdet(innovation)[1]))
    return np.asarray(scores)


def assert_scores(actual, expected, dtype, *, scale=64):
    assert bool(actual.valid)
    np.testing.assert_allclose(softmax(actual.score), softmax(expected),
                               rtol=scale * np.finfo(dtype).eps,
                               atol=scale * np.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("linear", [False, True])
def test_displaced_components_do_not_lose_selector_evidence(dtype, linear):
    convert = lambda value: jnp.asarray(value, dtype=dtype)
    mean = convert([[10000.], [10000.125], [9999.75]])
    variance = convert([[[.5]], [[1.]], [[2.]]])
    target, noise = convert([10001.]), convert([[.25]])
    eo, lo, w = target / noise[0, 0], 1 / noise, 1 / variance
    if linear:
        a, b = convert([[[1.]]] * 3), mean - 10000
        ei, li = convert([1e8]), convert([[1e4]])
        args = tuple(value[None] for value in (eo, lo, ei, li, a, b, w))
        message = jax.jit(mixg.mixlin_to_selector)(*args)[0]
        evidence = linear_evidence(eo, lo, ei, li, a, b, w)
        variance = variance + 1e-4
        stats = mixg.mixlin_stats(*args[:4], convert([[0., 0., NEG]]), *args[4:])
    else:
        args = tuple(value[None] for value in (eo, lo, mean, w))
        message = jax.jit(mixg.mix_to_selector)(*args)[0]
        evidence = prior_evidence(eo, lo, mean, w)
        stats = mixg.mix_stats(*args[:2], convert([[0., 0., NEG]]), *args[2:])
    reference = overlap(np.asarray(mean, float), np.asarray(variance, float),
                        np.asarray(target, float), np.asarray(noise, float))
    assert_scores(evidence, reference, dtype)
    np.testing.assert_allclose(softmax(message), softmax(reference), atol=64 * np.finfo(dtype).eps)
    assert stats.proper[0]
    expected = softmax(np.r_[reference[:2], -np.inf])
    np.testing.assert_allclose(stats.responsibility[0], expected, atol=64 * np.finfo(dtype).eps)
    assert stats.responsibility[0, 2] == 0.


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_partial_observation_at_large_dimension_matches_overlap(dtype):
    rng = np.random.default_rng(418)
    components, dimension = 32, 20
    means = rng.normal(size=(components, dimension)).astype(dtype)
    covariances = np.broadcast_to(np.diag(np.linspace(.5, 2, dimension)),
                                  (components, dimension, dimension)).astype(dtype)
    projection = np.eye(dimension, dtype=dtype)[:3]
    observation = np.array([.4, -.7, .1], dtype=dtype)
    noise = np.diag([.8, 1.2, 1.5]).astype(dtype)
    precision = projection.T @ np.linalg.solve(noise, projection)
    eta = projection.T @ np.linalg.solve(noise, observation)
    args = tuple(jnp.asarray(value) for value in (eta, precision, means, covariances))
    result = jax.jit(predictive_evidence)(*args)
    assert_scores(result, overlap(means, covariances, observation, noise, projection), dtype)
    batched = jax.jit(jax.vmap(predictive_evidence))(*(jnp.stack([value, value]) for value in args))
    np.testing.assert_array_equal(batched.score[0], batched.score[1])
    assert batched.valid.all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("precision", [0., 1e-12, -.25])
def test_weak_vacuous_and_integrable_negative_potentials(dtype, precision):
    eta = jnp.array([1.], dtype=dtype)
    lo = jnp.array([[precision]], dtype=dtype)
    means = jnp.array([[0.], [1.]], dtype=dtype)
    covariance = jnp.ones((2, 1, 1), dtype=dtype)
    evidence = predictive_evidence(eta, lo, means, covariance)
    # Scalar integration after translating by each predictive mean.
    m = np.asarray(means[:, 0], float)
    expected = m - .5 * precision * m**2 + .5 * (1 - precision * m)**2 / (1 + precision)
    assert_scores(evidence, expected, dtype)
    uniform = predictive_evidence(eta * 0, lo * 0, means, covariance)
    assert_scores(uniform, np.zeros(2), dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_singular_potential_keeps_linear_term_in_nullspace(dtype):
    means = jnp.array([[0., 0.], [.5, 1.]], dtype=dtype)
    covariance = jnp.broadcast_to(jnp.eye(2, dtype=dtype), (2, 2, 2))
    result = predictive_evidence(jnp.array([.2, 1.], dtype=dtype),
                                 jnp.diag(jnp.array([1., 0.], dtype=dtype)), means, covariance)
    m = np.asarray(means, float)
    expected = .2 * m[:, 0] - .5 * m[:, 0]**2 + (.2 - m[:, 0])**2 / 4 + m[:, 1] + .5
    assert_scores(result, expected, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_improper_input_retains_integrable_joint_jacobian(dtype):
    convert = lambda value: jnp.asarray(value, dtype=dtype)
    args = tuple(map(convert, ([0.], [[1.]], [0.], [[0.]], [[[1.]], [[2.]]], [[0.], [0.]], [[[1.]], [[1.]]])))
    result = jax.jit(linear_evidence)(*args)
    assert_scores(result, np.log([1., .5]), dtype)
    message = mixg.mixlin_to_selector(*(value[None] for value in args))[0]
    np.testing.assert_allclose(softmax(message), [2 / 3, 1 / 3], atol=64 * np.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ["singular", "indefinite", "nonsymmetric", "overflow", "nonintegrable"])
def test_invalid_intermediates_never_become_finite_accepted_evidence(dtype, case):
    means = jnp.array([[0., 0.], [1., 1.]], dtype=dtype)
    covariances = jnp.broadcast_to(jnp.eye(2, dtype=dtype), (2, 2, 2))
    precision = jnp.eye(2, dtype=dtype)
    eta = jnp.ones(2, dtype=dtype)
    if case == "singular":
        covariances = covariances.at[1, 1, 1].set(0.)
    elif case == "indefinite":
        covariances = covariances.at[1, 1, 1].set(-1.)
    elif case == "nonsymmetric":
        precision = precision.at[0, 1].set(.5)
    elif case == "overflow":
        eta = eta * np.finfo(dtype).max
    else:
        precision = -precision
    result = jax.jit(predictive_evidence)(eta, precision, means, covariances)
    assert not result.valid


@pytest.mark.parametrize("linear", [False, True])
def test_normalized_evidence_derivatives_match_independent_overlap(linear):
    means = jnp.array([[.3, -.5], [1.2, .7]])
    covariance = jnp.broadcast_to(jnp.eye(2) * .6, (2, 2, 2))
    precision = jnp.eye(2) * 2
    eta = jnp.array([.7, -.3])

    def weights(value):
        if linear:
            evidence = linear_evidence(eta, precision, jnp.zeros(2), jnp.eye(2) * 5,
                                       jnp.broadcast_to(jnp.eye(2), (2, 2, 2)), value,
                                       jnp.broadcast_to(jnp.eye(2) * 2.5, (2, 2, 2)))
        else:
            evidence = predictive_evidence(eta, precision, value, covariance)
        return jax.nn.softmax(evidence.score)

    def reference(value):
        residual = value - eta / 2
        return jax.nn.softmax(-.5 * jnp.sum(residual**2, axis=-1) / 1.1)

    expected = jax.jacfwd(reference)(means)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(weights))(means), expected, atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.jacrev(weights))(means), expected, atol=1e-12)


@pytest.mark.parametrize("output_precision", [-.2, 0., 1.])
def test_cavity_and_covariance_derivatives_match_direct_integral(output_precision):
    means = jnp.array([[.3, -.5], [1.2, .7]])
    parameters = jnp.array([.7, -.3, output_precision, 0., 0., .2])

    def inputs(value):
        covariance = jnp.diag(jnp.exp(value[4:]))
        return value[:2], jnp.diag(value[2:4]), jnp.stack([covariance, covariance * .7])

    def weights(value):
        eta, precision, covariance = inputs(value)
        return jax.nn.softmax(predictive_evidence(eta, precision, means, covariance).score)

    def reference(value):
        eta, precision, covariance = inputs(value)
        w = jnp.linalg.inv(covariance)
        weighted_mean = jnp.einsum("kij,kj->ki", w, means)
        q_eta, q_precision = weighted_mean + eta, w + precision
        q_mean = jnp.linalg.solve(q_precision, q_eta[..., None])[..., 0]
        scores = (.5 * jnp.sum(q_eta * q_mean - means * weighted_mean, axis=-1)
                  - .5 * jnp.linalg.slogdet(covariance)[1]
                  - .5 * jnp.linalg.slogdet(q_precision)[1])
        return jax.nn.softmax(scores)

    expected = jax.jacfwd(reference)(parameters)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(weights))(parameters), expected, atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.jacrev(weights))(parameters), expected, atol=1e-12)


@pytest.mark.parametrize("linear", [False, True])
def test_node_energy_and_statistics_use_the_same_sparse_responsibilities(linear):
    means = jnp.array([[.3], [1.2], [-.4]])
    w = jnp.array([[[2.]], [[1.]], [[3.]]])
    eo, lo, le = jnp.array([.7]), jnp.array([[2.]]), jnp.array([-.2, -.8, NEG])
    if linear:
        a, ei, li = jnp.ones((3, 1, 1)), jnp.array([.2]), jnp.array([[4.]])
        args = tuple(value[None] for value in (eo, lo, ei, li, le, a, means, w))
        stats, energy = mixg.mixlin_stats(*args), mixg.mixlin_energy(*args)[0]
        individual = [mixg.mixlin_energy(*args[:4], jnp.zeros((1, 1)),
                                         *(value[:, k:k+1] for value in args[5:]))[0] for k in range(3)]
    else:
        args = tuple(value[None] for value in (eo, lo, le, means, w))
        stats, energy = mixg.mix_stats(*args), mixg.mix_energy(*args)[0]
        individual = [mixg.mix_energy(*args[:2], jnp.zeros((1, 1)),
                                      *(value[:, k:k+1] for value in args[3:]))[0] for k in range(3)]
    rho = np.asarray(stats.responsibility[0])
    assert stats.proper[0] and rho[2] == 0.
    expected = rho @ individual + rho[:2] @ np.log(rho[:2])
    np.testing.assert_allclose(energy, expected, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_representable_inputs_with_overflowing_whitened_system_are_rejected(dtype):
    scale = dtype(1e20 if dtype is np.float32 else 1e200)
    eta = jnp.zeros(1, dtype=dtype)
    precision = jnp.array([[scale]], dtype=dtype)
    means = jnp.array([[0.], [np.sqrt(scale)]], dtype=dtype)
    covariances = jnp.ones((2, 1, 1), dtype=dtype) * scale
    result = predictive_evidence(eta, precision, means, covariances)
    assert not result.valid
    message = mixg.mix_to_selector(eta[None], precision[None], means[None], (1 / covariances)[None])
    assert np.isnan(message).all()


@pytest.mark.parametrize("linear", [False, True])
def test_empty_selector_support_rejects_statistics_and_energy(linear):
    eta, precision = jnp.zeros((1, 1)), jnp.ones((1, 1, 1))
    means, w = jnp.zeros((1, 2, 1)), jnp.ones((1, 2, 1, 1))
    le = jnp.full((1, 2), NEG)
    if linear:
        args = (eta, precision, eta, precision, le, w, means, w)
        stats, energy = mixg.mixlin_stats(*args), mixg.mixlin_energy(*args)
    else:
        args = (eta, precision, le, means, w)
        stats, energy = mixg.mix_stats(*args), mixg.mix_energy(*args)
    assert not stats.proper[0]
    assert np.isnan(stats.responsibility).all()
    assert np.isposinf(energy).all()
