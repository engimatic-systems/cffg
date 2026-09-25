"""Producer-owned rank certificates, without a global precision floor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import gauss, mixg
from cffg.gaussian_components import backward_component, forward_component


def message(direction, eta, precision, matrix, noise, offset=None, components=1):
    output = matrix.shape[0]
    if offset is None:
        offset = jnp.zeros(output, dtype=matrix.dtype)
    rule = mixg.mixlin_forward if direction == "forward" else mixg.mixlin_backward
    return rule(
        eta[None], precision[None], jnp.zeros((1, components), matrix.dtype),
        jnp.broadcast_to(matrix, (1, components, *matrix.shape)),
        jnp.broadcast_to(offset, (1, components, output)),
        jnp.broadcast_to(noise, (1, components, output, output)),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_forward_rotated_nullspace_is_not_reclassified_after_reconstruction(dtype):
    matrix = jnp.array([[3.], [4.]], dtype)
    noise = jnp.eye(2, dtype=dtype)
    component = forward_component(jnp.zeros(1, dtype), jnp.zeros((1, 1), dtype),
                                  matrix, jnp.zeros(2, dtype), noise)
    assert bool(component.valid)
    assert not bool(component.normalizable)
    np.testing.assert_allclose(component.precision, np.array([[16., -12.], [-12., 9.]]) / 25,
                               rtol=32 * np.finfo(dtype).eps, atol=32 * np.finfo(dtype).eps)
    for evaluate in (message, jax.jit(message, static_argnames=("direction", "components"))):
        result = evaluate("forward", jnp.zeros(1, dtype), jnp.zeros((1, 1), dtype), matrix, noise)
        for value in result:
            np.testing.assert_array_equal(value, 0.)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_backward_proportional_columns_remain_vacuous(dtype):
    matrix = jnp.array([[1., 2.], [2., 4.]], dtype)
    identity = jnp.eye(2, dtype=dtype)
    component = backward_component(jnp.zeros(2, dtype), identity, matrix,
                                   jnp.zeros(2, dtype), identity)
    assert bool(component.valid)
    assert not bool(component.normalizable)
    result = jax.jit(message, static_argnames=("direction", "components"))(
        "backward", jnp.zeros(2, dtype), identity, matrix, identity,
    )
    for value in result:
        np.testing.assert_array_equal(value, 0.)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_singular_output_information_can_still_inform_every_input_direction(dtype):
    matrix = jnp.array([[1.], [0.]], dtype)
    precision = jnp.diag(jnp.array([1., 0.], dtype))
    result = message("backward", jnp.array([2., 0.], dtype), precision, matrix,
                     jnp.eye(2, dtype=dtype))
    np.testing.assert_allclose(result[0], [[1.]], rtol=32 * np.finfo(dtype).eps)
    np.testing.assert_allclose(result[1], [[[.5]]], rtol=32 * np.finfo(dtype).eps)


@pytest.mark.parametrize("dtype,vector", [(jnp.float32, [2., 3.]), (jnp.float64, [3., 4.])])
def test_equilibration_does_not_amplify_an_annihilated_input_direction(dtype, vector):
    matrix = jnp.asarray(vector, dtype)[:, None]
    perpendicular = jnp.array([vector[1], -vector[0]], dtype)
    precision = jnp.outer(perpendicular, perpendicular) / (perpendicular @ perpendicular)
    identity = jnp.eye(2, dtype=dtype)
    component = backward_component(jnp.zeros(2, dtype), precision, matrix,
                                   jnp.zeros(2, dtype), identity)
    assert bool(component.valid)
    assert not bool(component.normalizable)
    result = message("backward", jnp.zeros(2, dtype), precision, matrix, identity)
    for value in result:
        np.testing.assert_array_equal(value, 0.)


@pytest.mark.parametrize("dtype,noise", [
    (jnp.float32, [[1.4636034966, -.7008606791], [-.7008606791, 2.9060730934]]),
    (jnp.float64, [[1.6869178030638454, 1.8398016399284158],
                   [1.8398016399284158, 6.544232290923466]]),
])
def test_backward_rank_uses_incoming_subspace_not_subtractive_eigenvector_noise(dtype, noise):
    matrix = jnp.array([[0.], [1.]], dtype)
    precision = jnp.diag(jnp.array([1., 0.], dtype))
    noise = jnp.asarray(noise, dtype)
    component = jax.jit(backward_component)(jnp.zeros(2, dtype), precision, matrix,
                                          jnp.zeros(2, dtype), noise)
    assert bool(component.valid)
    assert not bool(component.normalizable)
    result = message("backward", jnp.zeros(2, dtype), precision, matrix, noise)
    for value in result:
        np.testing.assert_array_equal(value, 0.)


@pytest.mark.parametrize("dtype,information", [
    (jnp.float32, [1e-3, 1e5, 0.]),
    (jnp.float64, [1e-8, 1e8, 0.]),
])
def test_singular_incoming_subspace_preserves_independently_scaled_information(dtype, information):
    incoming = jnp.diag(jnp.asarray(information, dtype))
    matrix = jnp.eye(3, dtype=dtype)[:, :2]
    result = message("backward", jnp.zeros(3, dtype), incoming, matrix, jnp.eye(3, dtype=dtype))
    expected = np.asarray(information[:2]) / (1 + np.asarray(information[:2]))
    # The singular branch still computes a subtractive Schur complement.
    tolerance = 32 * np.finfo(dtype).eps / min(information[:2])
    np.testing.assert_allclose(result[1][0], np.diag(expected), rtol=tolerance, atol=0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_coordinate_rescaling_preserves_proper_covariance_branch(dtype, direction):
    # Twelve orders of coordinate scale are not twelve orders of uncertainty.
    scales = np.array([1e-6, 1., 1e6])
    matrix = jnp.diag(jnp.asarray(scales, dtype))
    noise = jnp.diag(jnp.asarray(1 / scales**2, dtype))
    incoming_precision = jnp.eye(3, dtype=dtype) if direction == "forward" else noise
    eta = jnp.zeros(3, dtype)
    result = message(direction, eta, incoming_precision, matrix, noise)
    expected = np.diag(.5 / scales**2) if direction == "forward" else np.eye(3) * .5
    np.testing.assert_allclose(result[0], 0., atol=0)
    np.testing.assert_allclose(result[1][0], expected, rtol=128 * np.finfo(dtype).eps, atol=0)
    assert bool(gauss._pd_ok(incoming_precision))


@pytest.mark.parametrize("dtype,large", [(jnp.float32, 1e30), (jnp.float64, 1e200)])
def test_invalid_predictive_covariance_cannot_expose_an_inverse_placeholder(dtype, large):
    identity = jnp.ones((1, 1), dtype)
    component = forward_component(jnp.zeros(1, dtype), identity,
                                  identity * large, jnp.zeros(1, dtype), identity)
    assert not bool(component.valid)
    result = message("forward", jnp.zeros(1, dtype), identity, identity * large, identity)
    assert all(np.isnan(value).all() for value in result)


@pytest.mark.parametrize("dtype,large", [(jnp.float32, 1e30), (jnp.float64, 1e200)])
def test_invalid_backward_gram_is_not_a_rank_deficient_initialization(dtype, large):
    identity = jnp.ones((1, 1), dtype)
    result = message("backward", jnp.zeros(1, dtype), identity * large,
                     identity * large, identity * large)
    assert all(np.isnan(value).all() for value in result)


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("problem", ["indefinite", "nonsymmetric"])
def test_invalid_input_is_failure_not_initialization(direction, problem):
    precision = -jnp.eye(2) if problem == "indefinite" else jnp.array([[1., 1.], [0., 1.]])
    result = message(direction, jnp.zeros(2), precision, jnp.eye(2), jnp.eye(2))
    assert all(np.isnan(value).all() for value in result)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_large_rank_deficient_map_and_vmap_members_are_independent(dtype):
    dimension = 30
    identity = jnp.eye(dimension, dtype=dtype)
    singular = identity.at[:, -1].set(identity[:, 0])
    matrices = jnp.stack((identity, singular))

    def evaluate(matrix):
        return message("backward", jnp.zeros(dimension, dtype), identity, matrix,
                       identity, components=32)

    batched = jax.jit(jax.vmap(evaluate))(matrices)
    np.testing.assert_allclose(batched[1][0, 0], np.eye(dimension) * .5,
                               rtol=128 * np.finfo(dtype).eps, atol=0)
    np.testing.assert_array_equal(batched[1][1], 0.)
    for index in range(2):
        single = evaluate(matrices[index])
        for actual, expected in zip(batched, single):
            np.testing.assert_allclose(actual[index], expected, rtol=128 * np.finfo(dtype).eps)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_proper_component_gradients_match_scalar_covariance_oracle(direction):
    def production(value):
        matrix = value.reshape(1, 1)
        result = message(direction, jnp.array([.7]), jnp.array([[2.]]),
                         matrix, jnp.array([[3.]]), jnp.array([.2]))
        return jnp.concatenate([array.ravel() for array in result])

    def oracle(value):
        if direction == "forward":
            precision = 1 / (1 / 3 + value[0]**2 / 2)
            eta = precision * (.2 + value[0] * .7 / 2)
        else:
            precision = value[0]**2 / (1 / 3 + 1 / 2)
            eta = value[0] * (.7 / 2 - .2) / (1 / 3 + 1 / 2)
        return jnp.stack((eta, precision))

    for derivative in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(jax.jit(derivative(production))(jnp.array([1.3])),
                                   derivative(oracle)(jnp.array([1.3])), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_proper_matrix_gradients_do_not_differentiate_unused_repeated_eigenspace(direction):
    identity = jnp.eye(2)

    def production(matrix):
        return message(direction, jnp.zeros(2), identity, matrix, identity)[1][0]

    def oracle(matrix):
        return jnp.linalg.inv(identity + matrix @ matrix.T) if direction == "forward" else matrix.T @ matrix / 2

    for derivative in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(jax.jit(derivative(production))(identity),
                                   derivative(oracle)(identity), rtol=1e-12, atol=1e-12)
