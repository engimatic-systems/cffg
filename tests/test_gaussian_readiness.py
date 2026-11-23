"""Input readiness preserves propriety and refuses invented nullspace moments."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import gauss
from cffg.gaussian_readiness import InputReadiness, _information_direction, input_readiness


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_proper_admission_is_exactly_the_existing_contract(dtype):
    matrices = (
        np.eye(3), np.diag([1e-25, 1., 1e25]),
        np.array([[2., .5, .1], [.5, 1., .2], [.1, .2, 3.]]),
    )
    for matrix in matrices:
        precision = jnp.asarray(matrix, dtype=dtype)
        eta = jnp.ones(3, dtype=dtype)
        assert gauss._pd_ok(precision)
        assert input_readiness(eta, precision) == InputReadiness.READY
        assert input_readiness(eta.at[0].set(jnp.nan), precision) == InputReadiness.INVALID


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_vacuity_and_structural_missing_coordinates(dtype):
    precision = jnp.diag(jnp.array([1e-25, 0., 1e25, 0.], dtype=dtype))
    eta = jnp.array([1., 0., -1., 0.], dtype=dtype)
    assert input_readiness(eta, precision) == InputReadiness.PARTIAL
    assert input_readiness(eta * 0, precision * 0) == InputReadiness.PARTIAL
    # Missing-direction information is invalid even when tiny relative to other entries.
    assert input_readiness(eta.at[1].set(1e-25), precision) == InputReadiness.INVALID
    broken = precision.at[1, 0].set(1e-25).at[0, 1].set(1e-25)
    assert input_readiness(eta, broken) == InputReadiness.INVALID
    negative = precision.at[1, 1].set(-1e-25)
    assert input_readiness(eta, negative) == InputReadiness.INVALID


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("units", [(1., 1.), (1e-12, 1e12), (1e12, 1e-12)])
def test_rotated_rank_one_information_survives_coordinate_units(dtype, units):
    units = np.asarray([*units, 1.], dtype=dtype)
    matrix = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 0.]], dtype=dtype)
    precision = jnp.asarray(units[:, None] * matrix * units[None, :])
    eta = jnp.asarray(units * np.array([2., 2., 0.], dtype=dtype))
    assert input_readiness(eta, precision) == InputReadiness.PARTIAL
    incompatible = jnp.asarray(units * np.array([2., 1., 0.], dtype=dtype))
    assert input_readiness(incompatible, precision) == InputReadiness.INVALID


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resolved_negative_curvature_and_mixed_unit_asymmetry_fail(dtype):
    matrix = jnp.array([[1., 1.1, 0.], [1.1, 1., 0.], [0., 0., 0.]], dtype=dtype)
    eta = jnp.zeros(3, dtype=dtype)
    assert input_readiness(eta, matrix) == InputReadiness.INVALID
    units = jnp.array([1e-12, 1e12, 1.], dtype=dtype)
    asymmetric = jnp.array([[1., .2, 0.], [.3, 1., 0.], [0., 0., 0.]], dtype=dtype)
    asymmetric = units[:, None] * asymmetric * units[None, :]
    assert gauss._symmetry_ok(asymmetric)  # The large coordinate can hide the small asymmetry.
    assert input_readiness(eta, asymmetric) == InputReadiness.INVALID


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_subresolution_signed_perturbations_are_not_an_exact_psd_proof(dtype):
    eps = np.finfo(dtype).eps
    base = jnp.ones((2, 2), dtype=dtype)
    compatible = jnp.ones(2, dtype=dtype)
    assert input_readiness(compatible, base) == InputReadiness.PARTIAL
    # A tolerance-based fallback cannot distinguish these perturbations from rounding.
    negative = base.at[0, 1].add(2 * eps).at[1, 0].add(2 * eps)
    assert input_readiness(compatible, negative) == InputReadiness.PARTIAL
    tiny_mismatch = compatible + jnp.array([eps, -eps], dtype=dtype)
    assert input_readiness(tiny_mismatch, base) == InputReadiness.PARTIAL
    clear_mismatch = compatible + jnp.array([2048 * eps, -2048 * eps], dtype=dtype)
    assert input_readiness(clear_mismatch, base) == InputReadiness.INVALID


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_poorly_separated_noncoordinate_nullspace_is_refused(dtype):
    eps = np.finfo(dtype).eps
    # Rank two, but the weak positive direction is too close to the nullspace.
    vector = np.array([1., 1., 1., 1.], dtype=dtype)
    weak = np.array([1., -1., 1., -1.], dtype=dtype)
    precision = np.outer(vector, vector) + np.sqrt(eps) * np.outer(weak, weak)
    assert not gauss._pd_ok(jnp.asarray(precision))
    assert input_readiness(jnp.asarray(vector), jnp.asarray(precision)) == InputReadiness.INVALID


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_direction_normalization_preserves_ratios_beyond_direct_division_range(dtype):
    large, small = (dtype(1e30), dtype(1e-30)) if dtype is np.float32 else (dtype(1e300), dtype(1e-300))
    eta = jnp.asarray([large, large / 2, small, 0.], dtype=dtype)
    units = jnp.asarray([small, small, large, 1.], dtype=dtype)
    direction = np.asarray(_information_direction(eta, units))
    assert np.isfinite(direction).all()
    np.testing.assert_allclose(direction[1] / direction[0], .5, rtol=4 * np.finfo(dtype).eps)
    assert direction[3] == 0.


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_subnormal_fallback_information_is_not_erased_as_exact_zero(dtype):
    tiny = dtype(np.finfo(dtype).tiny / 2)
    eta = jnp.asarray([tiny, -tiny], dtype=dtype)
    assert np.asarray(eta)[0] > 0  # Stored bits remain nonzero even on flushing backends.
    classify = jax.jit(input_readiness)
    assert classify(eta, jnp.ones((2, 2), dtype=dtype)) == InputReadiness.INVALID
    precision = jnp.diag(jnp.asarray([1., 0.], dtype=dtype))
    assert classify(jnp.asarray([1., tiny], dtype=dtype), precision) == InputReadiness.INVALID
    subnormal_matrix = jnp.asarray([[0., tiny], [tiny, 0.]], dtype=dtype)
    assert classify(jnp.zeros(2, dtype=dtype), subnormal_matrix) == InputReadiness.INVALID
    assert classify(jnp.zeros(1, dtype=dtype), jnp.asarray([[-tiny]], dtype=dtype)) == InputReadiness.INVALID
    assert classify(eta, jnp.eye(2, dtype=dtype)) == InputReadiness.READY
    assert classify(jnp.asarray([0., -0.], dtype=dtype), precision * 0) == InputReadiness.PARTIAL


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_larger_batched_members_remain_independent_and_jit_equivalent(dtype):
    precision = np.tile(np.eye(40, dtype=dtype), (32, 1, 1))
    eta = np.ones((32, 40), dtype=dtype)
    precision[1::3, -1, -1] = 0
    eta[1::3, -1] = 0
    precision[2::3, -1, -1] = -1
    expected = np.arange(32) % 3
    actual = jax.jit(jax.vmap(input_readiness))(jnp.asarray(eta), jnp.asarray(precision))
    np.testing.assert_array_equal(actual, expected)
    for index in range(3):
        assert actual[index] == input_readiness(jnp.asarray(eta[index]), jnp.asarray(precision[index]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_matrices_are_invalid(bad):
    precision = jnp.eye(2).at[0, 0].set(bad)
    assert input_readiness(jnp.ones(2), precision) == InputReadiness.INVALID


def test_shape_and_dtype_errors_are_explicit():
    with pytest.raises(ValueError, match="matching square"):
        input_readiness(jnp.zeros(2), jnp.eye(3))
    with pytest.raises(ValueError, match="nonempty"):
        input_readiness(jnp.zeros(0), jnp.zeros((0, 0)))
    with pytest.raises(TypeError, match="matching floating"):
        input_readiness(jnp.ones(2, dtype=jnp.int32), jnp.eye(2))
