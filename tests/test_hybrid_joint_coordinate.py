"""Held-coordinate gates only: these tests never run a candidate optimizer."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import hybrid_joint as joint
from cffg.hybrid_natural import Belief, Natural, Status
from hybrid_joint_candidate import context as make_context, held_coordinate, output_arrays, to_reference
from hybrid_joint_reference import evaluate as reference_evaluate
from hybrid_joint_solver_reference import output_arrays as reference_arrays
from hybrid_joint_report import arithmetic_tolerance

context = lru_cache(maxsize=None)(make_context)


@pytest.mark.parametrize('name', ['centered_t3', 'source_masked_t2', 'variance_forward', 'coupled_t2'])
def test_held_independent_arithmetic_and_active_roundtrip(name):
    c, p, s, model = context(name)
    vector = held_coordinate(c, model)
    value = jax.jit(lambda x: joint.evaluate(c, p, s, x))(vector)
    assert int(value.status) == 0
    assert value.status.dtype == jnp.int32
    np.testing.assert_array_equal(c.pack(*c.unpack(vector)), vector)
    assert value.residual.shape == value.scales.shape == (c.residual_size,)
    reference = reference_evaluate(model, *to_reference(c, vector))
    expected = reference_arrays(reference)
    for name, actual in output_arrays(c, vector, value).items():
        atol, rtol = arithmetic_tolerance(name, c.shape[0], vector.dtype)
        np.testing.assert_allclose(actual, expected[name], atol=atol, rtol=rtol, err_msg=name)
    messages, xi = c.unpack(vector)
    for side, message in enumerate(messages):
        active = c.support[side:side + c.shape[0] - 1]
        assert bool(jnp.all(jnp.isneginf(message.constant[~active])))
        assert bool(jnp.all(message.linear[~active] == 0))
    assert np.all(np.asarray(xi)[~model.auxiliary_support] == 0)


@pytest.mark.parametrize('name,dtype', [('centered_t3', jnp.float64), ('source_masked_t2', jnp.float64),
                                        ('variance_forward', jnp.float64), ('coupled_t2', jnp.float32)])
def test_complete_residual_jvp_and_reused_transpose(name, dtype):
    c, p, s, model = context(name, dtype)
    x = held_coordinate(c, model, start=0.)
    assert_complete_derivatives(c, p, s, x)


def assert_complete_derivatives(c, p, s, x):
    dtype = x.dtype
    at = jax.jit(lambda x: joint.evaluate(c, p, s, x))
    value = at(x)
    assert int(value.status) == 0
    scales, coordinate_scales = value.scales, 1 + abs(x)
    v = jnp.cos(jnp.arange(c.size, dtype=dtype) + 1)
    v = v.at[c.message_size:].set(abs(v[c.message_size:]))
    w = jnp.sin(jnp.arange(c.residual_size, dtype=dtype) + 1)

    @jax.jit
    def products(x, scales, coordinate_scales, v, w):
        residual, forward, transpose = joint.linearize(c, p, s, x, scales, coordinate_scales)
        return residual, forward(v), transpose(w)

    residual, jv, jtw = products(x, scales, coordinate_scales, v, w)
    assert np.all(np.isfinite(jv)) and np.all(np.isfinite(jtw))
    lhs, rhs = jnp.vdot(jv, w), jnp.vdot(v, jtw)
    eps = np.finfo(dtype).eps
    assert abs(float(lhs - rhs)) / (1 + abs(float(lhs)) + abs(float(rhs))) <= 512 * eps
    step = eps**(1 / 3) * (1 + np.linalg.norm(x))
    plus = at(x + step * coordinate_scales * v)
    assert int(plus.status) == 0
    finite_difference = (plus.residual / scales - residual) / step
    tolerance = 64 * (eps**(1 / 3) if dtype == jnp.float64 else np.sqrt(eps))
    np.testing.assert_allclose(jv, finite_difference, atol=tolerance, rtol=tolerance)


def test_complete_retained_covariance_derivative_at_fixed_large_origin():
    from cffg import hybrid_natural as hn
    from test_hybrid_affine import case
    p, left, right = case(jnp.float32, 10000, 1e-4, [6.7], -67000)
    right = right._replace(center=right.origin, status=jnp.int32(0))
    left = left._replace(status=jnp.int32(left.status))
    s = jax.tree.map(lambda a, b: jnp.stack((a, b)), left, right)
    p = jax.tree.map(lambda a: a[None], p)
    c = joint.Coordinate.from_sites(s)
    messages, xi = c.unpack(jnp.zeros(c.size, jnp.float32))
    right_message = messages[1]._replace(precision=jnp.ones_like(messages[1].precision))
    x = c.pack((messages[0], right_message), xi)
    value = jax.jit(lambda x: joint.evaluate(c, p, s, x))(x)
    correction = jax.jit(lambda r: jax.vmap(lambda b: hn.marginal(b, 0).correction)(r))(value.local.regions)
    assert np.any(np.asarray(correction) != 0)
    assert_complete_derivatives(c, p, s, x)


def test_full_support_shape_formulas_and_non_diagonal_symmetric_coordinate():
    t, k, d = 11, 4, 3
    s = Natural(jnp.zeros((t, k)), jnp.zeros((t, k, d)),
                jnp.broadcast_to(jnp.eye(d), (t, k, d, d)), jnp.zeros(t, jnp.int32))
    c = joint.Coordinate.from_sites(s)
    n = 2 * (t - 1) * (k * (1 + d + d * (d + 1) // 2) - 1) + (t - 1) * k * (k - 1)
    assert c.size == n
    assert c.residual_size == n + 2 * (t - 1) * k * (1 + d + d * (d + 1) // 2)
    x = jnp.arange(n, dtype=s.linear.dtype)
    messages, xi = c.unpack(x)
    np.testing.assert_array_equal(c.pack(messages, xi), x)
    for m in messages:
        np.testing.assert_array_equal(m.precision, m.precision.swapaxes(-1, -2))


def test_packing_rejects_malformed_shapes_and_mixed_dtypes():
    c, _, _, _ = context('coupled_t2')
    messages, xi = c.unpack(jnp.zeros(c.size, c.origin.dtype))
    with pytest.raises(ValueError, match='message shapes'):
        c.pack(messages[:1], xi)
    with pytest.raises(ValueError, match='auxiliary shape'):
        c.pack(messages, xi.reshape(-1))
    mixed = messages[0]._replace(linear=messages[0].linear.astype(jnp.float32))
    with pytest.raises(ValueError, match='message dtypes'):
        c.pack((mixed, messages[1]), xi)
    with pytest.raises(ValueError, match='auxiliary dtype'):
        c.pack(messages, xi.astype(jnp.float32))
    with pytest.raises(ValueError, match='message status'):
        c.pack((messages[0]._replace(status=jnp.asarray(.5)), messages[1]), xi)


@pytest.mark.parametrize('name', ['coupled_t2', 'source_masked_t2'])
def test_invalid_message_metadata_survives_host_jit_and_vmap_packing(name):
    from cffg.hybrid_natural import coordinate_status
    c, p, s, _ = context(name)
    messages, xi = c.unpack(jnp.zeros(c.size, c.origin.dtype))
    baseline = tuple(m._replace(precision_correction=jnp.zeros_like(m.precision)) for m in messages)
    np.testing.assert_array_equal(c.pack(messages, xi), c.pack(baseline, xi))
    left, right = baseline
    corrupt = [left._replace(constant=left.constant.at[0, 0].set(value))
               for value in (jnp.nan, -jnp.inf)]
    corrupt += [left._replace(center=left.center + 1),
                left._replace(center=left.center.at[0, 0, 0].set(jnp.nan)),
                left._replace(status=jnp.full_like(left.status, int(Status.INVALID))),
                left._replace(status=jnp.full(left.status.shape, 2**32, jnp.int64)),
                left._replace(linear=left.linear.at[0, 0, 0].set(jnp.nan)),
                left._replace(precision=left.precision.at[0, 0, 0, 0].set(jnp.nan)),
                left._replace(precision=left.precision.at[0, 0, 0, 0].set(1.),
                              precision_correction=left.correction.at[0, 0, 0, 0].set(
                                  jnp.finfo(c.origin.dtype).eps))]
    assert np.all(np.asarray(coordinate_status(corrupt[-1])) == 0)
    if name == 'source_masked_t2':
        corrupt.append(left._replace(constant=left.constant.at[0, 1].set(0)))
    banks = [baseline, *((value, right) for value in corrupt)]
    batch = jax.tree.map(lambda *values: jnp.stack(values), *banks)
    host = jnp.stack([c.pack(value, xi) for value in banks])
    single = jax.jit(lambda messages: c.pack(messages, xi))
    np.testing.assert_array_equal(host, jnp.stack([single(value) for value in banks]))
    compiled = jax.jit(jax.vmap(lambda messages: c.pack(messages, xi)))(batch)
    np.testing.assert_array_equal(host, compiled)
    assert np.all(np.isfinite(host[0])) and np.all(np.isnan(host[1:]))
    for vector in host[1:]:
        unpacked, _ = c.unpack(vector)
        assert any(int(jnp.bitwise_or.reduce(coordinate_status(m))) & int(Status.NONFINITE)
                   for m in unpacked)
    status = jax.jit(lambda x: joint.evaluate(c, p, s, x).status)(compiled[1])
    assert int(status) & int(Status.NONFINITE)
    host_status = joint.evaluate(c, p, s, host[1]).status
    assert int(host_status) & int(Status.NONFINITE)


def test_represented_covariance_difference_retains_cancelling_low_words():
    high = jnp.asarray([[[2.**40]]])
    base = Belief(jnp.ones(1), jnp.zeros((1, 1)), high, jnp.zeros(1), jnp.zeros(()),
                  jnp.int32(0), jnp.zeros((1, 1)), jnp.asarray([[[2.**-15]]]))
    other = base._replace(covariance_correction=-base.covariance_correction)
    np.testing.assert_array_equal(base.full_covariance, other.full_covariance)
    np.testing.assert_array_equal(joint.covariance_difference(base, other), 2 * base.correction)


def test_stale_support_and_origins_are_rejected_at_held_coordinate():
    c, p, s, model = context('coupled_t2')
    x = held_coordinate(c, model)
    for changed in (s._replace(constant=s.constant.at[0, 0].set(-jnp.inf)),
                    s._replace(center=s.center + 1)):
        assert int(joint.context_status(c, changed)) != 0
    with pytest.raises(ValueError, match='shape or dtype'):
        c.unpack(x[:-1])
