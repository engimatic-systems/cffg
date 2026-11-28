"""Complete support, Jacobians and atomic failures of the interval algebra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import intervals as iv
from test_neural_delta_derivation import Unit, interval_tree


def cells(rows, capacity, dtype):
    n = len(rows)
    lo, hi, h = (np.zeros((n, capacity), dtype) for _ in range(3))
    count = np.array([len(row) for row in rows], np.int32)
    for i, row in enumerate(rows):
        for j, (l, u, v) in enumerate(row):
            lo[i, j], hi[i, j], h[i, j] = l, u, v
    return iv.IntervalMessage(*map(jnp.asarray, (lo, hi, h, count)))


def broadcast_map(spec, d, dtype):
    return tuple(jnp.full(d, v, dtype) for v in spec)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jit_inverse_density_ledger_contrast(dtype):
    d = 32
    t = np.arange(d, dtype=dtype)
    left = dtype(0.5) + t / 16
    right = dtype(2.0) + t / 8
    params = tuple(map(jnp.asarray, (t, -t, left, right)))
    x = cells([[(k - 2, k + 2, -np.log(4.0))] for k in t], 3, dtype)
    y = cells(
        [[(-k - l, -k + r / 2, 0.0)] for k, l, r in zip(t, left, right)], 3, dtype
    )
    forward = jax.jit(iv.pushforward)(x, *params)
    back = jax.jit(iv.pullback)(y, *params)
    assert forward.status == back.status == iv.IntervalStatus.OK
    np.testing.assert_array_equal(forward.message.count, np.full(d, 2))
    np.testing.assert_array_equal(back.message.count, np.ones(d))
    np.testing.assert_allclose(
        forward.message.log_height[:, :2],
        -np.log(4 * np.stack((left, right), axis=1)),
        rtol=8 * np.finfo(dtype).eps,
    )
    xpost = jax.jit(iv.product)(x, back.message)
    ypost = jax.jit(iv.product)(y, forward.message)
    assert xpost.status == ypost.status == iv.IntervalStatus.OK
    ix, iy = (jax.jit(iv.integral)(p.message) for p in (xpost, ypost))
    assert ix.status == iy.status == iv.IntervalStatus.OK
    expected = sum(
        interval_tree(Unit(l, r, k, -k), (k - 2, k + 2), (-k - l, -k + r / 2), dtype)[
            "free_energy"
        ]
        for k, l, r in zip(t, left, right)
    )
    tol = 32 * np.finfo(dtype).eps * d
    np.testing.assert_allclose(-ix.log_mass, expected, atol=tol)
    np.testing.assert_allclose(ix.log_mass, iy.log_mass, atol=tol)
    # The included negative and positive lengths in the input chart are 1 and 1/2.
    jacobian = (2 * np.log(left) + np.log(right)).sum() / 3
    np.testing.assert_allclose(iy.entropy - ix.entropy, jacobian, atol=tol)


def test_vmap_overflow_gaps_and_roundoff_are_atomic():
    def src(rows):
        return cells([rows], 2, np.float32)

    m = src([(-2, 2, 0), (3, 4, 0)])
    template = broadcast_map((0, 0, 0.5, 2), 1, np.float32)
    args = [
        template,
        broadcast_map((0, 0, 1, 1), 1, np.float32),
        broadcast_map((0, 2**27, 0.5, 2), 1, np.float32),
        broadcast_map((0, 0, 0, 2), 1, np.float32),
    ]
    parallel = jax.jit(jax.vmap(iv.pushforward, in_axes=(None, 0, 0, 0, 0)))
    result = parallel(m, *jax.tree.map(lambda *xs: jnp.stack(xs), *args))
    np.testing.assert_array_equal(
        result.status,
        [
            iv.IntervalStatus.CAPACITY,
            iv.IntervalStatus.OK,
            iv.IntervalStatus.NUMERICAL,
            iv.IntervalStatus.INVALID,
        ],
    )
    np.testing.assert_array_equal(result.message.count[:, 0], [0, 2, 0, 0])
    np.testing.assert_array_equal(result.message.lower[1], m.lower)
    assert iv.pullback(m, *template).status == iv.IntervalStatus.OK
    product = jax.jit(iv.product)
    fail = product(src([(0, 2, 0), (3, 5, 0)]), src([(1, 4, 0), (4, 5, 0)]))
    assert fail.status == iv.IntervalStatus.CAPACITY and fail.message.count[0] == 0
    ok = product(src([(0, 2, 0), (3, 5, 1)]), src([(1, 4, 2)]))
    np.testing.assert_array_equal(ok.message.lower, [[1, 3]])
    np.testing.assert_array_equal(ok.message.upper, [[2, 4]])
    np.testing.assert_array_equal(ok.message.log_height, [[2, 3]])
    # Distinct, disjoint cells can round into touching intervals although both
    # individually retain nonzero width. The physical gap must survive too.
    gapped = src([(-32, 0, 0), (1, 32, 1)])
    assert (
        iv.pushforward(gapped, *broadcast_map((0, 2**26, 1, 1), 1, np.float32)).status
        == iv.IntervalStatus.NUMERICAL
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_absolute_log_scale_and_unbounded_cells(dtype):
    unit = iv.unit(1, 3, dtype)
    iso = broadcast_map((0, 0, 0.5, 2), 1, dtype)
    forward = iv.pushforward(unit, *iso)
    assert forward.status == iv.IntervalStatus.OK
    np.testing.assert_allclose(forward.message.log_height[0, :2], np.log([2, 0.5]))
    pulled = iv.pullback(unit, *iso)
    assert pulled.message.count[0] == 1 and pulled.message.log_height[0, 0] == 0
    positive = cells([[(-1, 1, 0), (10, 12, -1000)]], 3, dtype)
    exact = iv.integral(positive)
    assert exact.status == iv.IntervalStatus.OK and positive.count[0] == 2
    assert exact.entropy == pytest.approx(np.log(2))
    shifted = iv.integral(positive._replace(log_height=positive.log_height + 1e4))
    assert shifted.entropy == pytest.approx(float(exact.entropy))
    assert shifted.log_mass == pytest.approx(1e4 + np.log(2))
    tail = iv.integral(unit)
    assert tail.status == iv.IntervalStatus.IMPROPER and np.isposinf(tail.log_mass)
    empty = iv.integral(cells([[(-np.inf, np.inf, 0)], []], 1, dtype))
    assert empty.status == iv.IntervalStatus.INFEASIBLE and np.isneginf(empty.log_mass)
    large = np.finfo(dtype).max
    wide = iv.integral(cells([[(-large, large, 0)]], 1, dtype))
    assert wide.status == iv.IntervalStatus.OK
    np.testing.assert_allclose(
        wide.log_mass, np.log(large) + np.log(2), rtol=4 * np.finfo(dtype).eps
    )
    invalid = iv.integral(cells([[(0, 2, 0), (1, 3, 0)]], 2, dtype))
    assert invalid.status == iv.IntervalStatus.INVALID and np.isnan(invalid.log_mass)
