"""Gaussian rule library vs moment-form ground truth (cff-4e5z).

Every message rule is checked against the covariance-form textbook result on
proper inputs, and against exact vacuous propagation on improper ones — the
property the information-form derivation was chosen for.
"""

import jax
import numpy as np
import pytest

from cffg import gauss

RNG = np.random.default_rng(7)


def spd(d):
    m = RNG.normal(size=(d, d))
    return m @ m.T + d * np.eye(d)


def info(mean, cov):
    lam = np.linalg.inv(cov)
    return lam @ mean, lam


DO, DI = 2, 3
A = RNG.normal(size=(1, DO, DI))
B = RNG.normal(size=(1, DO))
V = spd(DO)[None]
W = np.linalg.inv(V)


def test_forward_matches_moment_form():
    m_i, p_i = RNG.normal(size=DI), spd(DI)
    eta_i, lam_i = info(m_i, p_i)
    eta, lam = gauss.linear_forward(eta_i[None], lam_i[None], A, B, W)
    cov = np.linalg.inv(lam[0])
    np.testing.assert_allclose(cov, A[0] @ p_i @ A[0].T + V[0], rtol=1e-8)
    np.testing.assert_allclose(cov @ eta[0], A[0] @ m_i + B[0], rtol=1e-8)


def test_forward_vacuous_propagates():
    # Square invertible A: no information in ⇒ none out.
    a = RNG.normal(size=(1, DI, DI)) + 3 * np.eye(DI)
    v = spd(DI)[None]
    eta, lam = gauss.linear_forward(
        np.zeros((1, DI)), np.zeros((1, DI, DI)), a, np.zeros((1, DI)), np.linalg.inv(v)
    )
    assert np.abs(lam).max() < 1e-6 and np.abs(eta).max() < 1e-6


@pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.slow)])
def test_forward_vacuous_is_scale_and_coordinate_invariant(dtype):
    """Regression cff-3hgi: no absolute ridge may manufacture precision.

    Noise precision spans 24 orders and the invertible coordinate map spans
    12, independently; the symbolic vacuous limit is exactly zero in both
    working dtypes.
    """
    pairs = [(wp, ap) for wp in (-12, 0, 12) for ap in (-6, 0, 6)]
    w = np.asarray([[[10.0**wp]] for wp, _ in pairs], dtype=dtype)
    a = np.asarray([[[10.0**ap]] for _, ap in pairs], dtype=dtype)
    z1 = np.zeros((len(pairs), 1), dtype=dtype)
    z2 = np.zeros((len(pairs), 1, 1), dtype=dtype)
    eta, lam = gauss.linear_forward(z1, z2, a, z1, w)
    np.testing.assert_array_equal(np.asarray(eta), 0.0)
    np.testing.assert_array_equal(np.asarray(lam), 0.0)


def test_forward_rank_deficient_vacuous_limit_is_finite_psd():
    """A rank-deficient map retains information only off its column space."""
    a = np.array([[[1.0], [0.0]]])
    w = np.array([np.diag([1e-8, 1e8])])
    eta, lam = gauss.linear_forward(
        np.zeros((1, 1)), np.zeros((1, 1, 1)), a, np.zeros((1, 2)), w
    )
    np.testing.assert_allclose(eta, 0.0, atol=0.0)
    np.testing.assert_allclose(lam[0], np.diag([0.0, 1e8]), rtol=1e-12)
    assert np.linalg.eigvalsh(np.asarray(lam[0])).min() >= 0.0


def test_backward_matches_moment_form():
    m_o, p_o = RNG.normal(size=DO), spd(DO)
    eta_o, lam_o = info(m_o, p_o)
    eta, lam = gauss.linear_backward(eta_o[None], lam_o[None], A, B, W)
    g = np.linalg.inv(V[0] + p_o)  # (V + Σ_o)⁻¹
    np.testing.assert_allclose(lam[0], A[0].T @ g @ A[0], rtol=1e-8)
    np.testing.assert_allclose(eta[0], A[0].T @ g @ (m_o - B[0]), rtol=1e-8, atol=1e-10)


def test_backward_vacuous_propagates():
    eta, lam = gauss.linear_backward(np.zeros((1, DO)), np.zeros((1, DO, DO)), A, B, W)
    assert np.abs(lam).max() < 1e-9 and np.abs(eta).max() < 1e-9


def test_backward_float32_preserves_weak_proper_precision():
    dtype = np.float32
    eta_o = np.zeros((1, 1), dtype=dtype)
    lam_o = np.full((1, 1, 1), 1e-6, dtype=dtype)
    eta, lam = gauss.linear_backward(
        eta_o,
        lam_o,
        np.ones((1, 1, 1), dtype=dtype),
        eta_o,
        np.ones((1, 1, 1), dtype=dtype),
    )
    np.testing.assert_allclose(eta, 0.0, atol=0.0)
    np.testing.assert_allclose(float(lam[0, 0, 0]), 1e-6, rtol=2e-3)


def test_observed_backward_is_the_clamp_limit():
    y = RNG.normal(size=(1, DO))
    eta, lam = gauss.linear_observed_backward(y, np.ones(1), A, B, W)
    # Same as backward with a huge-precision message at y.
    big = 1e9
    eta_l, lam_l = gauss.linear_backward(big * y, big * np.eye(DO)[None], A, B, W)
    np.testing.assert_allclose(lam[0], lam_l[0], rtol=1e-6)
    np.testing.assert_allclose(eta[0], eta_l[0], rtol=1e-6)
    # Missing observation = vacuous.
    eta0, lam0 = gauss.linear_observed_backward(y, np.zeros(1), A, B, W)
    assert np.abs(lam0).max() == 0 and np.abs(eta0).max() == 0


def test_equality_messages_sum_others():
    eta = RNG.normal(size=(1, 3, DI))
    lam = np.stack([spd(DI) for _ in range(3)])[None]
    to, tl = gauss.equality_messages(eta, lam)
    np.testing.assert_allclose(to[0, 0], eta[0, 1] + eta[0, 2], rtol=1e-12)
    np.testing.assert_allclose(tl[0, 2], lam[0, 0] + lam[0, 1], rtol=1e-12)


def test_prior_energy_zero_against_vacuous():
    # q = f when nothing else arrives: F[q] = E[log q − log f] = 0.
    m, v = RNG.normal(size=(1, DI)), spd(DI)[None]
    f = gauss.prior_energy(np.zeros((1, DI)), np.zeros((1, DI, DI)), m, np.linalg.inv(v))
    assert abs(float(f[0])) < 1e-6


def test_linear_energy_reduces_to_input_neg_entropy():
    # Vacuous out-message: q(y,x) = f(y|x) g(x), so F = E[log g] = −H[g].
    m_i, p_i = RNG.normal(size=DI), spd(DI)
    eta_i, lam_i = info(m_i, p_i)
    f = gauss.linear_energy(
        np.zeros((1, DO)), np.zeros((1, DO, DO)), eta_i[None], lam_i[None], A, B, W
    )
    neg_h = -0.5 * np.linalg.slogdet(2 * np.pi * np.e * p_i)[1]
    np.testing.assert_allclose(float(f[0]), neg_h, rtol=1e-6)


def test_single_node_tree_identity():
    # Prior N(m0, P0) + observed emission y = Hx + noise R: node energies plus
    # the one interior-edge entropy must equal −log p(y) exactly.
    d = 2
    m0, p0 = RNG.normal(size=d), spd(d)
    h = RNG.normal(size=(1, d, d))
    r = spd(d)[None]
    w = np.linalg.inv(r)
    y = RNG.normal(size=(1, d))

    eta0, lam0 = info(m0, p0)  # prior → x message
    eta_b, lam_b = gauss.linear_observed_backward(y, np.ones(1), h, np.zeros((1, d)), w)

    f_prior = gauss.prior_energy(eta_b, lam_b, m0[None], np.linalg.inv(p0)[None])
    f_obs = gauss.linear_observed_energy(
        eta0[None], lam0[None], y, np.ones(1), h, np.zeros((1, d)), w
    )
    h_edge = gauss.edge_entropy(lam0[None], lam_b)
    bfe = float(f_prior[0] + f_obs[0] + h_edge[0])

    s = h[0] @ p0 @ h[0].T + r[0]  # marginal likelihood N(y; Hm0, S)
    resid = y[0] - h[0] @ m0
    nll = 0.5 * (np.linalg.slogdet(2 * np.pi * s)[1] + resid @ np.linalg.inv(s) @ resid)
    np.testing.assert_allclose(bfe, nll, rtol=1e-7)


@pytest.mark.parametrize("d", [1, 3])
def test_moments_round_trip(d):
    m, p = RNG.normal(size=(1, d)), spd(d)[None]
    eta, lam = np.linalg.solve(p[0], m[0])[None], np.linalg.inv(p[0])[None]
    mean, cov = gauss.moments(eta, lam)
    np.testing.assert_allclose(mean[0], m[0], rtol=1e-6)
    np.testing.assert_allclose(cov[0], p[0], rtol=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[-1.0]]),
        np.zeros((1, 1)),
        np.array([[1.0, 0.25], [0.0, 1.0]]),
        np.diag([1.0, -1.0]),
    ],
)
def test_improper_precision_readout_and_entropy_are_loud(bad):
    eta = np.zeros(bad.shape[0])
    mean, cov = gauss.moments(eta, bad)
    assert np.isnan(np.asarray(mean)).all() and np.isnan(np.asarray(cov)).all()
    assert float(gauss._neg_entropy(bad)) == np.inf


def test_invalid_factor_precision_cannot_produce_finite_energy():
    """A strong incoming message must not hide a user-mutated negative W."""
    bad_w = np.array([[[-1.0]]])
    eta = np.zeros((1, 1))
    strong = np.array([[[3.0]]])
    prior = gauss.prior_energy(eta, strong, eta, bad_w)
    linear = gauss.linear_energy(
        eta, strong, eta, strong, np.ones((1, 1, 1)), eta, bad_w
    )
    observed = gauss.linear_observed_energy(
        eta, strong, eta, np.ones(1), np.ones((1, 1, 1)), eta, bad_w
    )
    assert np.isposinf(np.asarray(prior)).all()
    assert np.isposinf(np.asarray(linear)).all()
    assert np.isposinf(np.asarray(observed)).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resolved_mean_solves_for_information_vector(dtype):
    # The Cholesky entries and precision column are exactly representable.
    root = np.array([[8., 0.], [7., 3/128]], dtype=dtype)
    precision = root @ root.T
    eta = precision[:, 1]
    with jax.enable_x64(dtype is np.float64):
        for moments in (gauss.moments, jax.jit(gauss.moments)):
            mean, covariance = moments(eta, precision)
            np.testing.assert_array_equal(mean, [0., 1.])
            assert mean.dtype == covariance.dtype == dtype
        # Two different natural vectors can share the same precision.
        mean, covariance = gauss.GAUSSIAN.from_natural((precision.T, precision))
        np.testing.assert_array_equal(mean, np.eye(2))
        assert covariance.shape == precision.shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_affine_energy_requires_joint_support(dtype):
    # One exact rank-one output likelihood cannot normalize two flat inputs.
    # A complementary partial input likelihood makes the joint proper.
    h = np.array([7., 11.], dtype=dtype)
    a = np.array([[.75, .5], [-.25, 1.]], dtype=dtype)
    b = np.array([.25, -.5], dtype=dtype)
    w = np.eye(2, dtype=dtype) * 3.
    lo = np.outer(h, h)
    li = np.array([[2., 0.], [0., 0.]], dtype=dtype)
    args = tuple(np.stack((v, v)) for v in (h, lo, np.zeros_like(h), li, a, b, w))
    args[3][0] = 0.
    with jax.enable_x64(dtype is np.float64):
        energy = np.asarray(jax.jit(gauss.linear_energy)(*args))
    assert np.isposinf(energy[0])
    # Dense joint in (y,x), evaluated in higher precision than the f32 case.
    a, b, w, h, lo, li = (v.astype(np.float64) for v in (a, b, w, h, lo, li))
    precision = np.block([[w + lo, -w @ a], [-a.T @ w, li + a.T @ w @ a]])
    natural = np.r_[w @ b + h, -a.T @ w @ b]
    residual_map = np.c_[np.eye(2), -a]
    residual = residual_map @ np.linalg.solve(precision, natural) - b
    spread = residual_map @ np.linalg.solve(precision, residual_map.T)
    expected = .5 * (
        np.linalg.slogdet(precision)[1] - np.linalg.slogdet(w)[1] - 4
        - 2*np.log(2*np.pi) + residual @ w @ residual + np.trace(w @ spread)
    )
    np.testing.assert_allclose(energy[1], expected, rtol=128*np.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.slow)])
def test_mixed_scale_d100_readout_survives(dtype):
    """Hostile scale/dimension contract: d>=100 remains proper and finite."""
    d = 100
    diag = np.logspace(-12, 12, d, dtype=dtype)
    lam = np.diag(diag)[None]
    eta = np.linspace(-1.0, 1.0, d, dtype=dtype)[None] * diag[None]
    mean, cov = gauss.moments(eta, lam)
    assert np.isfinite(np.asarray(mean)).all()
    assert np.isfinite(np.asarray(cov)).all()
    np.testing.assert_allclose(np.diag(np.asarray(cov[0])), 1.0 / diag, rtol=2e-5)

    # The same uncomfortable mixed scale cannot inject information through
    # an invertible identity map with a vacuous input.
    z1 = np.zeros((1, d), dtype=dtype)
    z2 = np.zeros((1, d, d), dtype=dtype)
    _, lam_f = gauss.linear_forward(z1, z2, np.eye(d, dtype=dtype)[None], z1, lam)
    np.testing.assert_array_equal(np.asarray(lam_f), 0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_backward_rounded_rank_one_likelihood_preserves_observed_direction(dtype):
    # The rounded rank-one Gram matrix can pass a raw Cholesky test in either dtype.
    h = np.array([1., -.2], dtype=dtype)
    a = np.array([[.85, -.15], [.1, .85]], dtype=dtype)
    b = np.array([.25, -.6], dtype=dtype)
    y = dtype(1.3)
    w = np.eye(2, dtype=dtype) * 5
    eta, lam = gauss.linear_backward(*[x[None] for x in (h*y, np.outer(h, h), a, b, w)])
    h, a, b = (x.astype(np.float64) for x in (h, a, b))
    sensor = h @ a
    variance = 1 + h @ np.linalg.solve(w.astype(np.float64), h)
    tol = 64 * np.finfo(dtype).eps
    np.testing.assert_allclose(eta[0], sensor * (y - h @ b) / variance, rtol=tol, atol=tol)
    np.testing.assert_allclose(lam[0], np.outer(sensor, sensor) / variance, rtol=tol, atol=tol)
    assert eta.dtype == lam.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.slow)])
def test_partial_information_from_weak_to_nearly_noiseless(dtype):
    h = np.ones(2, dtype=dtype)
    w = np.array([[4., -.5], [-.5, 1.]], dtype=dtype)
    a = np.array([[.875, .125], [-.25, .75]], dtype=dtype)
    b = np.array([.25, -.375], dtype=dtype)
    eps = np.finfo(dtype).eps
    scales = np.array([eps / 100, 1., 100 / eps], dtype=dtype)
    as_batch = lambda array: np.broadcast_to(array, (len(scales), *array.shape))
    for rule in (gauss.linear_forward, gauss.linear_backward):
        eta, precision = rule(scales[:, None] * h, scales[:, None, None] * np.outer(h, h),
                              as_batch(a), as_batch(b), as_batch(w))
        for i, magnitude in enumerate(scales):
            if rule is gauss.linear_forward:
                measured = np.linalg.solve(a.astype(float).T, h)
                direction, target = measured, 1. + measured @ b
            else:
                measured = h.astype(float)
                direction, target = measured @ a, 1. - measured @ b
            variance = 1. / float(magnitude) + measured @ np.linalg.solve(w.astype(float), measured)
            np.testing.assert_allclose(precision[i], np.outer(direction, direction) / variance,
                                       rtol=64 * eps, atol=0.)
            np.testing.assert_allclose(eta[i], direction * target / variance, rtol=64 * eps, atol=0.)


@pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.slow)])
def test_rectangular_prediction_with_partial_information(dtype):
    # One free input coordinate and one noisy measurement of the other.
    h = np.array([1., -1.], dtype=dtype)
    a = np.array([[1., .25], [-.25, .5], [.125, .25]], dtype=dtype)
    b = np.array([.25, -.5, .75], dtype=dtype)
    w = np.diag(np.array([.0625, 1., 16.], dtype=dtype))
    l = np.outer(h, h)
    e, pred = gauss.linear_forward(*[v[None] for v in (h, l, a, b, w)])
    a, h, w = (v.astype(float) for v in (a, h, w))
    gain = np.linalg.solve(a.T @ w @ a, a.T @ w)
    null_precision = w - w @ a @ gain
    v = h @ gain
    var = 1. + h @ np.linalg.solve(a.T @ w @ a, h)
    p = null_precision + np.outer(v, v) / var
    tol = 64 * np.finfo(dtype).eps
    np.testing.assert_allclose(pred[0], p, atol=tol, rtol=tol)
    np.testing.assert_allclose(e[0], p @ b + v / var, atol=tol, rtol=tol)


@pytest.mark.parametrize("rule", [gauss.linear_forward, gauss.linear_backward])
def test_isotropic_likelihood_directional_derivative(rule):
    import jax

    args = tuple(v[None] for v in (np.array([.5, -.25]), np.eye(2), np.array([[.8, -.1], [.2, .7]]),
                                    np.array([.1, .375]), 2. * np.eye(2)))
    delta = tuple(v[None] for v in (np.array([.1, .2]), np.array([[.13, .1], [.1, -.2]]),
                                     np.eye(2) * .125, np.array([-.075, .125]), np.eye(2) * .4))
    # Independent lanes with repeated eigenvalues exercise the compiled batch
    # derivative as well as the public eager, one-member rule.
    batched = tuple(np.stack((v, v)) for v in args)
    directions = tuple(np.stack((d, -d)) for d in delta)
    _, batched_derivative = jax.jit(
        lambda xs, ds: jax.jvp(jax.vmap(rule), xs, ds)
    )(batched, directions)
    _, derivative = jax.jvp(rule, args, delta)
    for local, lanes in zip(derivative, batched_derivative, strict=True):
        np.testing.assert_allclose(lanes, np.stack((local, -local)), rtol=1.e-10, atol=1.e-12)
    step = 1.e-5
    plus = rule(*(x + step * dx for x, dx in zip(args, delta, strict=True)))
    minus = rule(*(x - step * dx for x, dx in zip(args, delta, strict=True)))
    for tangent, p, m in zip(derivative, plus, minus, strict=True):
        np.testing.assert_allclose(tangent, (p - m) / (2 * step), rtol=1.e-7, atol=1.e-9)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_affine_gradient_keeps_independent_coordinate_units(dtype):
    import jax

    identity = np.eye(2, dtype=dtype)[None]
    zero = np.zeros((1, 2), dtype=dtype)
    precision = np.diag(np.array([1e-10, 1e10], dtype=dtype))[None]
    for rule in (gauss.linear_forward, gauss.linear_backward):
        def output(e):
            return rule(e, precision, identity, zero, precision)[0]
        _, tangent = jax.jvp(output, (zero,), (np.ones_like(zero),))
        np.testing.assert_allclose(tangent, .5, rtol=8 * np.finfo(dtype).eps)
