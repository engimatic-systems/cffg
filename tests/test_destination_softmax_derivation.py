"""D18 research oracle. Only independent NumPy/SciPy operations, no runtime factor."""

from itertools import combinations

import numpy as np
import pytest
from scipy.linalg import helmert
from scipy.special import expit, logsumexp, roots_hermitenorm, xlogy


def entropy_covariance(cov):
    d = len(cov)
    if not np.isfinite(cov).all() or np.any(np.diag(cov) <= 0):
        raise ValueError("proper Gaussian covariance required")
    scales = np.sqrt(np.diag(cov))
    cor = (cov / scales[:, None]) / scales[None, :]
    eps = np.finfo(cov.dtype).eps
    if np.max(np.abs(cor - cor.T)) > 16 * d * eps:
        raise ValueError("asymmetric covariance")
    cor = (cor + cor.T) / 2
    # Require positive curvature distinguishable from roundoff in correlation units.
    np.linalg.cholesky(cor - d * eps * np.eye(d, dtype=cov.dtype))
    logdet = 2 * (np.log(scales).sum() + np.log(np.diag(np.linalg.cholesky(cor))).sum())
    return (d * (1 + np.log(2 * np.pi)) + logdet) / 2


def local(r, s, m, cov, weights, offsets):
    """Optimized bound, factor-to-variable coordinates and constrained node energy."""
    if m is None or cov is None:
        raise ValueError("proper Gaussian drive moments required")
    k, d = len(r), len(m)
    if (weights.shape != (k, k, d) or offsets.shape != (k, k)
            or s.shape != (k,) or cov.shape != (d, d)):
        raise ValueError("incompatible shapes")
    if not all(np.isfinite(x).all() for x in (r, s, m, weights, offsets)):
        raise ValueError("nonfinite input")
    for q in (r, s):
        if np.any(q < 0) or abs(q.sum() - 1) > 8 * k * np.finfo(q.dtype).eps:
            raise ValueError("normalized nonempty categorical belief required")
    hx = entropy_covariance(cov)
    with np.errstate(over="raise", invalid="raise"):
        # A common affine function of x cancels separately for each source.
        w = weights - weights[:, -1:, :]
        w = w - w.mean(axis=1, keepdims=True)
        b = offsets - offsets[:, -1:]
        b = b - b.mean(axis=1, keepdims=True)
        a = w @ m + b
        logp = a - logsumexp(a, axis=1, keepdims=True)
        g = s - np.exp(logp) + a / 2
        var = np.einsum("ikd,de,ike->i", w, cov, w, optimize=True)
        table = logp - var[:, None] / 4
        to_src, to_dst = table @ s, r @ table
        lam = np.einsum("i,ikd,ike->de", r, w, w, optimize=True) / 2
        eta = np.einsum("i,ik,ikd->d", r, g - b / 2, w)
        energy = xlogy(r, r).sum() + xlogy(s, s).sum() - hx - r @ to_src
    result = (table, to_src, to_dst, lam, eta, energy)
    if not all(np.isfinite(v).all() for v in result):
        raise FloatingPointError("nonfinite bound or message")
    return result


def posterior(cavity, message):
    if not np.all(np.isfinite(cavity) | np.isneginf(cavity)) or not np.isfinite(message).all():
        raise ValueError("invalid log scores")
    live = np.isfinite(cavity)
    if not live.any():
        raise ValueError("empty categorical support")
    result = np.full_like(message, -np.inf)
    with np.errstate(over="raise", invalid="raise"):
        values = cavity[live] + message[live]
        values -= np.max(values)
        result[live] = values - logsumexp(values)
    if not np.isfinite(result[live]).all():
        raise FloatingPointError("live scores outside working precision")
    return result


def fixture(k, d=2, dtype=np.float64):
    rng = np.random.default_rng(1734 + k)
    root = rng.normal(size=(d, d)) / np.sqrt(d)
    arrays = (rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k)), rng.normal(size=d),
              root @ root.T + np.eye(d) / 3, rng.normal(size=(k, k, d)), rng.normal(size=(k, k)))
    return tuple(np.asarray(a, dtype=dtype) for a in arrays)


def sigma_points(mean, cov):
    offsets = np.sqrt(len(mean)) * np.linalg.cholesky(cov).T
    return np.concatenate([mean + offsets, mean - offsets])


def held_value(r, s, m, cov, weights, offsets, anchors):
    """Direct pointwise Taylor bound at fixed anchors; moments exact for a quadratic."""
    value = 0.
    for x in sigma_points(m, cov):
        for i, mass in enumerate(r):
            z = weights[i] @ x + offsets[i]
            a = anchors[i]
            delta = z - a
            p = np.exp(a - logsumexp(a))
            upper = logsumexp(a) + p @ delta + np.sum((delta - delta.mean())**2) / 4
            value += mass * (s @ z - upper)
    return value / (2 * len(m))


@pytest.mark.parametrize("k", [2, 3, 4])
def test_stationary_messages_and_entropy_counting(k):
    r, s, m, v, w, b = fixture(k)
    table, src, dst, lam, eta, energy = local(r, s, m, v, w, b)
    anchors = w @ m + b
    rng = np.random.default_rng(k)
    expected = held_value(r, s, m, v, w, b, anchors)
    np.testing.assert_allclose(r @ table @ s, expected, atol=4e-14)
    entropy = -r @ np.log(r) - s @ np.log(s) + .5 * (
        len(m) * np.log(2 * np.pi * np.e) + np.linalg.slogdet(v)[1])
    np.testing.assert_allclose(energy, -expected - entropy, atol=4e-14)

    dr, ds, dm = (rng.normal(size=n) for n in (k, k, len(m)))
    dr -= dr.mean()
    ds -= ds.mean()
    h = rng.normal(size=v.shape)
    dv = (h + h.T) / 2
    # Each tangent varies just one normalized belief or Gaussian moment.
    for i, direction, derivative in ((0, dr, dr @ src), (1, ds, ds @ dst),
                                      (2, dm, dm @ (eta - lam @ m)),
                                      (3, dv, -np.sum(lam * dv) / 2)):
        def evaluate(t):
            args = [r, s, m, v, w, b]
            args[i] = args[i] + t * direction
            return held_value(*args, anchors)
        step = 3e-5
        np.testing.assert_allclose((evaluate(step) - evaluate(-step)) / (2 * step),
                                   derivative, atol=2e-9)
    assert np.linalg.eigvalsh(lam).min() >= -64 * np.finfo(lam.dtype).eps


@pytest.mark.parametrize("k", [2, 3, 4])
def test_gaussian_expectation_bound_gauge_and_ssm_mapping(k):
    r, s, m, v, w, b = fixture(k, d=1)
    result = local(r, s, m, v, w, b)
    moved = local(r, s, m, v, w + np.arange(k)[:, None, None],
                  b + np.arange(k)[:, None] * 5)
    for original, shifted in zip(result, moved, strict=True):
        np.testing.assert_allclose(original, shifted, atol=4e-14)
    nodes, mass = roots_hermitenorm(96)
    z = (m[0] + nodes[:, None, None] * np.sqrt(v[0, 0])) * w[None, :, :, 0] + b
    logp = z - logsumexp(z, axis=-1, keepdims=True)
    np.testing.assert_allclose(np.exp(logp).sum(axis=-1), 1, atol=4e-14)
    exact = np.einsum("t,tij->ij", mass / np.sqrt(2 * np.pi), logp)
    assert np.all(result[0] <= exact + 5e-13)
    if k == 2:
        np.testing.assert_allclose(np.exp(logp[..., 0]), expit(z[..., 0] - z[..., 1]))

    # ssm RecurrentTransitions shares destination slopes across source rows.
    x, control = np.array([.4, -.8, .1]), np.array([.9, -.2, .3])
    slopes, input_weights = w[0, :, 0], np.linspace(-.8, .6, k)
    for bias in (b, np.tile(b[0], (k, 1))):  # RecurrentOnlyTransitions ties biases too.
        ref = bias[None] + x[:-1, None, None] * slopes + control[1:, None, None] * input_weights
        ref -= logsumexp(ref, axis=-1, keepdims=True)
        coefficients = np.stack((slopes, input_weights), axis=1)
        logits = np.array([bias + coefficients @ [x[t], control[t+1]] for t in range(len(x)-1)])
        odds = logits - logits[..., -1:]
        np.testing.assert_allclose(odds - logsumexp(odds, axis=-1, keepdims=True), ref, atol=5e-15)


@pytest.mark.parametrize("k", [3, 4])
def test_no_affine_binary_tree_for_generic_softmax_and_converse(k):
    slopes = 2.**np.arange(k)
    # Every possible root split has a nonzero second derivative at x=0 here.
    for size in range(1, k):
        for subset in combinations(range(k), size):
            right = [j for j in range(k) if j not in subset]
            assert abs(slopes[list(subset)].var() - slopes[right].var()) > .01
    # Binary root chooses a two-leaf subtree with probability 1/2.
    x = np.array([-1., 0., 1.])
    cross_leaf_log_odds = np.log(.5 * expit(x)) - np.log(.5 / (k - 2))
    assert abs(np.diff(cross_leaf_log_odds, n=2)[0]) > .2


def test_source_specific_and_shared_coefficient_statistics():
    k = 4
    r, s, m, v, w, b = fixture(k)
    r[0] = 0.
    r /= r.sum()
    u = helmert(k).T
    beta = u.T @ np.concatenate((w, b[:, :, None]), axis=2)
    # Tie the slopes, leaving source-specific intercepts.
    beta[:, :, :-1] = beta[0, :, :-1]
    full = u @ beta
    anchors = full[:, :, :-1] @ m + full[:, :, -1]
    g = s - np.exp(anchors - logsumexp(anchors, axis=1, keepdims=True)) + anchors / 2
    phi = np.r_[m, 1.]
    second = np.outer(phi, phi)
    second[:-1, :-1] += v
    linear = r[:, None, None] * (g @ u)[:, :, None] * phi
    gradient = linear - r[:, None, None] * (beta @ second) / 2
    np.testing.assert_array_equal(gradient[0], 0)
    rng = np.random.default_rng(29)
    directions = [rng.normal(size=beta.shape), np.zeros_like(beta)]
    directions[1][:, :, :-1] = rng.normal(size=(k-1, len(m)))
    for direction in directions:
        def objective(t):
            z = u @ (beta + t * direction)
            return held_value(r, s, m, v, z[:, :, :-1], z[:, :, -1], anchors)
        h = .001
        pred1 = np.sum(gradient * direction)
        pred2 = -.5 * np.sum(r[:, None, None] * direction * (direction @ second))
        np.testing.assert_allclose((objective(h) - objective(-h)) / (2*h), pred1, atol=2e-10)
        np.testing.assert_allclose((objective(h) - 2*objective(0) + objective(-h)) / h**2,
                                   pred2, atol=1e-8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_replication_changes_surrogate_source_odds_at_scale(dtype):
    k, d = 32, 16
    r, s = np.zeros(k, dtype=dtype), np.full(k, 1 / k, dtype=dtype)
    r[:2] = .5
    m, cov = np.zeros(d, dtype=dtype), np.eye(d, dtype=dtype)
    w, b = np.zeros((k, k, d), dtype=dtype), np.zeros((k, k), dtype=dtype)
    w[0, :, 0] = np.tile([1, -1], k // 2)
    table, src, dst, lam, eta, energy = local(r, s, m, cov, w, b)
    assert table.shape == (k, k) and lam.shape == (d, d)
    assert all(y.dtype == dtype for y in (table, src, dst, lam, eta))
    cavity = np.full(k, -np.inf, dtype=dtype)
    cavity[:2] = np.log(dtype(.5))
    logpost = posterior(cavity, src)
    assert np.all(np.isneginf(logpost[2:]))
    np.testing.assert_allclose(np.exp(logpost[0]), expit(-8), rtol=32*np.finfo(dtype).eps)

    def exact_penalty(order):
        x, weights = roots_hermitenorm(order)
        return weights @ (np.logaddexp(x, -x) - np.log(2)) / np.sqrt(2*np.pi)
    exact = exact_penalty(96)
    np.testing.assert_allclose(exact, exact_penalty(192), atol=1e-12, rtol=0)
    np.testing.assert_allclose(expit(-exact), .4074378859299, atol=1e-12, rtol=0)
    np.testing.assert_allclose(src[1] - src[0], 8., atol=32*np.finfo(dtype).eps)


def test_invalid_support_gaussian_and_numerical_failure():
    r, s, m, v, w, b = fixture(3)
    s[1] = 0
    s /= s.sum()
    _, _, dst, *_ = local(r, s, m, v, w, b)
    cavity = np.array([0., -np.inf, -1000.])
    updated = posterior(cavity, dst)
    assert np.isneginf(updated[1]) and np.isfinite(updated[2])
    with pytest.raises(ValueError, match="empty"):
        posterior(np.full(3, -np.inf), dst)
    with pytest.raises(ValueError, match="categorical"):
        local(r, np.zeros_like(s), m, v, w, b)
    with pytest.raises(ValueError, match="proper Gaussian"):
        local(r, s, None, None, w, b)
    for invalid in (np.zeros_like(v), np.diag([1., -1.]), np.diag([np.nan, 1.])):
        with pytest.raises(ValueError, match="Gaussian"):
            local(r, s, m, invalid, w, b)
    with pytest.raises(np.linalg.LinAlgError):
        local(r, s, m, np.ones_like(v), w, b)
    with pytest.raises(ValueError, match="nonfinite"):
        local(r, s, m, v, w, b * np.inf)
    huge = np.full_like(w, np.finfo(w.dtype).max)
    huge[:, -1] *= -1
    with pytest.raises(FloatingPointError):
        local(r, s, m, v, huge, b)


def test_uncertain_coefficients_cross_moments_and_covariance_coordinates():
    """Independent product cubature for E[W.T H W], E[W.T H b] and coefficient precision."""
    k, d = 3, 2
    _, s, m, v, _, _ = fixture(k, d)
    u = helmert(k).T
    rng = np.random.default_rng(71)
    theta = rng.normal(size=(k - 1) * (d + 1))
    factor = rng.normal(size=(len(theta), len(theta)))
    vc = .3 * np.eye(len(theta)) + factor @ factor.T / 8
    beta = theta.reshape(k - 1, d + 1)
    phi = np.r_[m, 1.]
    moment = np.outer(phi, phi)
    moment[:-1, :-1] += v
    a = u @ beta @ phi
    log_norm = logsumexp(a)
    p = np.exp(a - log_norm)
    g = s - p + a / 2

    def value(mean, variance, coef, cv):
        xpoints = sigma_points(mean, variance)
        design = np.column_stack((xpoints, np.ones(len(xpoints))))
        coeffs = sigma_points(coef, cv).reshape(-1, k - 1, d + 1)
        z = np.einsum("kj,njp,mp->nmk", u, coeffs, design)
        delta = z - a
        return np.mean(z @ s - log_norm - delta @ p - (delta**2).sum(axis=-1) / 4)

    covariance = vc.reshape(k - 1, d + 1, k - 1, d + 1)
    row_cov = sum(covariance[j, :, j, :] for j in range(k - 1))
    expected_gram = beta.T @ beta + row_cov
    lam = expected_gram[:-1, :-1] / 2
    eta = beta[:, :-1].T @ (u.T @ g) - expected_gram[:-1, -1] / 2
    coef_lam = np.kron(np.eye(k - 1), moment) / 2
    coef_eta = np.outer(u.T @ g, phi).ravel()
    penalty = (np.trace(beta[:, :-1] @ v @ beta[:, :-1].T) + np.trace(row_cov @ moment)) / 4
    np.testing.assert_allclose(value(m, v, theta, vc), s @ a - log_norm - penalty, atol=2e-13)

    for i, gradient in enumerate((eta - lam @ m, -lam / 2, coef_eta - coef_lam @ theta, -coef_lam / 2)):
        direction = rng.normal(size=gradient.shape)
        if direction.ndim == 2:
            direction = (direction + direction.T) / 2
        def at(t):
            args = [m, v, theta, vc]
            args[i] = args[i] + t * direction
            return value(*args)
        h = 3e-5
        np.testing.assert_allclose((at(h) - at(-h)) / (2*h), np.sum(gradient * direction), atol=2e-9)
