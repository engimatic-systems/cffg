"""Independent finite-dimensional oracle for the shifted Jensen bound in D18.6."""

from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermitenorm, softmax, xlogy

from test_destination_softmax_derivation import entropy_covariance, fixture, posterior


def optimal_shift(logits, slopes, *, budget=60):
    """Minimize the shifted upper bound in whitened drive coordinates."""
    v = slopes.mean(axis=0)
    identity = np.eye(len(v), dtype=v.dtype)
    eps = np.finfo(v.dtype).eps
    for _ in range(budget):
        centered = slopes - v
        a = logits + np.sum(centered**2, axis=1) / 2
        p = softmax(a)
        grad = -(p @ centered)
        if np.linalg.norm(grad) <= 32 * eps * (1 + np.max(np.abs(slopes))):
            return v, logsumexp(a), p
        deviations = centered + grad
        hessian = identity + deviations.T @ (p[:, None] * deviations)
        direction = np.linalg.solve(hessian, grad)
        old = logsumexp(a)
        for step in range(50):
            fraction = 2.0**-step
            trial = v - fraction * direction
            squared = np.sum((slopes - trial) ** 2, axis=1)
            value = logsumexp(logits + squared / 2)
            # Include objective roundoff near its minimum, and verify the
            # stationarity residual on every accepted iterate.
            if value <= old - 0.25 * fraction * (grad @ direction) + 4 * eps * (
                1 + abs(old)
            ):
                v = trial
                break
        else:
            raise ArithmeticError("shift line search exhausted")
    raise ArithmeticError("shift iteration budget exhausted")


def local(r, s, m, cov, w, b, *, budget=60):
    if m is None or cov is None:
        raise ValueError("proper Gaussian drive required")
    if not all(np.isfinite(x).all() for x in (r, s, m, cov, w, b)):
        raise ValueError("nonfinite data")
    for mass in (r, s):
        if (
            np.any(mass < 0)
            or abs(mass.sum() - 1) > 8 * len(mass) * np.finfo(mass.dtype).eps
        ):
            raise ValueError("normalized categorical distribution required")
    entropy = entropy_covariance(cov)
    L = np.linalg.cholesky(cov)
    # Fix a common affine gauge before exponentiation, including nonzero shifts.
    w = w - w[:, :1, :]
    b = b - b[:, :1]
    means = w @ m + b
    table = np.empty_like(means)
    precision = np.zeros_like(cov)
    gradient = np.zeros_like(m)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for i, (a, rows) in enumerate(zip(means, w)):
            _, upper, p = optimal_shift(a, rows @ L, budget=budget)
            table[i] = a - upper
            diff = rows - p @ rows
            precision += r[i] * diff.T @ (p[:, None] * diff)
            gradient += r[i] * ((s - p) @ rows)
        energy = xlogy(r, r).sum() + xlogy(s, s).sum() - entropy - r @ table @ s
        return table, precision, precision @ m + gradient, energy


def quadrature(r, s, m, cov, w, b, order):
    """Direct expectation of the exact log law and categorical-logit Hessian."""
    nodes, weights = roots_hermitenorm(order)
    L = np.linalg.cholesky(cov)
    table = np.zeros_like(b, dtype=float)
    curvature = np.zeros_like(cov, dtype=float)
    for index in product(range(order), repeat=len(m)):
        x = m + L @ nodes[list(index)]
        q = np.prod(weights[list(index)] / np.sqrt(2 * np.pi))
        logits = w @ x + b
        table += q * (logits - logsumexp(logits, axis=1)[:, None])
        for i, (p, rows) in enumerate(zip(softmax(logits, axis=1), w)):
            d = rows - p @ rows
            curvature += q * r[i] * d.T @ (p[:, None] * d)
    energy = (
        xlogy(r, r).sum() + xlogy(s, s).sum() - entropy_covariance(cov) - r @ table @ s
    )
    return table, curvature, energy


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_replication_absolute_ledger_curvature_and_source_odds(dtype):
    results = []
    for k in (2, 4, 32):
        r = np.zeros(k, dtype)
        r[:2] = 0.5
        s = np.full(k, 1 / k, dtype)
        m = np.zeros(16, dtype)
        cov = np.eye(16, dtype=dtype)
        w = np.zeros((k, k, 16), dtype)
        w[0, :, 0] = np.tile([1.0, -1.0], k // 2)
        b = np.zeros((k, k), dtype)
        table, lam, eta, energy = local(r, s, m, cov, w, b)
        cavity = np.full(k, -np.inf, dtype=dtype)
        cavity[:2] = np.log(dtype(0.5))
        got = posterior(cavity, table @ s)
        assert np.all(np.isneginf(got[2:]))
        tol = 60 * np.finfo(dtype).eps
        np.testing.assert_allclose(np.exp(got[0]), expit(-0.5), rtol=tol)
        expected = np.zeros_like(cov)
        expected[0, 0] = 0.5
        np.testing.assert_allclose(lam, expected, rtol=tol)
        np.testing.assert_allclose(eta, 0, atol=tol)
        np.testing.assert_allclose(
            energy, -0.5 * np.log(4) - entropy_covariance(cov) + 0.25, atol=tol
        )
        assert table.dtype == lam.dtype == eta.dtype == dtype
        results.append(energy)
    np.testing.assert_allclose(results, results[0], atol=tol)


@pytest.mark.parametrize("strength", [0.1, 1.0, 4.0])
def test_strong_drive_accuracy_against_adaptive_quadrature(strength):
    r = s = np.full(2, 0.5)
    w = np.array([[[strength], [-strength]], [[0.0], [0.0]]])
    m, v, b = np.zeros(1), np.eye(1), np.zeros((2, 2))
    tab, lam, _, energy = local(r, s, m, v, w, b)
    # Independent 1D quadrature split at 0 for saturated logits.
    normal = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
    penalty = (
        2
        * quad(
            lambda x: (
                normal(x) * (np.logaddexp(strength * x, -strength * x) - np.log(2))
            ),
            0,
            np.inf,
            epsabs=1e-13,
        )[0]
    )
    exact_precision = (
        2
        * quad(
            lambda x: (
                normal(x)
                * (4 * strength**2 * expit(2 * strength * x) * expit(-2 * strength * x))
            ),
            0,
            np.inf,
        )[0]
    )
    exact_energy = -np.log(2.0) - entropy_covariance(v) + penalty / 2
    exact = quadrature(r, s, m, v, w, b, 512)
    np.testing.assert_allclose(exact[2], exact_energy, atol=2e-7, rtol=0)
    np.testing.assert_allclose(exact[1], exact_precision / 2, atol=1e-5, rtol=0)
    assert energy >= exact_energy
    assert (lam[0, 0] - exact_precision / 2) >= 0
    np.testing.assert_allclose(-(tab[0] - tab[1]) @ s, strength**2 / 2)
    if strength == 4.0:
        assert expit(-penalty) / expit(-(strength**2) / 2) > 200


def test_envelope_messages_sparse_support_and_gauge():
    r, s, m, cov, w, b = fixture(3)
    s[0] = 0
    s /= s.sum()
    table, lam, eta, energy = local(r, s, m, cov, w, b)
    eq = local(
        r, s, m, cov, w + np.arange(6).reshape(3, 1, 2), b + np.arange(3)[:, None]
    )
    for a, c in zip((table, lam, eta, energy), eq):
        np.testing.assert_allclose(a, c, atol=3e-13)
    exact, _, exact_energy = quadrature(r, s, m, cov, w, b, 32)
    assert np.all(table <= exact + 1e-10)
    assert energy >= exact_energy

    # Check the factor objective without entropy while re-optimizing the shift.
    def expected(m, cov):
        return r @ local(r, s, m, cov, w, b)[0] @ s

    direction = np.array([-0.25, 0.4])
    dcov = np.array([[0.1, 0.2], [0.2, -0.1]])
    h = 1.0e-5
    dm = (expected(m + h * direction, cov) - expected(m - h * direction, cov)) / (2 * h)
    ds = (expected(m, cov + h * dcov) - expected(m, cov - h * dcov)) / (2 * h)
    np.testing.assert_allclose(dm, (eta - lam @ m) @ direction, atol=3e-9)
    np.testing.assert_allclose(ds, -np.sum(lam * dcov) / 2, atol=3e-9)
    # Removing a zero-belief destination from the denominator changes the law.
    smaller = local(r, s[1:], m, cov, w[:, 1:], b[:, 1:])[0]
    assert np.max(np.abs(smaller - table[:, 1:])) > 0.1
    for bad in (None, cov * 0):
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            local(r, s, m, bad, w, b)
    with pytest.raises(ValueError, match="categorical"):
        local(r, s * 0, m, cov, w, b)
    with pytest.raises(ArithmeticError, match="budget"):
        local(r, s, m, cov, w, b, budget=0)


def test_joint_gaussian_coordinate_converges_to_same_stationarity_point():
    # Independent convex optimization in (m,L>0,v1,v2). The two slopes and
    # offsets differ by source, so both optimal shifts vary with the cavity.
    r, s = np.array([0.3, 0.7]), np.array([0.2, 0.8])
    w = np.array([[-1.0, 0.6], [0.4, 1.3]])
    b = np.array([[0.9, -0.2], [-0.8, 0.7]])
    m0, j0 = 0.3, 1.7

    def objective(z):
        m, L = z[:2]
        v = z[2:]
        d = L * w - v[:, None]
        a = m * w + b
        corrected = a + d * d / 2
        p = softmax(corrected, axis=1)
        value = (
            j0 * ((m - m0) ** 2 + L**2) / 2
            - np.log(L)
            + r @ (logsumexp(corrected, axis=1) - a @ s)
        )
        grad = [
            j0 * (m - m0) + r @ np.sum((p - s) * w, axis=1),
            j0 * L - 1 / L + r @ np.sum(p * d * w, axis=1),
        ]
        return value, np.r_[grad, -r * np.sum(p * d, axis=1)]

    fits = []
    for start in ((0.4, 0.9, -0.3, 0.7), (-3.0, 2.0, 3.0, -2.0)):
        fit = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(None, None), (np.finfo(float).eps, None)] + [(None, None)] * 2,
            options={"ftol": np.finfo(float).eps, "gtol": 1e-10, "maxiter": 250},
        )
        assert fit.success and np.linalg.norm(objective(fit.x)[1]) < 2e-8
        fits.append(fit.x)
    np.testing.assert_allclose(*fits, atol=2e-8)
    m, L = fits[0][:2]
    _, lam, eta, _ = local(r, s, np.array([m]), np.array([[L * L]]), w[..., None], b)
    np.testing.assert_allclose(1 / L**2, j0 + lam[0, 0], atol=2e-8)
    np.testing.assert_allclose(m / L**2, j0 * m0 + eta[0], atol=2e-8)
