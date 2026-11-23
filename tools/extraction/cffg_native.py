"""Affine extraction inference using CFFG's production directed Gaussian messages.

The caller supplies local Taylor coefficients, carries the filter moments between
ticks, and clips the returned physical action. This backend schedules the affine
messages directly. Its timings exclude template compilation, Engine.run, and
the engine's fixed-point certificate.
"""

import numpy as np

from cffg.directed_chain import Affine, Natural, _natural, _observation_update


def _array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _cov(value, shape, name, *, semidefinite=False):
    array = _array(value, shape, name)
    transpose = array.swapaxes(-2, -1)
    magnitude = np.max(abs(array), axis=(-2, -1), keepdims=True)
    tolerance = 16 * array.shape[-1] * np.finfo(array.dtype).eps * magnitude
    if np.any(abs(array - transpose) > tolerance):
        raise ValueError(f"{name} must be symmetric")
    array = (array + transpose) * 0.5
    if semidefinite:
        if np.any(np.linalg.eigvalsh(array)[..., 0] < -tolerance[..., 0, 0]):
            raise ValueError(f"{name} must be positive semidefinite")
    else:
        np.linalg.cholesky(array)
    return array


def _finite_moments(mean, covariance):
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise FloatingPointError("nonfinite Gaussian posterior moments")
    return mean, covariance


class Backend:
    """Stateless native inference apart from the last observation's diagnostic."""

    def __init__(self):
        self.last_innovation_nll = None

    def update(self, m, P, A, b, Q, H, R, y):
        """Predict and condition once. A missing row contributes zero log evidence."""
        self.last_innovation_nll = None
        n, k = len(m), len(H)
        mean = _array(m, (n,), "m")
        cov = _cov(P, (n, n), "P", semidefinite=True)
        transition = Affine(_array(A, (n, n), "A"), _array(b, (n,), "b"),
                            _cov(Q, (n, n), "Q", semidefinite=True))
        H = _array(H, (k, n), "H")
        R = _cov(R, (k, k), "R")
        mean, cov = _finite_moments(*transition.predict(mean, cov))
        if y is None:
            self.last_innovation_nll = 0.
            return mean, cov

        y = _array(y, (k,), "y")
        # Work in standard normal coordinates. A deterministic coordinate has
        # no latent noise direction and contributes only to the emission offset.
        try:
            scale = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            eigenvalues, vectors = np.linalg.eigh(cov)
            tol = n * np.finfo(cov.dtype).eps * np.max(abs(eigenvalues))
            if eigenvalues[0] < -tol:
                raise ValueError("predicted covariance must be positive semidefinite") from None
            scale = vectors * np.sqrt(np.maximum(eigenvalues, 0.))
        emission = Affine(H @ scale, H @ mean, R)
        standard = Natural(np.zeros(n), np.eye(n))
        likelihood, nll = _observation_update(
            emission, np.linalg.inv(R), standard.eta, standard.precision, y, True)
        posterior = (standard + likelihood).moments()
        state_from_standard = Affine(scale, mean, np.zeros_like(cov))
        result = _finite_moments(*state_from_standard.predict(*posterior))
        if not np.isfinite(nll):
            raise FloatingPointError("nonfinite Gaussian innovation evidence")
        self.last_innovation_nll = nll
        return result

    def plan(self, m, P, As, bs, Bs, Qs, C, targets, Rgoal, U0, Ucov):
        """Return E[u0 | all outlet preferences], in physical flow units."""
        n, h, nu = len(m), len(As), 2
        if h == 0:
            raise ValueError("planning needs at least one transition")
        m = _array(m, (n,), "m")
        P = _cov(P, (n, n), "P", semidefinite=True)
        As = _array(As, (h, n, n), "As")
        bs = _array(bs, (h, n), "bs")
        Bs = _array(Bs, (h, n, nu), "Bs")
        Qs = _cov(Qs, (h, n, n), "Qs", semidefinite=True)
        C = _array(C, (1, n), "C")
        targets = _array(targets, (h, 1), "targets")
        goal_cov = _cov(np.atleast_2d(Rgoal), (1, 1), "Rgoal")
        U0 = _array(U0, (h, nu), "U0")
        Ucov = _cov(Ucov, (nu, nu), "Ucov")

        # Pull each Gaussian preference back through the exact outlet projection.
        outlet = Affine(C, np.zeros(1), np.zeros((1, 1)))
        goal_precision = np.linalg.inv(goal_cov)
        future = Natural(np.zeros(n), np.zeros((n, n)))
        for j in reversed(range(h)):
            goal = outlet.backward(Natural(goal_precision @ targets[j], goal_precision))
            if j:
                # Marginalize the independent local control prior in the dynamics.
                offset, noise = Affine(Bs[j], bs[j], Qs[j]).predict(U0[j], Ucov)
                transition = Affine(As[j], offset, noise)
            else:
                # Marginalize the initial state instead, leaving u0 as the input.
                offset, noise = Affine(As[0], bs[0], Qs[0]).predict(m, P)
                transition = Affine(Bs[0], offset, noise)
            future = transition.backward(goal + future)
        posterior = _natural(U0[0], Ucov) + future
        first, _ = _finite_moments(*posterior.moments())
        return first
