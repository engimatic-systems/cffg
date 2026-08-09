"""Gaussian conjugate updates: DERIVATIONS.md D4 and D8 in NumPy.

The between-trial half of continuous learning, mirroring the Dirichlet count
loop in cffg.agent: accumulate node-belief sufficient statistics, fold them
into NIW / MNIW / Pólya–Gamma Gaussian posteriors, and roll point parameters
back into the next trial. Gaussian covariance families use the expected
precision E[W] = ν Ψ⁻¹, not the point covariance (see D4.1.3). NumPy-only:
these updates run between engine invocations, never inside jit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cffg import ef


def _spd(name: str, mat: np.ndarray) -> np.ndarray:
    """Assert SPD (exact-arithmetic guarantee per D4; a violation means the
    accumulated statistics are inconsistent, not numerical noise)."""
    if not bool(ef.GAUSSIAN.symmetry_ok(mat)):
        raise ValueError(f"{name} not symmetric")
    try:
        np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name} not positive definite: broken statistics") from None
    return mat


def _symmetrize_cancellation(
    name: str,
    matrix: np.ndarray,
    terms: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Accept only roundoff-sized skew from an algebraically symmetric sum."""
    matrix = np.asarray(matrix)
    if not np.all(np.isfinite(matrix)) or any(
        not np.all(np.isfinite(term)) for term in terms
    ):
        raise ValueError(f"{name} must be finite")
    scale = sum(float(np.max(np.abs(term), initial=0.0)) for term in terms)
    scale = max(scale, float(np.finfo(matrix.dtype).tiny))
    error = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
    # The bound protects symmetry through cancellation of the named terms.
    tolerance = 16.0 * matrix.shape[-1] * np.finfo(matrix.dtype).eps * scale
    if error > tolerance:
        raise ValueError(f"{name} not symmetric")
    return 0.5 * (matrix + matrix.T)


@dataclass(frozen=True)
class NIW:
    """Normal-inverse-Wishart over a gaussian node's (mean, cov)."""

    mu: np.ndarray  # [d]
    kappa: float
    psi: np.ndarray  # [d, d]
    nu: float

    def __post_init__(self):
        d = self.mu.shape[0]
        _spd("psi", self.psi)
        if self.kappa <= 0 or self.nu <= d - 1:
            raise ValueError(f"improper NIW: kappa={self.kappa}, nu={self.nu}, d={d}")

    def update(self, means: np.ndarray, seconds: np.ndarray) -> "NIW":
        """Fold n trials of soft stats: means [n, d] = E[x] per trial,
        seconds [n, d, d] = E[xxᵀ] per trial (D4.2 NIW update)."""
        means = np.atleast_2d(np.asarray(means, dtype=np.float64))
        seconds = np.asarray(seconds, dtype=np.float64).reshape(-1, *self.psi.shape)
        n = means.shape[0]
        kn = self.kappa + n
        nun = self.nu + n
        mun = (self.kappa * self.mu + means.sum(axis=0)) / kn
        within = (seconds - np.einsum("ni,nj->nij", means, means)).sum(axis=0)
        scatter = np.einsum("ni,nj->nij", means, means).sum(axis=0)
        psin = (
            self.psi
            + within
            + scatter
            + self.kappa * np.outer(self.mu, self.mu)
            - kn * np.outer(mun, mun)
        )
        return replace(self, mu=mun, kappa=kn, psi=_spd("psi_n", psin), nu=nun)

    def rollover(self) -> tuple[np.ndarray, np.ndarray]:
        """(mean, cov) to write into the next trial's gaussian node params:
        cov = E[W]⁻¹ = Ψ/ν — the VMP geometric-mean parameter (D4.1.3)."""
        return self.mu.copy(), self.psi / self.nu

    def mean_cov(self) -> np.ndarray:
        """The posterior-mean covariance Ψ/(ν−d−1) — for *reporting*, not
        for rollover (see D4.1.3)."""
        d = self.mu.shape[0]
        if self.nu <= d + 1:
            raise ValueError("posterior-mean covariance undefined for nu <= d+1")
        return self.psi / (self.nu - d - 1)


@dataclass(frozen=True)
class MNIW:
    """Matrix-normal-inverse-Wishart over a gaussian_linear node's
    ([A b], V), with x̃ = [x; 1] absorbing the offset."""

    m: np.ndarray  # [dy, dx+1]  mean of [A b]
    lam: np.ndarray  # [dx+1, dx+1]  column precision
    psi: np.ndarray  # [dy, dy]
    nu: float

    def __post_init__(self):
        _spd("lam", self.lam)
        _spd("psi", self.psi)
        if self.nu <= self.psi.shape[0] - 1:
            raise ValueError(f"improper MNIW: nu={self.nu}")

    def update(self, syy: np.ndarray, syx: np.ndarray, sxx: np.ndarray, n: float) -> "MNIW":
        """Fold accumulated stats over n node occurrences: syy = Σ E[yyᵀ],
        syx = Σ E[yx̃ᵀ], sxx = Σ E[x̃x̃ᵀ] (D4.2 MNIW update)."""
        lamn = self.lam + sxx
        mn = np.linalg.solve(lamn.T, (self.m @ self.lam + syx).T).T
        prior_square = self.m @ self.lam @ self.m.T
        posterior_square = mn @ lamn @ mn.T
        psin = self.psi + syy + prior_square - posterior_square
        psin = _symmetrize_cancellation(
            "psi_n",
            psin,
            (self.psi, syy, prior_square, posterior_square),
        )
        return replace(
            self,
            m=mn,
            lam=_spd("lam_n", lamn),
            psi=_spd("psi_n", psin),
            nu=self.nu + n,
        )

    def rollover(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A, b, cov) for the next trial's gaussian_linear params;
        cov = E[W]⁻¹ = Ψ/ν (D4.1.3)."""
        return self.m[:, :-1].copy(), self.m[:, -1].copy(), self.psi / self.nu


def _numpy_symmetry_ok(matrices: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(matrices), axis=(-2, -1))
    scale = np.maximum(scale, np.finfo(matrices.dtype).tiny)
    error = np.max(
        np.abs(matrices - matrices.swapaxes(-1, -2)), axis=(-2, -1)
    )
    tolerance = (
        16.0 * matrices.shape[-1] * np.finfo(matrices.dtype).eps * scale
    )
    return np.all(np.isfinite(matrices), axis=(-2, -1)) & (
        error <= tolerance
    )


def _batched_spd(name: str, matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    if not np.all(np.isfinite(matrices)):
        raise ValueError(f"{name} must be finite")
    if matrices.ndim < 2 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(f"{name} must end in square matrices")
    if not bool(np.all(_numpy_symmetry_ok(matrices))):
        raise ValueError(f"{name} not symmetric")
    try:
        np.linalg.cholesky(matrices)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"{name} not positive definite: broken statistics"
        ) from None
    return matrices


def _batched_psd(name: str, matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    if not np.all(np.isfinite(matrices)):
        raise ValueError(f"{name} must be finite")
    if matrices.ndim < 2 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(f"{name} must end in square matrices")
    if not bool(np.all(_numpy_symmetry_ok(matrices))):
        raise ValueError(f"{name} not symmetric")
    symmetric = 0.5 * (matrices + matrices.swapaxes(-1, -2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = np.max(np.abs(symmetric), axis=(-2, -1))
    scale = np.maximum(scale, np.finfo(matrices.dtype).tiny)
    tolerance = (
        16.0
        * matrices.shape[-1]
        * np.finfo(matrices.dtype).eps
        * scale
    )
    if np.any(eigenvalues[..., 0] < -tolerance):
        raise ValueError(f"{name} not positive semidefinite: broken statistics")
    return symmetric


@dataclass(frozen=True)
class PGWeights:
    """Independent Gaussian posteriors over recurrence coefficients.

    The final coefficient axis is augmented as ``[R_il; r_il]`` for source
    regime ``i`` and stick ``l``. Parameters are stored in Gaussian natural
    form so independent factor occurrences add without an intermediate solve.
    """

    eta: np.ndarray  # [K, K-1, d+1]
    precision: np.ndarray  # [K, K-1, d+1, d+1]

    def __post_init__(self):
        eta = np.asarray(self.eta, dtype=np.float64)
        precision = np.asarray(self.precision, dtype=np.float64)
        if eta.ndim != 3:
            raise ValueError(f"eta must have shape [K,K-1,d+1], got {eta.shape}")
        regimes, sticks, augmented_dimension = eta.shape
        if regimes < 2 or sticks != regimes - 1 or augmented_dimension < 2:
            raise ValueError(
                "eta must have K>=2, K-1 sticks, and a positive drive dimension"
            )
        if precision.shape != eta.shape + (augmented_dimension,):
            raise ValueError(
                f"precision shape {precision.shape} != {eta.shape + (augmented_dimension,)}"
            )
        if not np.all(np.isfinite(eta)):
            raise ValueError("eta must be finite")
        object.__setattr__(self, "eta", eta.copy())
        object.__setattr__(
            self, "precision", _batched_spd("precision", precision).copy()
        )

    @property
    def regimes(self) -> int:
        return self.eta.shape[0]

    @property
    def dimension(self) -> int:
        return self.eta.shape[-1] - 1

    def update(self, linear: np.ndarray, quadratic: np.ndarray) -> "PGWeights":
        """Add one or a batch of PG natural-statistic increments."""
        linear = np.asarray(linear, dtype=np.float64)
        quadratic = np.asarray(quadratic, dtype=np.float64)
        linear_tail = self.eta.shape
        quadratic_tail = self.precision.shape
        if (
            linear.ndim < len(linear_tail)
            or linear.shape[-len(linear_tail) :] != linear_tail
        ):
            raise ValueError(
                f"linear must end in {linear_tail}, got {linear.shape}"
            )
        if (
            quadratic.ndim < len(quadratic_tail)
            or quadratic.shape[-len(quadratic_tail) :] != quadratic_tail
        ):
            raise ValueError(
                f"quadratic must end in {quadratic_tail}, got {quadratic.shape}"
            )
        linear_batch = linear.shape[: -len(linear_tail)]
        quadratic_batch = quadratic.shape[: -len(quadratic_tail)]
        if linear_batch != quadratic_batch:
            raise ValueError(
                "linear and quadratic occurrence dimensions must match"
            )
        if not np.all(np.isfinite(linear)):
            raise ValueError("linear statistics must be finite")
        quadratic = _batched_psd("quadratic statistics", quadratic)
        axes = tuple(range(len(linear_batch)))
        linear_increment = linear.sum(axis=axes) if axes else linear
        quadratic_increment = quadratic.sum(axis=axes) if axes else quadratic
        return PGWeights(
            eta=self.eta + linear_increment,
            precision=self.precision + quadratic_increment,
        )

    def mean(self) -> np.ndarray:
        """Posterior mean of every augmented stick coefficient."""
        return np.linalg.solve(self.precision, self.eta[..., None])[..., 0]

    def covariance(self) -> np.ndarray:
        """Posterior covariance of every augmented stick coefficient."""
        identity = np.eye(self.eta.shape[-1], dtype=self.precision.dtype)
        return np.linalg.solve(self.precision, identity)

    def rollover(self) -> tuple[np.ndarray, np.ndarray]:
        """Point weights and offsets consumed by the next inference state."""
        mean = self.mean()
        return mean[..., :-1].copy(), mean[..., -1].copy()


def vague_niw(d: int, scale: float = 1.0) -> NIW:
    """A weakly-informative starting prior: unit-ish scatter, minimal counts."""
    return NIW(mu=np.zeros(d), kappa=1e-3, psi=scale * np.eye(d), nu=float(d + 2))


def vague_mniw(dy: int, dx: int, scale: float = 1.0) -> MNIW:
    return MNIW(
        m=np.zeros((dy, dx + 1)),
        lam=1e-3 * np.eye(dx + 1),
        psi=scale * np.eye(dy),
        nu=float(dy + 2),
    )


def vague_pg_weights(
    regimes: int,
    dimension: int,
    strength: float = 1e-3,
) -> PGWeights:
    """Proper weak zero-mean prior for recurrence stick coefficients.

    ``strength`` is prior information, not a numerical ridge; callers should
    choose it in the units of the augmented drive features.
    """
    if regimes < 2 or dimension < 1:
        raise ValueError("PG weights require regimes>=2 and dimension>=1")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("PG prior strength must be finite and positive")
    augmented_dimension = dimension + 1
    eta = np.zeros((regimes, regimes - 1, augmented_dimension))
    precision = np.broadcast_to(
        strength * np.eye(augmented_dimension),
        (regimes, regimes - 1, augmented_dimension, augmented_dimension),
    ).copy()
    return PGWeights(eta=eta, precision=precision)
