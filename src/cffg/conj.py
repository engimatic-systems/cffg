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
from cffg._conj_sampling import sample_joint as _sample_joint


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


def _conditional_covariances(covariances: np.ndarray, *, root: bool = False) -> np.ndarray:
    """Check within-component covariance on the scale of each coordinate."""
    source_dtype = np.result_type(covariances.dtype, np.float32)
    cov = np.asarray(covariances, dtype=np.float64)
    if not np.all(np.isfinite(cov)):
        raise ValueError("positive-weight covariances must be finite")
    variances = np.diagonal(cov, axis1=-2, axis2=-1)
    if np.any(variances < 0):
        raise ValueError("covariances must be positive semidefinite")
    if cov.shape[-1] == 1 or not np.any(cov):
        return np.sqrt(cov) if root else cov
    sd = np.sqrt(variances)
    scale = sd[:, :, None] * sd[:, None, :]
    if np.any((scale == 0) & (cov != 0)):
        raise ValueError("zero variance requires zero covariance")
    with np.errstate(over="ignore"):
        corr = cov / np.where(scale == 0, 1., scale)
    # Correlations are bounded by one. Scaling prevents a large variance in
    # another coordinate from masking invalid correlations or asymmetry.
    epsilon = np.finfo(source_dtype).eps
    tolerance = 16 * cov.shape[-1] * epsilon
    if np.any(np.abs(corr) > 1 + tolerance):
        raise ValueError("covariances must be positive semidefinite")
    skew = corr - corr.swapaxes(-1, -2)
    if np.any(np.abs(skew) > tolerance):
        raise ValueError("covariances not symmetric")
    corr -= .5 * skew
    symmetric = cov + .5 * (cov.swapaxes(-1, -2) - cov)
    try:
        lower = np.linalg.cholesky(corr)
        return sd[:, :, None] * lower if root else symmetric
    except np.linalg.LinAlgError:
        pass  # Singular covariance is valid conditional evidence.
    if root:
        eigenvalues, vectors = np.linalg.eigh(corr)
    else:
        eigenvalues = np.linalg.eigvalsh(corr)
    if np.any(eigenvalues[:, 0] < -tolerance):
        raise ValueError("covariances must be positive semidefinite")
    if root:
        return sd[:, :, None] * vectors * np.sqrt(np.maximum(eigenvalues, 0.))[:, None, :]
    # A rounded singular input can have tiny negative eigenvalues. Remove
    # only those directions so accumulating more mass cannot amplify them.
    rounded = eigenvalues[:, 0] < 0
    if np.any(rounded):
        values, vectors = np.linalg.eigh(corr[rounded])
        root = vectors * np.sqrt(np.maximum(values, 0.))[:, None, :]
        symmetric[rounded] = (root @ root.swapaxes(-1, -2)) * scale[rounded]
    return symmetric


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
        within = (seconds - np.einsum("ni,nj->nij", means, means)).sum(axis=0)
        return self._pool(means, within, np.ones(len(means)))

    def update_centered(
        self, means: np.ndarray, covariances: np.ndarray, weights: np.ndarray
    ) -> "NIW":
        """Update one component from conditional moments and their masses.

        Supply unweighted means [n, d] and covariances [n, d, d], and finite
        nonnegative weights [n]. A positive-weight covariance may be singular
        (a point observation has zero covariance). Zero-weight rows contribute
        no evidence and may contain undefined moments. Pooling uses float64.
        Each subsequent batch starts at the stored posterior mean. If its
        float64 spacing approaches the spread of the observations, center
        all batches and the prior at a common origin before updating.
        """
        means, covariances, weights = map(np.asarray, (means, covariances, weights))
        if any(a.dtype.kind not in "buif" for a in (means, covariances, weights)):
            raise ValueError("NIW statistics and weights must be real numbers")
        if means.ndim != 2 or means.shape[1:] != self.mu.shape:
            raise ValueError(f"means must have shape [n,{len(self.mu)}]")
        if covariances.shape != (len(means), *self.psi.shape):
            raise ValueError(f"covariances must have shape [n,{len(self.mu)},{len(self.mu)}]")
        if weights.shape != (len(means),):
            raise ValueError("weights must have shape [n]")
        weights = weights.astype(np.float64, copy=False)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and nonnegative")
        active = weights > 0
        if not active.any():
            return self
        if not active.all():
            means, covariances, weights = means[active], covariances[active], weights[active]
        means = means.astype(np.float64, copy=False)
        if not np.all(np.isfinite(means)):
            raise ValueError("positive-weight means must be finite")
        covariances = _conditional_covariances(covariances)
        within = np.einsum("n,nij->ij", weights, covariances)
        return self._pool(means, within, weights)

    def _pool(self, means: np.ndarray, within: np.ndarray, weights: np.ndarray) -> "NIW":
        if not len(weights):
            return self
        mass = weights.sum()
        kappa = float(self.kappa)
        kn = kappa + mass
        nun = float(self.nu) + mass
        if not np.isfinite(kn) or not np.isfinite(nun):
            raise ValueError("posterior NIW mass must be finite")

        # Work in local coordinates so rounding the returned absolute mean
        # cannot change the scatter. Anchor near the group carrying more mass.
        prior_mean = np.asarray(self.mu, dtype=np.float64)
        anchor = means[np.argmax(weights)] if mass >= kappa else prior_mean
        offsets = means - anchor
        prior_offset = prior_mean - anchor
        mean_offset = (weights / kn) @ offsets + (kappa / kn) * prior_offset
        residuals = np.sqrt(weights)[:, None] * (offsets - mean_offset)
        prior_residual = np.sqrt(kappa) * (prior_offset - mean_offset)
        psin = (
            self.psi + within + residuals.T @ residuals
            + np.outer(prior_residual, prior_residual)
        )
        return replace(
            self, mu=anchor + mean_offset, kappa=kn, psi=_spd("psi_n", psin), nu=nun
        )

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

    def sample(self, key, sample_shape=()) -> tuple[np.ndarray, np.ndarray]:
        """Draw (location, covariance) on the host using one explicit JAX key.

        For batch shape S (an integer or tuple), the arrays have shapes
        S + (d,) and S + (d,d). The default returns one draw. Conditional on
        Sigma ~ IW(psi, nu), the location is N(mu, Sigma/kappa).
        """
        return _sample_joint(key, sample_shape, self.mu, self.psi, self.nu, kappa=self.kappa)


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

    def update_centered(self, means, covariances, weights) -> "MNIW":
        """Fold weighted conditional moments without subtracting raw second moments.

        The augmented design D contains the prior column-precision root, then
        the mean and covariance-root rows of each observation scaled by sqrt(w).
        Regress the residual E relative to the prior coefficients on D. If the
        fitted increment is T, the posterior precision is D.T @ D and its noise
        scale is Psi + (E - D @ T).T @ (E - D @ T).
        """
        means, covariances, weights = map(np.asarray, (means, covariances, weights))
        dy, augmented = self.m.shape
        dimension = dy + augmented - 1
        if (means.ndim != 2 or means.shape[-1] != dimension
                or covariances.shape != (len(means), dimension, dimension)
                or weights.shape != (len(means),)):
            raise ValueError("incompatible centered regression statistic shapes")
        if any(a.dtype.kind not in "buif" for a in (means, covariances, weights)):
            raise ValueError("centered regression statistics must be real")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("regression weights must be finite and nonnegative")
        active = weights > 0
        if not active.any():
            return self
        if not active.all():
            means, covariances, weights = means[active], covariances[active], weights[active]
        means = np.asarray(means, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(means).all():
            raise ValueError("positive-weight joint means must be finite")
        root = _conditional_covariances(covariances, root=True)

        matrix, offset = np.asarray(self.m, dtype=np.float64)[:, :-1], self.m[:, -1]
        y, x = means[:, :dy], means[:, dy:]
        mass = np.sqrt(weights)[:, None]
        design = [np.linalg.cholesky(np.asarray(self.lam, dtype=np.float64)).T,
                  mass * np.column_stack((x, np.ones(len(x))))]
        residual = [np.zeros((augmented, dy)), mass * ((y - offset) - x @ matrix.T)]
        if root.any():
            # Every column of the joint root contributes a paired x/y deviation.
            deviations = (mass[:, None, :] * root).swapaxes(1, 2).reshape(-1, dimension)
            xdev = deviations[:, dy:]
            design.append(np.column_stack((xdev, np.zeros(len(xdev)))))
            residual.append(deviations[:, :dy] - xdev @ matrix.T)
        design, residual = np.vstack(design), np.vstack(residual)
        if not np.isfinite(design).all() or not np.isfinite(residual).all():
            raise ValueError("regression design and residual must be finite")
        q, r = np.linalg.qr(design, mode="reduced")
        increment = np.linalg.solve(r, q.T @ residual)
        error = residual - design @ increment
        m, lam = self.m + increment.T, design.T @ design
        psi, nu = self.psi + error.T @ error, float(self.nu) + weights.sum()
        if not all(np.isfinite(a).all() for a in (m, lam, psi, nu)):
            raise ValueError("regression posterior must be finite")
        # A well-posed QR fit can lose a direction when its dense precision
        # is rounded for the next update. Check the actual stored arrays.
        for name, matrix in (("column precision", lam), ("noise scale", psi)):
            if len(matrix) > 1:
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        ef.resolved_correlation(matrix)
                except (FloatingPointError, np.linalg.LinAlgError):
                    raise ValueError(f"MNIW {name} unresolved at stored precision") from None
        return replace(self, m=m, lam=lam, psi=psi, nu=nu)

    def rollover(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A, b, cov) for the next trial's gaussian_linear params;
        cov = E[W]⁻¹ = Ψ/ν (D4.1.3)."""
        return self.m[:, :-1].copy(), self.m[:, -1].copy(), self.psi / self.nu

    def mean_cov(self) -> np.ndarray:
        """Posterior-mean noise covariance Ψ/(ν−d−1), as in D4.1.3."""
        dimension = self.psi.shape[0]
        if self.nu <= dimension + 1:
            raise ValueError("posterior-mean covariance undefined for nu <= d+1")
        return self.psi / (self.nu - dimension - 1)

    def sample(self, key, sample_shape=()) -> tuple[np.ndarray, np.ndarray]:
        """Draw (regression matrix including offset, noise covariance) on the host.

        One explicit JAX key produces batch shape S (an integer or tuple),
        with output shapes S + (dy,dx+1) and S + (dy,dy). The default draws
        one pair. Given Sigma ~ IW(psi, nu), the matrix is matrix-normal
        with center m, row covariance Sigma and column covariance inv(lam).
        """
        return _sample_joint(key, sample_shape, self.m, self.psi, self.nu, column_precision=self.lam)


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
