"""Gaussian conjugate updates (cff-7c5o): DERIVATIONS.md D4 in numpy.

The between-trial half of continuous learning, mirroring the Dirichlet count
loop in cffg.agent: accumulate node-belief sufficient statistics
(`Engine.gaussian_node_stats`), fold them into NIW / MNIW posteriors, and
roll the *expected natural parameters* (mean and E[W] = ν Ψ⁻¹ — not the
point covariance; see D4.1.3) back into the next trial's params. NumPy-only:
this runs between engine invocations, never inside jit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def _spd(name: str, mat: np.ndarray) -> np.ndarray:
    """Assert SPD (exact-arithmetic guarantee per D4; a violation means the
    accumulated statistics are inconsistent, not numerical noise)."""
    if not np.allclose(mat, mat.T, atol=1e-10):
        raise ValueError(f"{name} not symmetric")
    try:
        np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name} not positive definite: broken statistics") from None
    return mat


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
        psin = self.psi + syy + self.m @ self.lam @ self.m.T - mn @ lamn @ mn.T
        return replace(self, m=mn, lam=_spd("lam_n", lamn), psi=_spd("psi_n", psin), nu=self.nu + n)

    def rollover(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A, b, cov) for the next trial's gaussian_linear params;
        cov = E[W]⁻¹ = Ψ/ν (D4.1.3)."""
        return self.m[:, :-1].copy(), self.m[:, -1].copy(), self.psi / self.nu


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
