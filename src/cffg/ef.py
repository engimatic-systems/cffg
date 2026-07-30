"""Exponential-family math: natural parameters, log-partitions, moments (cff-8s6a).

The message contract every EF distribution satisfies (DERIVATIONS.md D2):

    p(x | η) = h(x) · exp( ηᵀ T(x) − A(η) )

- **Products are sums**: multiplying two messages of the same family adds
  their natural parameters η — exactly why discrete messages live in
  log-space and Gaussian messages in information form. A vacuous message is
  η = 0.
- **Moments are gradients**: E[T(x)] = ∇A(η), so every expectation a VMP
  rule or an FE ledger term needs comes from one closed-form gradient.
- **Entropy and KL are algebra**: H = A(η) − ηᵀ∇A(η) − E[log h];
  KL(η₁‖η₂) = (η₁−η₂)ᵀ∇A(η₁) − A(η₁) + A(η₂).
- **Conjugate likelihood messages are constants in T-space**: observing data
  under a conjugate likelihood sends a message that is just an increment to
  η (Poisson count x → Gamma-natural (x, −1); Bernoulli y → Beta-natural
  (y, 1−y)). Conditioning (cff-4e5z's Gaussian DATA convention) applies
  unchanged: the observed node contributes a closed-form energy, never a
  δ-clamp.

This module is the pure math: each family is a table entry with
`to_natural / from_natural / mean_stats (∇A) / log_partition / entropy / kl`,
NumPy-only (imports no JAX — usable at compile time and in tests; the arena
integration mirrors it in jnp). Categorical, Dirichlet, and Gaussian retain
their shape-specialized arenas, but use the same natural-parameter algebra
through the family entries below.

Parameterization conventions (standard ↔ natural):

- **Gamma(shape α, rate β)** over x > 0, T(x) = (log x, x), h = 1:
  η = (α − 1, −β);  A(η) = ln Γ(α) − α ln β.
  ∇A = (E[log x], E[x]) = (ψ(α) − ln β, α/β).
- **Beta(a, b)** over x ∈ (0,1), T(x) = (log x, log(1−x)), h = 1:
  η = (a − 1, b − 1);  A(η) = ln Γ(a) + ln Γ(b) − ln Γ(a+b).
  ∇A = (ψ(a) − ψ(a+b), ψ(b) − ψ(a+b)).

Both are proper iff the natural parameters sit inside the family's domain
(α > 0, β > 0 / a > 0, b > 0); improper messages (e.g. a bare likelihood
increment) are representable and combine by addition, exactly like a
singular-Λ Gaussian message — `is_proper` tells you when moments exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math

import numpy as np

# jax.scipy would drag JAX into cffg.core-adjacent code; math.lgamma is
# scalar-only. Vectorize the two special functions we need — accuracy is
# ample at parameter-scale arguments, and tests cross-check against JAX.


def _vectorized(fn: Callable[[float], float]):
    v = np.vectorize(fn, otypes=[np.float64])

    def wrapped(x):
        return v(np.asarray(x, dtype=np.float64))

    return wrapped


def _digamma_scalar(x: float) -> float:
    """ψ(x) by upward recurrence + asymptotic series (|err| < 1e-13 for x≥10)."""
    r = 0.0
    while x < 10.0:
        r -= 1.0 / x
        x += 1.0
    inv, inv2 = 1.0 / x, 1.0 / (x * x)
    return r + np.log(x) - 0.5 * inv - inv2 * (
        1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132)))
    )


digamma = _vectorized(_digamma_scalar)
gammaln = _vectorized(math.lgamma)


@dataclass(frozen=True)
class NaturalParameterFamily:
    """Shared message algebra for every natural-parameter representation."""

    name: str

    def product(self, *etas: np.ndarray) -> np.ndarray:
        """Multiply messages by adding their natural parameters."""
        if not etas:
            raise ValueError("a message product needs at least one natural parameter")
        total = np.asarray(etas[0])
        for eta in etas[1:]:
            total = total + np.asarray(eta)
        return total

    def product_axis(
        self, eta: np.ndarray, axis: int, keepdims: bool = False
    ) -> np.ndarray:
        """Multiply a stack of messages along one axis."""
        return np.asarray(eta).sum(axis=axis, keepdims=keepdims)

    def vacuous_like(self, eta: np.ndarray) -> np.ndarray:
        """The multiplicative identity in natural coordinates."""
        return np.zeros_like(eta)


@dataclass(frozen=True)
class EFFamily(NaturalParameterFamily):
    """One exponential family: everything rules and ledgers need, closed form.

    All functions take/return numpy arrays with a trailing axis of size
    `n_params` for natural parameters and mean statistics E[T(x)]."""

    name: str
    n_params: int
    to_natural: Callable[..., np.ndarray]  # standard params -> η
    from_natural: Callable[[np.ndarray], tuple]  # η -> standard params
    log_partition: Callable[[np.ndarray], np.ndarray]  # A(η)
    mean_stats: Callable[[np.ndarray], np.ndarray]  # ∇A(η) = E[T(x)]
    mean: Callable[[np.ndarray], np.ndarray]  # E[x]
    entropy: Callable[[np.ndarray], np.ndarray]
    is_proper: Callable[[np.ndarray], np.ndarray]

    def kl(self, eta1: np.ndarray, eta2: np.ndarray) -> np.ndarray:
        """KL(p_{η1} ‖ p_{η2}) = (η1−η2)ᵀ∇A(η1) − A(η1) + A(η2)."""
        g1 = self.mean_stats(eta1)
        return (
            ((np.asarray(eta1) - np.asarray(eta2)) * g1).sum(axis=-1)
            - self.log_partition(eta1)
            + self.log_partition(eta2)
        )


# --- Shape-specialized families ----------------------------------------------


@dataclass(frozen=True)
class CategoricalFamily(NaturalParameterFamily):
    """Categorical log-potentials with variable cardinality."""

    def to_natural(self, probability: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(np.asarray(probability))

    def from_natural(self, eta: np.ndarray) -> np.ndarray:
        return self.mean_stats(eta)

    def log_partition(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.logaddexp.reduce(np.asarray(eta), axis=axis)

    def normalize(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        eta = np.asarray(eta)
        return eta - np.expand_dims(self.log_partition(eta, axis=axis), axis=axis)

    def mean_stats(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.exp(self.normalize(eta, axis=axis))

    def mean(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        """Stable softmax used by host readouts."""
        eta = np.asarray(eta)
        shifted = eta - eta.max(axis=axis, keepdims=True)
        probability = np.exp(shifted)
        return probability / probability.sum(axis=axis, keepdims=True)

    def entropy(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        log_probability = self.normalize(eta, axis=axis)
        probability = np.exp(log_probability)
        return -np.where(
            probability > 0, probability * log_probability, 0.0
        ).sum(axis=axis)

    def is_proper(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.isfinite(self.log_partition(eta, axis=axis))

    def kl(
        self, eta1: np.ndarray, eta2: np.ndarray, axis: int = -1
    ) -> np.ndarray:
        log_probability1 = self.normalize(eta1, axis=axis)
        log_probability2 = self.normalize(eta2, axis=axis)
        probability1 = np.exp(log_probability1)
        return np.where(
            probability1 > 0,
            probability1 * (log_probability1 - log_probability2),
            0.0,
        ).sum(axis=axis)


@dataclass(frozen=True)
class DirichletFamily(NaturalParameterFamily):
    """Dirichlet natural parameters over an arbitrary simplex axis."""

    def to_natural(self, concentration: np.ndarray) -> np.ndarray:
        return np.asarray(concentration) - 1.0

    def from_natural(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(eta) + 1.0

    def log_partition(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        return self.log_partition_from_concentration(
            self.from_natural(eta), axis=axis
        )

    def log_partition_from_concentration(
        self, concentration: np.ndarray, axis: int = -1
    ) -> np.ndarray:
        concentration = np.asarray(concentration)
        return gammaln(concentration).sum(axis=axis) - gammaln(
            concentration.sum(axis=axis)
        )

    def mean_stats(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        return self.mean_stats_from_concentration(
            self.from_natural(eta), axis=axis
        )

    def mean_stats_from_concentration(
        self, concentration: np.ndarray, axis: int = -1
    ) -> np.ndarray:
        concentration = np.asarray(concentration)
        return digamma(concentration) - digamma(
            concentration.sum(axis=axis, keepdims=True)
        )

    def mean(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        return self.mean_from_concentration(self.from_natural(eta), axis=axis)

    def mean_from_concentration(
        self, concentration: np.ndarray, axis: int = -1
    ) -> np.ndarray:
        concentration = np.asarray(concentration)
        return concentration / concentration.sum(axis=axis, keepdims=True)

    def is_proper(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        return self.is_proper_concentration(self.from_natural(eta), axis=axis)

    def entropy(self, eta: np.ndarray, axis: int = -1) -> np.ndarray:
        concentration = self.from_natural(eta)
        expected_log = self.mean_stats_from_concentration(
            concentration, axis=axis
        )
        return self.log_partition_from_concentration(
            concentration, axis=axis
        ) - ((concentration - 1.0) * expected_log).sum(axis=axis)

    def kl(
        self, eta1: np.ndarray, eta2: np.ndarray, axis: int = -1
    ) -> np.ndarray:
        return (
            ((np.asarray(eta1) - np.asarray(eta2)) * self.mean_stats(eta1, axis=axis)).sum(
                axis=axis
            )
            - self.log_partition(eta1, axis=axis)
            + self.log_partition(eta2, axis=axis)
        )

    def is_proper_concentration(
        self, concentration: np.ndarray, axis: int = -1
    ) -> np.ndarray:
        concentration = np.asarray(concentration)
        return np.isfinite(concentration).all(axis=axis) & (
            concentration > 0
        ).all(axis=axis)

    def posterior(
        self, prior_concentration: np.ndarray, count_message: np.ndarray
    ) -> np.ndarray:
        """Concentration-coordinate adapter for η_prior + sufficient counts."""
        return self.product(prior_concentration, count_message)


@dataclass(frozen=True)
class GaussianFamily(NaturalParameterFamily):
    """Gaussian information pairs ``(η, Λ)`` with variable dimension."""

    def product(
        self, *naturals: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not naturals:
            raise ValueError("a message product needs at least one information pair")
        return (
            super().product(*(natural[0] for natural in naturals)),
            super().product(*(natural[1] for natural in naturals)),
        )

    def product_axis(
        self,
        natural: tuple[np.ndarray, np.ndarray],
        axis: int,
        keepdims: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            super().product_axis(natural[0], axis=axis, keepdims=keepdims),
            super().product_axis(natural[1], axis=axis, keepdims=keepdims),
        )

    @staticmethod
    def _matrix_scale(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix)
        scale = np.max(np.abs(matrix), axis=(-2, -1))
        return np.maximum(scale, np.asarray(np.finfo(matrix.dtype).tiny))

    def symmetry_ok(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix)
        dimension = matrix.shape[-1]
        error = np.max(
            np.abs(matrix - matrix.swapaxes(-1, -2)), axis=(-2, -1)
        )
        tolerance = (
            16.0
            * dimension
            * np.finfo(matrix.dtype).eps
            * self._matrix_scale(matrix)
        )
        return np.isfinite(matrix).all(axis=(-2, -1)) & (error <= tolerance)

    def is_proper_precision(self, precision: np.ndarray) -> np.ndarray:
        precision = np.asarray(precision)
        symmetric = 0.5 * (precision + precision.swapaxes(-1, -2))
        eigenvalues = np.linalg.eigvalsh(symmetric)
        return (
            self.symmetry_ok(precision)
            & np.isfinite(eigenvalues).all(axis=-1)
            & (eigenvalues[..., 0] > 0)
        )

    def is_proper(
        self, natural: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        eta, precision = natural
        return self.is_proper_precision(precision) & np.isfinite(eta).all(
            axis=-1
        )

    def to_natural(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        precision = np.linalg.inv(np.asarray(covariance))
        eta = np.einsum("...ij,...j->...i", precision, np.asarray(mean))
        return eta, precision

    def from_natural(
        self, natural: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        eta, precision = natural
        covariance = np.linalg.inv(np.asarray(precision))
        mean = np.einsum("...ij,...j->...i", covariance, np.asarray(eta))
        return mean, covariance

    def mean(
        self, natural: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        return self.from_natural(natural)[0]

    def log_partition(
        self, natural: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        eta, precision = natural
        mean, _ = self.from_natural(natural)
        _, logdet = np.linalg.slogdet(precision)
        dimension = precision.shape[-1]
        return (
            0.5 * np.sum(np.asarray(eta) * mean, axis=-1)
            - 0.5 * logdet
            + 0.5 * dimension * np.log(2.0 * np.pi)
        )

    def mean_stats(
        self, natural: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        mean, covariance = self.from_natural(natural)
        second = covariance + np.einsum("...i,...j->...ij", mean, mean)
        return mean, -0.5 * second

    def entropy(
        self, natural: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        _, precision = natural
        _, logdet = np.linalg.slogdet(precision)
        dimension = precision.shape[-1]
        return 0.5 * dimension * (1.0 + np.log(2.0 * np.pi)) - 0.5 * logdet

    def kl(self, natural1, natural2) -> np.ndarray:
        eta1, precision1 = natural1
        eta2, precision2 = natural2
        mean1, precision_stats1 = self.mean_stats(natural1)
        natural_inner_product = (
            ((eta1 - eta2) * mean1).sum(axis=-1)
            + ((precision1 - precision2) * precision_stats1).sum(
                axis=(-2, -1)
            )
        )
        return (
            natural_inner_product
            - self.log_partition(natural1)
            + self.log_partition(natural2)
        )


CATEGORICAL = CategoricalFamily(name="categorical")
DIRICHLET = DirichletFamily(name="dirichlet")
GAUSSIAN = GaussianFamily(name="gaussian")


# --- Gamma --------------------------------------------------------------------


def _gamma_std(eta):
    eta = np.asarray(eta, dtype=np.float64)
    return eta[..., 0] + 1.0, -eta[..., 1]  # (α, β)


GAMMA = EFFamily(
    name="gamma",
    n_params=2,
    to_natural=lambda shape, rate: np.stack(
        [np.asarray(shape, dtype=np.float64) - 1.0, -np.asarray(rate, dtype=np.float64)],
        axis=-1,
    ),
    from_natural=_gamma_std,
    log_partition=lambda eta: (
        gammaln(_gamma_std(eta)[0]) - _gamma_std(eta)[0] * np.log(_gamma_std(eta)[1])
    ),
    mean_stats=lambda eta: np.stack(
        [
            digamma(_gamma_std(eta)[0]) - np.log(_gamma_std(eta)[1]),  # E[log x]
            _gamma_std(eta)[0] / _gamma_std(eta)[1],  # E[x]
        ],
        axis=-1,
    ),
    mean=lambda eta: _gamma_std(eta)[0] / _gamma_std(eta)[1],
    entropy=lambda eta: (
        (lambda a, b: a - np.log(b) + gammaln(a) + (1.0 - a) * digamma(a))(*_gamma_std(eta))
    ),
    is_proper=lambda eta: (_gamma_std(eta)[0] > 0) & (_gamma_std(eta)[1] > 0),
)


# --- Beta ---------------------------------------------------------------------


def _beta_std(eta):
    eta = np.asarray(eta, dtype=np.float64)
    return eta[..., 0] + 1.0, eta[..., 1] + 1.0  # (a, b)


def _beta_logpart(eta):
    a, b = _beta_std(eta)
    return gammaln(a) + gammaln(b) - gammaln(a + b)


BETA = EFFamily(
    name="beta",
    n_params=2,
    to_natural=lambda a, b: np.stack(
        [np.asarray(a, dtype=np.float64) - 1.0, np.asarray(b, dtype=np.float64) - 1.0],
        axis=-1,
    ),
    from_natural=_beta_std,
    log_partition=_beta_logpart,
    mean_stats=lambda eta: (
        (lambda a, b: np.stack(
            [digamma(a) - digamma(a + b), digamma(b) - digamma(a + b)], axis=-1
        ))(*_beta_std(eta))
    ),
    mean=lambda eta: (lambda a, b: a / (a + b))(*_beta_std(eta)),
    entropy=lambda eta: (
        (lambda a, b: _beta_logpart(eta)
         - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b)
         + (a + b - 2.0) * digamma(a + b))(*_beta_std(eta))
    ),
    is_proper=lambda eta: (_beta_std(eta)[0] > 0) & (_beta_std(eta)[1] > 0),
)


FAMILIES = {f.name: f for f in (GAMMA, BETA)}
NATURAL_FAMILIES = {
    family.name: family
    for family in (CATEGORICAL, DIRICHLET, GAUSSIAN, GAMMA, BETA)
}

# Observation families (cff-8s6a): legal `VariableSpec.ef` tags for edges that
# are *conditioned DATA ports only*. They get no EFFamily table because their
# message algebra is degenerate — Poisson has no closed-form entropy and is
# not closed under products (h(x) = 1/x! doubles), and an unobserved
# predictive leaves the family (negative-binomial / Beta-Bernoulli mixture).
# Observing them is exact: a constant conjugate increment to the parameter
# edge (the *_likelihood_message functions below) plus a closed-form energy.
OBS_FAMILIES = frozenset({"poisson", "bernoulli"})


# --- Conjugate likelihood messages (constants in T-space) ----------------------


def poisson_likelihood_message(x: np.ndarray) -> np.ndarray:
    """Observing a count x under Poisson(λ): the exact likelihood message to
    λ in Gamma-natural coordinates is (x, −1) — λ^x e^{−λ} as a function of
    λ has T-coefficients (x, −1) (a proper Gamma(x+1, 1), as it happens;
    the canonically improper message is the vacuous η = 0)."""
    x = np.asarray(x, dtype=np.float64)
    return np.stack([x, -np.ones_like(x)], axis=-1)


def poisson_obs_energy(eta_q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Node energy of a conditioned Poisson observation against the λ-belief
    with natural params eta_q (cf. the Gaussian conditioning convention):
    −E_q[log Poisson(x|λ)] = −x·E[log λ] + E[λ] + ln x!  — closed form."""
    stats = GAMMA.mean_stats(eta_q)
    x = np.asarray(x, dtype=np.float64)
    return -(x * stats[..., 0]) + stats[..., 1] + gammaln(x + 1.0)


def bernoulli_likelihood_message(y: np.ndarray) -> np.ndarray:
    """Observing y ∈ {0,1} under Bernoulli(p): message to p in Beta-natural
    coordinates is (y, 1−y) — the textbook count increment."""
    y = np.asarray(y, dtype=np.float64)
    return np.stack([y, 1.0 - y], axis=-1)


def bernoulli_obs_energy(eta_q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """−E_q[log Bernoulli(y|p)] = −y·E[log p] − (1−y)·E[log(1−p)]."""
    stats = BETA.mean_stats(eta_q)
    y = np.asarray(y, dtype=np.float64)
    return -(y * stats[..., 0]) - (1.0 - y) * stats[..., 1]
