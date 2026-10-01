"""JAX exponential-family rules (cff-8s6a): the jnp mirror of cffg.ef.

cffg.ef is the NumPy contract (compile-time checks, tests, docs); this module
is the same closed forms in jnp for the jitted step and BFE, batched over a
group's members ([M, ...] leading axis) with the family static per group.
Special functions come from jax.scipy.special (digamma, gammaln): dtype-aware
under both f32 and f64 — the dependency decision recorded on cff-8s6a (no
distrax/tfp; two special functions do not justify a framework dependency).

Message algebra (DERIVATIONS.md D2): messages on an EF edge are natural
parameters η; products add; vacuous = 0. Prior nodes send a *constant* η_p
(stacked at compile time — no rule needed), conjugate observation nodes send
a constant likelihood increment, equality nodes sum the other legs. That is
the entire update vocabulary: flooding on EF subgraphs converges in one
sweep per graph diameter.

Propriety and loudness: η is proper iff it lies in the domain of A. Improper
*messages* are legal mid-flooding (a bare likelihood increment, a vacuous
zero) and cost nothing. Improper *beliefs* (sums entering the free-energy
ledger) make every term undefined, so all energy and entropy functions here
return +inf on improper input — the BFE of a not-yet-coherent state is +inf,
deliberately loud, and never NaN (an unguarded −inf edge entropy would
NaN-cancel against a +inf node energy).

The scalar arena uses Gamma/Beta directly. Categorical, Dirichlet, and
Gaussian retain their shape-specialized arenas and call the corresponding
family entries here for the same algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln

from cffg._cavities import sum_other_legs


@dataclass(frozen=True)
class JNaturalParameterFamily:
    """JAX message algebra shared by every natural-parameter family."""

    name: str

    def product(self, *etas: jnp.ndarray) -> jnp.ndarray:
        if not etas:
            raise ValueError("a message product needs at least one natural parameter")
        total = jnp.asarray(etas[0])
        for eta in etas[1:]:
            total = total + jnp.asarray(eta)
        return total

    def product_axis(
        self, eta: jnp.ndarray, axis: int, keepdims: bool = False
    ) -> jnp.ndarray:
        return jnp.asarray(eta).sum(axis=axis, keepdims=keepdims)

    def vacuous_like(self, eta: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(eta)


@dataclass(frozen=True)
class JEFFamily(JNaturalParameterFamily):
    """jnp closed forms for one family; all take η [..., n_params]."""

    name: str
    mean_stats: Callable[[jnp.ndarray], jnp.ndarray]  # ∇A(η) = E[T(x)]
    log_partition: Callable[[jnp.ndarray], jnp.ndarray]  # A(η)
    entropy: Callable[[jnp.ndarray], jnp.ndarray]
    is_proper: Callable[[jnp.ndarray], jnp.ndarray]


def _gamma_std(eta):
    return eta[..., 0] + 1.0, -eta[..., 1]  # (α, β)


def _gamma_stats(eta):
    a, b = _gamma_std(eta)
    return jnp.stack([digamma(a) - jnp.log(b), a / b], axis=-1)


def _gamma_entropy(eta):
    a, b = _gamma_std(eta)
    return a - jnp.log(b) + gammaln(a) + (1.0 - a) * digamma(a)


GAMMA = JEFFamily(
    name="gamma",
    mean_stats=_gamma_stats,
    log_partition=lambda eta: (
        (lambda a, b: gammaln(a) - a * jnp.log(b))(*_gamma_std(eta))
    ),
    entropy=_gamma_entropy,
    is_proper=lambda eta: (
        (lambda a, b: (a > 0) & (b > 0))(*_gamma_std(eta))
    ),
)


def _beta_std(eta):
    return eta[..., 0] + 1.0, eta[..., 1] + 1.0  # (a, b)


def _beta_logpart(eta):
    a, b = _beta_std(eta)
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def _beta_stats(eta):
    a, b = _beta_std(eta)
    return jnp.stack([digamma(a) - digamma(a + b), digamma(b) - digamma(a + b)], axis=-1)


def _beta_entropy(eta):
    a, b = _beta_std(eta)
    return (
        _beta_logpart(eta)
        - (a - 1.0) * digamma(a)
        - (b - 1.0) * digamma(b)
        + (a + b - 2.0) * digamma(a + b)
    )


BETA = JEFFamily(
    name="beta",
    mean_stats=_beta_stats,
    log_partition=_beta_logpart,
    entropy=_beta_entropy,
    is_proper=lambda eta: (
        (lambda a, b: (a > 0) & (b > 0))(*_beta_std(eta))
    ),
)


JFAMILIES = {f.name: f for f in (GAMMA, BETA)}

_INF = jnp.inf
_CATEGORICAL_TERM_CUTOFF = 1e-12
LOG2PI = 1.8378770664093453


# --- Shape-specialized families ----------------------------------------------


@dataclass(frozen=True)
class JCategoricalFamily(JNaturalParameterFamily):
    """Categorical log-potentials with variable cardinality."""

    def to_natural(self, probability):
        return jnp.log(probability)

    def from_natural(self, eta):
        return self.mean_stats(eta)

    def log_partition(self, eta, axis=-1):
        return jax.scipy.special.logsumexp(eta, axis=axis)

    def normalize(self, eta, axis=-1):
        return eta - jax.scipy.special.logsumexp(
            eta, axis=axis, keepdims=True
        )

    def mean_stats(self, eta, axis=-1):
        return jnp.exp(self.normalize(eta, axis=axis))

    def moments(self, eta, axis=-1):
        log_probability = self.normalize(eta, axis=axis)
        return jnp.exp(log_probability), log_probability

    def negative_entropy_from_moments(
        self,
        probability,
        log_probability,
        axis=-1,
        cutoff=_CATEGORICAL_TERM_CUTOFF,
    ):
        # Preserve the legacy ledger's insignificant-term cutoff exactly. It
        # is not a probability floor and does not create support.
        return jnp.where(
            probability > cutoff,
            probability * log_probability,
            0.0,
        ).sum(axis=axis)

    def negative_entropy(self, eta, axis=-1):
        probability, log_probability = self.moments(eta, axis=axis)
        return self.negative_entropy_from_moments(
            probability, log_probability, axis=axis
        )

    def entropy(self, eta, axis=-1):
        return -self.negative_entropy(eta, axis=axis)

    def entropy_total(self, eta, category_axis=-1):
        probability, log_probability = self.moments(
            eta, axis=category_axis
        )
        return -self.negative_entropy_from_moments(
            probability, log_probability, axis=None
        )

    def energy(self, log_belief_un, log_factor):
        """F[q] with the legacy flattening and operation order preserved."""
        members = log_belief_un.shape[0]
        log_belief = log_belief_un.reshape(members, -1)
        log_factor = log_factor.reshape(members, -1)
        log_belief = self.normalize(log_belief)
        probability = jnp.exp(log_belief)
        return jnp.where(
            probability > _CATEGORICAL_TERM_CUTOFF,
            probability * (log_belief - log_factor),
            0.0,
        ).sum(axis=-1)

    def is_proper(self, eta, axis=-1):
        return jnp.isfinite(self.log_partition(eta, axis=axis))

    def kl(self, eta1, eta2, axis=-1):
        log_probability1 = self.normalize(eta1, axis=axis)
        log_probability2 = self.normalize(eta2, axis=axis)
        probability1 = jnp.exp(log_probability1)
        return jnp.where(
            probability1 > 0,
            probability1 * (log_probability1 - log_probability2),
            0.0,
        ).sum(axis=axis)


@dataclass(frozen=True)
class JDirichletFamily(JNaturalParameterFamily):
    """Dirichlet natural parameters over an arbitrary simplex axis."""

    def to_natural(self, concentration):
        return concentration - 1.0

    def from_natural(self, eta):
        return eta + 1.0

    def log_partition(self, eta, axis=-1):
        return self.log_partition_from_concentration(
            self.from_natural(eta), axis=axis
        )

    def log_partition_from_concentration(self, concentration, axis=-1):
        return gammaln(concentration).sum(axis=axis) - gammaln(
            concentration.sum(axis=axis)
        )

    def mean_stats(self, eta, axis=-1):
        return self.mean_stats_from_concentration(
            self.from_natural(eta), axis=axis
        )

    def mean_stats_from_concentration(self, concentration, axis=-1):
        return digamma(concentration) - digamma(
            concentration.sum(axis=axis, keepdims=True)
        )

    def mean_from_concentration(self, concentration, axis=-1):
        return concentration / concentration.sum(axis=axis, keepdims=True)

    def mean(self, eta, axis=-1):
        return self.mean_from_concentration(
            self.from_natural(eta), axis=axis
        )

    def entropy_from_concentration(self, concentration, axis=-1):
        expected_log = self.mean_stats_from_concentration(
            concentration, axis=axis
        )
        return self.log_partition_from_concentration(
            concentration, axis=axis
        ) - ((concentration - 1.0) * expected_log).sum(axis=axis)

    def entropy(self, eta, axis=-1):
        return self.entropy_from_concentration(
            self.from_natural(eta), axis=axis
        )

    def expected_probability_log_probability(self, concentration, axis=-1):
        total = concentration.sum(axis=axis, keepdims=True)
        probability = concentration / total
        return probability * (
            digamma(concentration + 1.0) - digamma(total + 1.0)
        )

    def is_proper_concentration(self, concentration, axis=-1):
        return jnp.isfinite(concentration).all(axis=axis) & (
            concentration > 0
        ).all(axis=axis)

    def is_proper(self, eta, axis=-1):
        return self.is_proper_concentration(
            self.from_natural(eta), axis=axis
        )

    def kl(self, eta1, eta2, axis=-1):
        return (
            ((eta1 - eta2) * self.mean_stats(eta1, axis=axis)).sum(axis=axis)
            - self.log_partition(eta1, axis=axis)
            + self.log_partition(eta2, axis=axis)
        )

    def posterior(self, prior_concentration, count_message):
        """Concentration-coordinate adapter for η_prior + sufficient counts."""
        return self.product(prior_concentration, count_message)


def _matrix_scale(matrix: jnp.ndarray) -> jnp.ndarray:
    """Per-matrix absolute scale with a dtype-derived nonzero fallback."""
    scale = jnp.max(jnp.abs(matrix), axis=(-2, -1))
    return jnp.maximum(
        scale, jnp.asarray(jnp.finfo(matrix.dtype).tiny, dtype=matrix.dtype)
    )


def _symmetry_ok(matrix: jnp.ndarray) -> jnp.ndarray:
    """Numerical symmetry test relative to matrix scale and dimension."""
    dimension = matrix.shape[-1]
    error = jnp.max(
        jnp.abs(matrix - matrix.swapaxes(-1, -2)), axis=(-2, -1)
    )
    # Invariant protected: only a genuinely symmetric precision enters a
    # Cholesky/readout. The tolerance rescales with dtype, units, and size.
    tolerance = (
        16.0
        * dimension
        * jnp.finfo(matrix.dtype).eps
        * _matrix_scale(matrix)
    )
    return jnp.isfinite(matrix).all(axis=(-2, -1)) & (error <= tolerance)


def _pd_ok(matrix: jnp.ndarray) -> jnp.ndarray:
    """Scale-invariant strict-PD predicate, batched and JIT-safe."""
    symmetric = 0.5 * (matrix + matrix.swapaxes(-1, -2))
    cholesky = jnp.linalg.cholesky(symmetric)
    diagonal = jnp.diagonal(cholesky, axis1=-2, axis2=-1)
    sign, logdet = jnp.linalg.slogdet(symmetric)
    return (
        _symmetry_ok(matrix)
        & jnp.isfinite(cholesky).all(axis=(-2, -1))
        & (diagonal > 0).all(axis=-1)
        & (sign > 0)
        & jnp.isfinite(logdet)
    )


def _safe_pd(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(proper, safe_matrix)`` for branch-safe linear algebra."""
    proper = _pd_ok(matrix)
    symmetric = 0.5 * (matrix + matrix.swapaxes(-1, -2))
    identity = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    fallback = _matrix_scale(matrix)[..., None, None] * identity
    return proper, jnp.where(proper[..., None, None], symmetric, fallback)


def _pd_solve(matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Unregularized solve on proper matrices; finite placeholder otherwise."""
    _, safe = _safe_pd(matrix)
    cholesky = jnp.linalg.cholesky(safe)
    vector_rhs = rhs.ndim == matrix.ndim - 1
    rhs_matrix = rhs[..., None] if vector_rhs else rhs
    intermediate = jax.lax.linalg.triangular_solve(
        cholesky, rhs_matrix, left_side=True, lower=True
    )
    solved = jax.lax.linalg.triangular_solve(
        cholesky,
        intermediate,
        left_side=True,
        lower=True,
        transpose_a=True,
    )
    return solved[..., 0] if vector_rhs else solved


def _pd_inverse(matrix: jnp.ndarray) -> jnp.ndarray:
    identity = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    return _pd_solve(matrix, jnp.broadcast_to(identity, matrix.shape))


def _pd_logdet(matrix: jnp.ndarray) -> jnp.ndarray:
    """Unregularized log determinant on the safe PD branch."""
    _, safe = _safe_pd(matrix)
    cholesky = jnp.linalg.cholesky(safe)
    return 2.0 * jnp.log(
        jnp.diagonal(cholesky, axis1=-2, axis2=-1)
    ).sum(axis=-1)


@dataclass(frozen=True)
class JGaussianFamily(JNaturalParameterFamily):
    """Gaussian information pairs ``(η, Λ)`` with variable dimension."""

    def product(self, *naturals):
        if not naturals:
            raise ValueError("a message product needs at least one information pair")
        return (
            super().product(*(natural[0] for natural in naturals)),
            super().product(*(natural[1] for natural in naturals)),
        )

    def product_axis(self, natural, axis, keepdims=False):
        return (
            super().product_axis(natural[0], axis=axis, keepdims=keepdims),
            super().product_axis(natural[1], axis=axis, keepdims=keepdims),
        )

    def precision_product(self, *precisions):
        return super().product(*precisions)

    def is_proper_precision(self, precision):
        return _pd_ok(jnp.asarray(precision))

    def is_proper(self, natural):
        eta, precision = natural
        return self.is_proper_precision(precision) & jnp.isfinite(eta).all(
            axis=-1
        )

    def to_natural(self, mean, covariance):
        precision = _pd_inverse(covariance)
        eta = jnp.einsum("...ij,...j->...i", precision, mean)
        return eta, precision

    def from_natural(self, natural):
        eta, precision = natural
        covariance = _pd_inverse(precision)
        mean = jnp.einsum("...ij,...j->...i", covariance, eta)
        return mean, covariance

    def log_partition(self, natural):
        eta, precision = natural
        mean = _pd_solve(precision, eta)
        dimension = precision.shape[-1]
        return (
            0.5 * (eta * mean).sum(axis=-1)
            - 0.5 * _pd_logdet(precision)
            + 0.5 * dimension * LOG2PI
        )

    def mean_stats(self, natural):
        mean, covariance = self.from_natural(natural)
        second = covariance + jnp.einsum("...i,...j->...ij", mean, mean)
        return mean, -0.5 * second

    def mean(self, natural):
        return self.from_natural(natural)[0]

    def negative_entropy_from_precision(self, precision):
        dimension = precision.shape[-1]
        value = (
            0.5 * _pd_logdet(precision)
            - 0.5 * dimension * (1.0 + LOG2PI)
        )
        return jnp.where(_pd_ok(precision), value, jnp.inf)

    def entropy_from_precision(self, precision):
        return -self.negative_entropy_from_precision(precision)

    def entropy(self, natural):
        return self.entropy_from_precision(natural[1])

    def kl(self, natural1, natural2):
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

    def moments(self, natural):
        eta, precision = natural
        proper = self.is_proper(natural)
        covariance = _pd_inverse(precision)
        mean = jnp.einsum("...ij,...j->...i", covariance, eta)
        return (
            jnp.where(proper[..., None], mean, jnp.nan),
            jnp.where(proper[..., None, None], covariance, jnp.nan),
        )


CATEGORICAL = JCategoricalFamily(name="categorical")
DIRICHLET = JDirichletFamily(name="dirichlet")
GAUSSIAN = JGaussianFamily(name="gaussian")
JNATURAL_FAMILIES = {
    family.name: family
    for family in (CATEGORICAL, DIRICHLET, GAUSSIAN, GAMMA, BETA)
}


# --- Messages ------------------------------------------------------------------


def poisson_likelihood_message(x, mask):
    """Observed count x → Gamma-natural increment (x, −1); mask 0 = missing =
    vacuous (the node drops out, exactly the Gaussian missing-data rule)."""
    return mask[..., None] * jnp.stack([x, -jnp.ones_like(x)], axis=-1)


def bernoulli_likelihood_message(y, mask):
    """Observed flip y ∈ {0,1} → Beta-natural increment (y, 1−y)."""
    return mask[..., None] * jnp.stack([y, 1.0 - y], axis=-1)


def equality_messages(eta):
    """equality → each leg: sum of the other legs. eta [M, R, p] → same shape."""
    return sum_other_legs(eta)


# --- Node energies F[q_a] --------------------------------------------------------


def prior_energy(fam: JEFFamily, eta_in, eta_p):
    """Prior leaf f(x) = p(x | η_p): q_a ∝ f · μ_in has η_q = η_p + η_in and
    F = −H[q] − E_q[log f] with E_q[log f] = η_pᵀ∇A(η_q) − A(η_p) (h = 1)."""
    eta_q = fam.product(eta_p, eta_in)
    e_logf = (eta_p * fam.mean_stats(eta_q)).sum(axis=-1) - fam.log_partition(eta_p)
    return jnp.where(fam.is_proper(eta_q), -fam.entropy(eta_q) - e_logf, _INF)


def poisson_obs_energy(eta_in, x, mask):
    """Conditioned Poisson observation against the rate belief: q(λ) ∝
    Poisson(x|λ)·μ_in(λ), E_q[log f] = x·E[log λ] − E[λ] − lnΓ(x+1). Missing
    (mask 0) telescopes to −H[μ_in]: ∫f dx = 1, and the +H of the interior
    rate edge cancels it in the Bethe ledger — the node truly drops out."""
    eta_q = eta_in + poisson_likelihood_message(x, mask)
    stats = GAMMA.mean_stats(eta_q)
    e_logf = x * stats[..., 0] - stats[..., 1] - gammaln(x + 1.0)
    return jnp.where(
        GAMMA.is_proper(eta_q), -GAMMA.entropy(eta_q) - mask * e_logf, _INF
    )


def bernoulli_obs_energy(eta_in, y, mask):
    """Conditioned Bernoulli observation against the probability belief:
    E_q[log f] = y·E[log p] + (1−y)·E[log(1−p)]; missing as in Poisson."""
    eta_q = eta_in + bernoulli_likelihood_message(y, mask)
    stats = BETA.mean_stats(eta_q)
    e_logf = y * stats[..., 0] + (1.0 - y) * stats[..., 1]
    return jnp.where(
        BETA.is_proper(eta_q), -BETA.entropy(eta_q) - mask * e_logf, _INF
    )


def equality_energy(fam: JEFFamily, eta):
    """equality: −H[q] from the product of all legs (log f = 0 on support)."""
    eta_q = fam.product_axis(eta, axis=1)
    return jnp.where(fam.is_proper(eta_q), -fam.entropy(eta_q), _INF)


def edge_entropy(fam: JEFFamily, eta0, eta1):
    """+H[q_e] per interior EF edge (Bethe counting number d_e − 1 = 1).
    Improper edge beliefs return +inf, not −inf: the ledger is undefined and
    the total must stay loud (see module docstring)."""
    eta_q = fam.product(eta0, eta1)
    return jnp.where(fam.is_proper(eta_q), fam.entropy(eta_q), _INF)
