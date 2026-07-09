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
integration will mirror it in jnp). The engine-facing work — a scalar-EF
arena family, VariableSpec domains beyond discrete/Gaussian, and the
`poisson_obs` / `bernoulli_obs` conditioned factors — is specced in the
cff-8s6a ticket and lands separately.

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
class EFFamily:
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
