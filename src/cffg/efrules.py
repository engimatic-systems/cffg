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
a constant likelihood increment, equality nodes send sum-minus-self. That is
the entire update vocabulary: flooding on EF subgraphs converges in one
sweep per graph diameter.

Propriety and loudness: η is proper iff it lies in the domain of A. Improper
*messages* are legal mid-flooding (a bare likelihood increment, a vacuous
zero) and cost nothing. Improper *beliefs* (sums entering the free-energy
ledger) make every term undefined, so all energy and entropy functions here
return +inf on improper input — the BFE of a not-yet-coherent state is +inf,
deliberately loud, and never NaN (an unguarded −inf edge entropy would
NaN-cancel against a +inf node energy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln


@dataclass(frozen=True)
class JEFFamily:
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
    return eta.sum(axis=1, keepdims=True) - eta


# --- Node energies F[q_a] --------------------------------------------------------


def prior_energy(fam: JEFFamily, eta_in, eta_p):
    """Prior leaf f(x) = p(x | η_p): q_a ∝ f · μ_in has η_q = η_p + η_in and
    F = −H[q] − E_q[log f] with E_q[log f] = η_pᵀ∇A(η_q) − A(η_p) (h = 1)."""
    eta_q = eta_p + eta_in
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
    eta_q = eta.sum(axis=1)
    return jnp.where(fam.is_proper(eta_q), -fam.entropy(eta_q), _INF)


def edge_entropy(fam: JEFFamily, eta0, eta1):
    """+H[q_e] per interior EF edge (Bethe counting number d_e − 1 = 1).
    Improper edge beliefs return +inf, not −inf: the ledger is undefined and
    the total must stay loud (see module docstring)."""
    eta_q = eta0 + eta1
    return jnp.where(fam.is_proper(eta_q), fam.entropy(eta_q), _INF)
