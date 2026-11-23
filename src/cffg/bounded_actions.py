"""Finite box conditioning of a closed, structurally factorized Gaussian plan (D15)."""

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp, log_ndtr
from jax.scipy.linalg import block_diag

from cffg.efrules import _symmetry_ok


class BoundStatus(IntEnum):
    SUCCESS = 0
    INFEASIBLE = 1
    INVALID_GAUSSIAN = 2
    NUMERICAL_FAILURE = 3


class BoundedResult(NamedTuple):
    status: jax.Array
    command: jax.Array
    mean: jax.Array
    covariance: jax.Array
    log_mass: jax.Array
    free_energy: jax.Array
    action_entropy: jax.Array


def _rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1) / 2, weights / 2


_RULES = (_rule(24), _rule(48))
_SCALAR_RULES = (_rule(64), _rule(96))


def _checked_correlation(cov):
    """Check positive definiteness in units of each coordinate's deviation."""
    scale = jnp.sqrt(jnp.diag(cov))
    if cov.shape == (1, 1):
        valid = jnp.isfinite(cov[0, 0]) & (cov[0, 0] > 0)
        return valid, jnp.ones_like(cov), scale
    correlation = (cov / scale[:, None]) / scale[None, :]
    # Each row of a correlation Cholesky factor has unit norm. Its pivot
    # subtracts a sum of up to dim products from 1. Reserve roundoff for
    # that subtraction so a singular direction cannot pass on a tiny pivot.
    dim = cov.shape[0]
    margin = dim * jnp.finfo(cov.dtype).eps
    guarded = correlation - margin * jnp.eye(dim, dtype=cov.dtype)
    valid = _symmetry_ok(correlation) & jnp.isfinite(jnp.linalg.cholesky(guarded)).all()
    return valid, correlation, scale


def _covariance_root(cov):
    valid, correlation, scale = _checked_correlation(cov)
    root = scale[:, None] * jnp.linalg.cholesky(correlation)
    return jnp.where(valid, root, jnp.nan)


def _integrate(mean, chol, low, high, rule):
    """Tensor quadrature relative to the marginal mode, with centered moments."""
    nodes, weights = (jnp.asarray(a, dtype=mean.dtype) for a in rule)
    pivot = jnp.clip(mean, low, high)
    left, right = pivot - low, high - pivot
    if mean.size == 1:
        # Resolve the boundary layer of a distant tail even on a wide interval.
        z = (pivot - mean) / chol[0, 0]
        scale = chol[0, 0] / jnp.maximum(1., abs(z))
        def side(width):
            length = jnp.log1p(width / scale)
            position = length[:, None] * nodes
            return scale * jnp.expm1(position), scale * length[:, None] * jnp.exp(position) * weights
        dl, wl = side(left)
        dr, wr = side(right)
        offsets = jnp.concatenate((-dl, dr), axis=1)
        volumes = jnp.concatenate((wl, wr), axis=1)
    else:
        offsets = jnp.concatenate((-left[:, None] * nodes, right[:, None] * nodes), axis=1)
        volumes = jnp.concatenate((left[:, None] * weights, right[:, None] * weights), axis=1)
    grid = jnp.stack(jnp.meshgrid(*offsets, indexing="ij")).reshape(mean.size, -1)
    log_volume = sum(jnp.meshgrid(*jnp.log(volumes), indexing="ij")).ravel()
    center = jax.scipy.linalg.solve_triangular(chol, pivot - mean, lower=True)
    delta = jax.scipy.linalg.solve_triangular(chol, grid, lower=True)
    penalty = jnp.sum(delta * (center[:, None] + delta / 2), axis=0)
    logits = log_volume - penalty
    integral = logsumexp(logits)
    prob = jax.nn.softmax(logits)
    shift = grid @ prob
    residual = grid - shift[:, None]
    cov = (residual * prob) @ residual.T
    mass = (integral - center @ center / 2 - jnp.log(jnp.diag(chol)).sum()
            - mean.size * jnp.log(2 * jnp.pi) / 2)
    if mean.size == 1:
        outside = jnp.logaddexp(log_ndtr((low[0]-mean[0])/chol[0, 0]),
                                log_ndtr((mean[0]-high[0])/chol[0, 0]))
        mass = jnp.where(outside < -jnp.log(2.), jnp.log1p(-jnp.exp(outside)), mass)
    return mass, pivot + shift, cov, integral + penalty @ prob


def _block(mean, cov, low, high):
    chol = _covariance_root(cov)
    valid = jnp.isfinite(chol).all() & jnp.isfinite(mean).all()
    coarse, fine = (_integrate(mean, chol, low, high, rule) for rule in (_SCALAR_RULES if mean.size == 1 else _RULES))
    mass, action, variance, entropy = fine
    tol = 128 * jnp.finfo(mean.dtype).eps
    std = jnp.sqrt(jnp.diag(variance))
    scales = (1 + abs(mass), std + abs(action), std[:, None] * std, 1 + abs(entropy))
    resolved = jnp.all(jnp.stack([jnp.all(abs(a-b) <= tol*s)
                                  for a, b, s in zip(coarse, fine, scales)]))
    resolved &= ((low < action) & (action < high)).all()
    resolved &= jnp.isfinite(jnp.linalg.cholesky(variance)).all() & (mass < 0)
    if mean.size == 2:
        # Near-unit mass loses its logarithm to subtraction in the tensor integral.
        resolved &= -mass > tol
    return valid, resolved, mass, action, variance, entropy


@dataclass(frozen=True)
class BoundedGaussianPlan:
    """Each width specifies one independent action marginal in command order.

    Bounds are static source values; preserving them distinguishes a positive
    width lost in the solve dtype from an originally empty interval.
    """

    widths: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self):
        for name in ("widths", "lower", "upper"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.widths or any(type(d) is not int or d not in (1, 2) for d in self.widths):
            raise ValueError("independent Gaussian action marginals must have dimension 1 or 2")
        for bounds in (self.lower, self.upper):
            if np.shape(bounds) != (sum(self.widths),) or not np.isfinite(bounds).all():
                raise ValueError("one finite bound required per action coordinate")

    def log_factor(self, actions):
        """Joint box-to-action log message for one or more action vectors."""
        actions = jnp.asarray(actions)
        if actions.ndim == 0 or actions.shape[-1] != sum(self.widths):
            raise ValueError("expected trailing dimension equal to the bounded action count")
        inside = (actions >= jnp.asarray(self.lower)) & (actions <= jnp.asarray(self.upper))
        return jnp.where(inside.all(axis=-1), 0., -jnp.inf)

    def to_dict(self):
        """Portable declaration. Runtime Gaussian data are supplied to solve."""
        return dict(kind="closed_gaussian_box_v1", widths=list(self.widths),
                    lower=list(self.lower), upper=list(self.upper))

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"kind", "widths", "lower", "upper"} or data["kind"] != "closed_gaussian_box_v1":
            raise ValueError("unknown closed Gaussian box declaration")
        return cls(*(tuple(data[key]) for key in ("widths", "lower", "upper")))

    def solve_joint(self, mean, covariance, gaussian_free_energy):
        """Solve from a full joint Gaussian with one bounded action block first.

        The trailing coordinates are arbitrary nuisance states. A dense input
        supplies no independence proof for a partition into multiple blocks.
        """
        if len(self.widths) != 1:
            raise ValueError("dense input requires one action block")
        mean = jnp.asarray(mean)
        covariance = jnp.asarray(covariance, mean.dtype)
        n = self.widths[0]
        if mean.ndim != 1 or mean.size < n or covariance.shape != (mean.size, mean.size):
            raise ValueError("joint Gaussian dimensions disagree")
        marginal = covariance[:n, :n]
        if mean.size == n:
            return self.solve(mean, (marginal,), gaussian_free_energy)
        # In the joint Cholesky factor, L_xu = gain @ L_uu and L_xx is
        # the conditional covariance root. Use the unshifted input here.
        root = _covariance_root(covariance)
        gain = jax.scipy.linalg.solve_triangular(root[:n, :n].T, root[n:, :n].T).T
        conditional = root[n:, n:] @ root[n:, n:].T
        return self.solve(mean[:n], (marginal,), gaussian_free_energy,
                          state_mean=mean[n:], state_gain=gain,
                          conditional_covariance=conditional)

    def solve(self, mean, covariances, gaussian_free_energy, *, state_mean=(),
              state_gain=None, conditional_covariance=None):
        """Condition product marginals and lift p(x|u) = N(mx + L(u-m), C).

        Return joint coordinates in action-then-state order. F_G is the complete
        normalized baseline objective of this same closed region. No external
        separator messages can be attached. Bind this method for jit/vmap.
        """
        mean = jnp.asarray(mean)
        n = sum(self.widths)
        if mean.shape != (n,) or mean.dtype not in (jnp.float32, jnp.float64):
            raise ValueError("expected a float32 or float64 action mean vector")
        if len(covariances) != len(self.widths):
            raise ValueError("one covariance required per independent marginal")
        dtype = mean.dtype
        low, high = (jnp.asarray(v, dtype) for v in (self.lower, self.upper))
        offsets = np.cumsum((0, *self.widths))
        parts = {}
        valid = jnp.array(True)
        resolved = jnp.array(True)
        for d in sorted(set(self.widths)):
            indices = [i for i, size in enumerate(self.widths) if size == d]
            rows = np.stack([np.arange(offsets[i], offsets[i+1]) for i in indices])
            cov = jnp.stack([jnp.asarray(covariances[i], dtype) for i in indices])
            if cov.shape[1:] != (d, d):
                raise ValueError("marginal covariance shape disagrees with declared dimension")
            ok, accurate, *values = jax.vmap(_block)(mean[rows], cov, low[rows], high[rows])
            valid &= ok.all()
            resolved &= accurate.all()
            for k, i in enumerate(indices):
                parts[i] = tuple(v[k] for v in values)
        masses, actions, variances, entropies = zip(*(parts[i] for i in range(len(parts))))
        action = jnp.concatenate(actions)
        variance = block_diag(*variances)
        mass, entropy = sum(masses), sum(entropies)
        baseline = jnp.asarray(gaussian_free_energy, dtype)
        if baseline.shape:
            raise ValueError("Gaussian free energy must be scalar")
        valid &= jnp.isfinite(baseline)
        objective = baseline - mass
        mx = jnp.asarray(state_mean, dtype)
        if mx.ndim != 1:
            raise ValueError("state mean must be a vector")
        if mx.size or state_gain is not None or conditional_covariance is not None:
            if state_gain is None or conditional_covariance is None:
                raise ValueError("state lifting requires a gain and conditional covariance")
            gain, c = (jnp.asarray(a, dtype) for a in (state_gain, conditional_covariance))
            if gain.shape != (mx.size, n) or c.shape != (mx.size, mx.size):
                raise ValueError("conditional Gaussian shape mismatch")
            proper, _, _ = _checked_correlation(c)
            valid &= proper & jnp.isfinite(gain).all() & jnp.isfinite(mx).all()
            c = c / 2 + c.T / 2
            cross = gain @ variance
            variance = jnp.block([[variance, cross.T], [cross, c + cross @ gain.T]])
            action = jnp.concatenate((action, mx + gain @ (action - mean)))
        proper, _, _ = _checked_correlation(variance)
        resolved &= (proper & jnp.isfinite(action).all() & jnp.isfinite(objective)
                     & jnp.isfinite(entropy))
        infeasible = any(h <= l for l, h in zip(self.lower, self.upper))
        status = jnp.select([infeasible, ~valid, ~resolved],
                            [BoundStatus.INFEASIBLE, BoundStatus.INVALID_GAUSSIAN,
                             BoundStatus.NUMERICAL_FAILURE], default=BoundStatus.SUCCESS)
        def read(value):
            return jnp.where(status == BoundStatus.SUCCESS, value, jnp.nan)
        return BoundedResult(status, read(action[0]), read(action), read(variance),
                             jnp.where(infeasible, -jnp.inf, read(mass)),
                             jnp.where(infeasible, jnp.inf, read(objective)), read(entropy))


class UnitBoxBelief(NamedTuple):
    status: jax.Array
    mean: jax.Array
    variance: jax.Array
    log_integral: jax.Array
    entropy: jax.Array


def unit_interval_belief(lower: float, upper: float, dtype=jnp.float32):
    """Normalize a scalar unit message on finite static endpoints in its own gauge."""
    if not np.isfinite([lower, upper]).all() or dtype not in (jnp.float32, jnp.float64):
        raise ValueError("finite endpoints and a floating solve dtype required")
    lo, hi = jnp.asarray((lower, upper), dtype)
    width = hi - lo
    mean = lo + width / 2
    variance = (width / jnp.sqrt(12.))**2
    log_width = jnp.log(width)
    valid = ((lo < mean) & (mean < hi) & (variance > 0) & jnp.isfinite(variance)
             & jnp.isfinite(log_width))
    status = jnp.where(upper <= lower, BoundStatus.INFEASIBLE,
                       jnp.where(valid, BoundStatus.SUCCESS, BoundStatus.NUMERICAL_FAILURE))
    def read(x):
        return jnp.where(status == BoundStatus.SUCCESS, x, jnp.nan)
    return UnitBoxBelief(status, read(mean), read(variance),
                         jnp.where(upper <= lower, -jnp.inf, read(log_width)), read(log_width))
