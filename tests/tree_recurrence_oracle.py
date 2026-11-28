"""Small NumPy reference for a labeled binary gate tree. Rows are sources, columns are destinations."""

from typing import NamedTuple

import numpy as np


def leaf_paths(tree):
    """An integer is a leaf label. A pair gives (accept child, reject child) in preorder."""
    leaves = {}
    gates = 0

    def visit(node, path):
        nonlocal gates
        if type(node) is int:
            if node in leaves:
                raise ValueError("duplicate destination label")
            leaves[node] = path
        elif isinstance(node, tuple) and len(node) == 2:
            gate = gates
            gates += 1
            for child, sign in zip(node, (+1, -1)):
                visit(child, path + ((gate, sign),))
        else:
            raise ValueError("expected a leaf label or two ordered children")

    visit(tree, ())
    if gates < 1 or set(leaves) != set(range(gates + 1)):
        raise ValueError("expected at least two leaves with labels 0 through K-1")
    return tuple(leaves[j] for j in range(len(leaves)))


def _logits(paths, weights, offsets, mean):
    k = len(paths)
    if (
        weights.shape != (k, k - 1, mean.size)
        or offsets.shape != (k, k - 1)
        or mean.ndim != 1
    ):
        raise ValueError("invalid coefficient or drive shape")
    if not all(np.isfinite(x).all() for x in (weights, offsets, mean)):
        raise ValueError("non-finite coefficients or drive")
    logits = weights @ mean + offsets
    if not np.isfinite(logits).all():
        raise ValueError("unrepresentable drive logits")
    return logits


def _path_sums(paths, success, failure):
    scores = np.zeros((len(paths), len(paths)), dtype=success.dtype)
    for destination, path in enumerate(paths):
        for gate, sign in path:
            scores[:, destination] += (success if sign == 1 else failure)[:, gate]
    return scores


def log_transition(paths, weights, offsets, drive):
    """Exact fixed-drive log probabilities, including very unlikely leaves."""
    psi = _logits(paths, weights, offsets, drive)
    return _path_sums(paths, -np.logaddexp(0, -psi), -np.logaddexp(0, psi))


class Bound(NamedTuple):
    log_potential: np.ndarray
    omega: np.ndarray


def jj_bound(paths, weights, offsets, mean, covariance):
    """Evaluate the local PG coordinate for fixed weights and a Gaussian drive."""
    mu = _logits(paths, weights, offsets, mean)
    if (
        covariance.shape != (mean.size, mean.size)
        or not np.isfinite(covariance).all()
        or not np.array_equal(covariance, covariance.T)
    ):
        raise ValueError("invalid drive covariance")
    values, axes = np.linalg.eigh(covariance)
    if np.any(values < 0):
        raise ValueError("unresolved or negative drive covariance eigenvalue")
    projected = (weights @ axes) * np.sqrt(values)
    std = np.hypot.reduce(projected, axis=-1)
    xi = np.hypot(mu, std)
    if not np.isfinite(xi).all():
        raise ValueError("unrepresentable logit second moment")
    # Below sqrt(eps) the quadratic correction to 1/4 rounds away.
    omega = np.full_like(xi, 0.25)
    np.divide(
        0.5 * np.tanh(0.5 * xi),
        xi,
        where=xi > np.sqrt(np.finfo(xi.dtype).eps),
        out=omega,
    )
    log_accept_at_xi = -np.logaddexp(0, -xi)
    # xi - |mu| = std² / (xi + |mu|), even when xi rounds to |mu|.
    relative_std = np.divide(std, xi, where=xi > 0, out=np.zeros_like(xi))
    relative_mu = np.divide(np.abs(mu), xi, where=xi > 0, out=np.zeros_like(xi))
    small_gap = 0.5 * std * relative_std / (1 + relative_mu)
    large_gap = 0.5 * xi + 0.5 * np.abs(mu)
    success = log_accept_at_xi - np.where(mu >= 0, small_gap, large_gap)
    failure = log_accept_at_xi - np.where(mu <= 0, small_gap, large_gap)
    return Bound(_path_sums(paths, success, failure), omega)


def _distribution(p):
    if not np.isfinite(p).all() or np.any(p < 0) or not np.any(p > 0):
        raise ValueError("expected finite nonnegative mass with nonempty support")
    scaled = p / p.max()
    return scaled / scaled.sum()


class Categorical(NamedTuple):
    to_destination: np.ndarray
    to_source: np.ndarray
    pair: np.ndarray


def categorical_messages(log_potential, source, destination):
    """Positive-mass cavity weights. Normalize messages in separate gauges."""
    if log_potential.shape != (source.size, destination.size):
        raise ValueError("cavity and table shapes differ")
    with np.errstate(divide="ignore"):
        left = np.log(_distribution(source))[:, None]
        right = np.log(_distribution(destination))[None, :]
    to_dest = np.logaddexp.reduce(log_potential + left, axis=0)
    to_src = np.logaddexp.reduce(log_potential + right, axis=1)
    joint = left + log_potential + right

    def normalize(logs):
        log_z = np.logaddexp.reduce(logs.ravel())
        if not np.isfinite(log_z):
            raise ValueError("non-finite or empty categorical support")
        return np.exp(logs - log_z)

    return Categorical(normalize(to_dest), normalize(to_src), normalize(joint))


class Statistics(NamedTuple):
    exposure: np.ndarray
    signed: np.ndarray
    drive_eta: np.ndarray
    drive_precision: np.ndarray
    coefficient_eta: np.ndarray
    coefficient_precision: np.ndarray


def statistics(paths, pair, weights, offsets, mean, covariance):
    """Walk each destination path with its own column of source responsibilities."""
    k = len(paths)
    if pair.shape != (k, k):
        raise ValueError("expected a square pair table")
    pair = _distribution(pair)
    bound = jj_bound(paths, weights, offsets, mean, covariance)
    exposure = np.zeros_like(offsets)
    signed = np.zeros_like(offsets)
    for destination, path in enumerate(paths):
        for gate, sign in path:
            exposure[:, gate] += pair[:, destination]
            signed[:, gate] += (sign * 0.5) * pair[:, destination]

    mass = exposure * bound.omega
    drive_eta = np.einsum("ig,igd->d", signed - mass * offsets, weights)
    drive_precision = np.einsum("ig,iga,igb->ab", mass, weights, weights)
    h = np.append(mean, mean.dtype.type(1))
    second = np.outer(h, h)
    second[:-1, :-1] += covariance
    return Statistics(
        exposure,
        signed,
        drive_eta,
        drive_precision,
        signed[:, :, None] * h,
        mass[:, :, None, None] * second,
    )


def pair_energy(pair, log_potential):
    """Categorical part of one recurrence node; 0 log 0 contributes zero."""
    pair = _distribution(pair)
    keep = pair > 0
    return np.sum(pair[keep] * (np.log(pair[keep]) - log_potential[keep]))
