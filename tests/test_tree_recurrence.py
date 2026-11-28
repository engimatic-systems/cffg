"""Executable tree transition contract, independent of the production recurrence rules."""

from itertools import permutations

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from tree_recurrence_oracle import (
    categorical_messages,
    jj_bound,
    leaf_paths,
    log_transition,
    pair_energy,
    statistics,
)


def trees(labels):
    if len(labels) == 1:
        yield labels[0]
    for split in range(1, len(labels)):
        for accept in trees(labels[:split]):
            for reject in trees(labels[split:]):
                yield accept, reject


def branch_probabilities(tree, gates):
    """Split probability mass from the root; internal row numbers follow preorder."""
    result = np.zeros((len(gates), len(gates)), dtype=gates.dtype)
    branches = iter(gates.T)

    def split(node, incoming):
        if isinstance(node, int):
            result[:, node] = incoming
        else:
            p = next(branches)
            split(node[0], incoming * p)
            split(node[1], incoming * (1 - p))

    split(tree, np.ones(len(gates), dtype=gates.dtype))
    return result


def chain(labels):
    return labels[0] if len(labels) == 1 else (labels[0], chain(labels[1:]))


def assert_close(actual, expected):
    tol = 32 * np.finfo(actual.dtype).eps
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_labeled_small_trees_and_prefix_statistics(dtype):
    rng = np.random.default_rng(1449)
    for k in (2, 3, 4):
        weights = rng.normal(size=(k, k - 1, 2)).astype(dtype)
        offsets = rng.normal(size=(k, k - 1)).astype(dtype)
        drive = np.array([0.4, -1.2], dtype=dtype)
        cov = np.array([[0.8, -0.2], [-0.2, 0.9]], dtype=dtype)
        psi = weights @ drive + offsets
        p = 1 / (1 + np.exp(-psi))
        for labels in permutations(range(k)):
            for tree in trees(labels):
                paths = leaf_paths(tree)
                actual = np.exp(log_transition(paths, weights, offsets, drive))
                assert actual.dtype == dtype
                assert_close(actual, branch_probabilities(tree, p))
                assert_close(actual.sum(axis=1), np.ones(k))
        # D8's independent singleton product and closed-form prefix coordinates.
        alpha = np.arange(1, k + 1, dtype=dtype)
        alpha /= alpha.sum()
        beta = alpha[::-1]
        paths = leaf_paths(chain(tuple(range(k))))
        bound = jj_bound(paths, weights, offsets, drive, cov)
        stats = statistics(paths, np.outer(alpha, beta), weights, offsets, drive, cov)
        risk = np.cumsum(beta[::-1])[::-1][:-1]
        signed = beta[:-1] - risk / 2
        assert_close(stats.exposure, np.outer(alpha, risk))
        assert_close(stats.signed, np.outer(alpha, signed))
        xi = np.sqrt(psi**2 + np.einsum("igx,xy,igy->ig", weights, cov, weights))
        omega = np.tanh(xi / 2) / (2 * xi)
        s = -np.logaddexp(0, -xi) + (psi - xi) / 2
        f = s - psi
        prefix = np.pad(np.cumsum(f, axis=1), ((0, 0), (1, 0)))
        prefix[:, :-1] += s
        assert_close(bound.log_potential, prefix)
        assert_close(bound.omega, omega)
        mass = alpha[:, None] * risk * omega
        assert_close(
            stats.drive_eta,
            ((alpha[:, None] * signed - mass * offsets)[:, :, None] * weights).sum(
                axis=(0, 1)
            ),
        )
        assert_close(
            stats.drive_precision,
            weights.reshape(-1, 2).T @ (mass[:, :, None] * weights).reshape(-1, 2),
        )


def test_pair_cavities_zero_mass_and_chain_entropy():
    first = leaf_paths(((2, 0), 1))
    second = leaf_paths((1, (2, 0)))
    weights = np.arange(6.0).reshape(3, 2, 1) / 4
    offsets = np.array([[0.5, -1.0], [-1.5, 2.0], [0.8, 0.1]])
    x = np.array([0.7])
    cov = np.zeros((1, 1))
    log1 = jj_bound(first, weights, offsets, x, cov).log_potential
    log2 = jj_bound(second, weights, -offsets, x, cov).log_potential
    left = np.array([0.25, 0.0, 0.75])
    right = np.array([0.0, 0.95, 0.05])
    p1 = branch_probabilities(((2, 0), 1), 1 / (1 + np.exp(-(weights @ x + offsets))))
    p2 = branch_probabilities((1, (2, 0)), 1 / (1 + np.exp(-(weights @ x - offsets))))
    joint = left[:, None, None] * p1[:, :, None] * p2[None, :, :] * right[None, None, :]
    z = joint.sum()
    joint /= z
    a = categorical_messages(log1, left, p2 @ right)
    b = categorical_messages(log2, left @ p1, right)
    assert_close(a.to_destination, left @ p1)
    assert_close(b.to_source, (p2 @ right) / (p2 @ right).sum())
    assert_close(a.pair, joint.sum(axis=2))
    assert_close(b.pair, joint.sum(axis=0))
    assert np.all(a.pair[1] == 0) and np.all(b.pair[:, 0] == 0)
    mid = joint.sum(axis=(0, 2))
    endpoint = -(
        joint.sum(axis=(1, 2))[left > 0] @ np.log(left[left > 0])
        + joint.sum(axis=(0, 1))[right > 0] @ np.log(right[right > 0])
    )
    ledger = (
        pair_energy(a.pair, log1)
        + pair_energy(b.pair, log2)
        - mid @ np.log(mid)
        + endpoint
    )
    assert_close(ledger, -np.log(z))
    # Correlated joint, with mass only on paths to labels 2 and 0.
    rho = np.array([[0.0, 0.0, 0.7], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    stat = statistics(first, rho, weights, offsets, x, cov)
    assert_close(stat.exposure, [[0.7, 0.7], [0.3, 0.3], [0.0, 0.0]])
    assert_close(stat.signed, [[0.35, 0.35], [0.15, -0.15], [0.0, 0.0]])
    for values in (
        stat.exposure,
        stat.signed,
        stat.coefficient_eta,
        stat.coefficient_precision,
    ):
        assert np.all(values[2] == 0)
    rho[:] = 0
    rho[0, 1] = 1  # Root rejects; the entire accept subtree is unvisited.
    stat = statistics(first, rho, weights, offsets, x, cov)
    assert np.all(stat.coefficient_precision[:, 1] == 0) and np.all(
        stat.coefficient_eta[:, 1] == 0
    )


def test_uncertain_drive_bound_and_tied_owner_curvature():
    tree = ((2, 0), (3, 1))
    paths = leaf_paths(tree)
    x, cov = np.array([0.4]), np.array([[0.7]])
    weights = np.linspace(-1.5, 1.3, 12).reshape(4, 3, 1)
    offsets = np.linspace(0.3, -0.6, 12).reshape(4, 3)
    # One owner is used by source 0/root and source 2/right-subtree gate.
    uses = [(0, 0), (2, 2)]
    weights[2, 2], offsets[2, 2] = weights[0, 0], offsets[0, 0]
    bound = jj_bound(paths, weights, offsets, x, cov)
    nodes, quadrature = hermgauss(64)
    samples = x[0] + np.sqrt(2 * cov[0, 0]) * nodes
    quadrature /= np.sqrt(np.pi)
    exact = np.zeros((4, 4))
    for value, mass in zip(samples, quadrature):
        exact += mass * log_transition(paths, weights, offsets, np.array([value]))
    assert np.all(bound.log_potential <= exact + 32 * np.finfo(float).eps)
    assert (exact - bound.log_potential).max() > 0.001
    assert np.all(np.exp(bound.log_potential).sum(axis=1) < 1)
    pair = categorical_messages(
        bound.log_potential,
        np.array([2.0, 1.0, 3.0, 0.0]),
        np.array([1.0, 3.0, 1.0, 2.0]),
    ).pair
    stat = statistics(paths, pair, weights, offsets, x, cov)
    eta = sum(stat.coefficient_eta[i, g] for i, g in uses)
    precision = sum(stat.coefficient_precision[i, g] for i, g in uses)
    # Integrate the fixed-auxiliary quadratic one reached branch at a time.
    # Differences cancel the auxiliary normalizer. Include the Gaussian prior once.
    prior = np.array([[1.3, 0.2], [0.2, 2.0]])
    for theta in ([0.0, 1.0], [-0.6, 1.2], [0.3, -2.1]):
        theta = np.array(theta)
        psi = theta[0] * samples + theta[1]
        expected = -0.5 * theta @ prior @ theta
        for i, g in uses:
            for j, path in enumerate(paths):
                for gate, sign in path:
                    if gate == g:
                        expected += pair[i, j] * (
                            quadrature
                            @ (sign * psi / 2 - bound.omega[i, g] * psi**2 / 2)
                        )
        assert_close(eta @ theta - 0.5 * theta @ (prior + precision) @ theta, expected)


def test_balanced_k32_d16_in_float32():
    def balanced(labels):
        if len(labels) == 1:
            return labels[0]
        half = len(labels) // 2
        return balanced(labels[:half]), balanced(labels[half:])

    k, d = 32, 16
    paths = leaf_paths(balanced(list(range(k))))
    assert {len(path) for path in paths} == {5}
    rng = np.random.default_rng(612)
    weights = rng.normal(scale=0.1, size=(k, k - 1, d)).astype(np.float32)
    offsets = rng.normal(size=(k, k - 1)).astype(np.float32)
    x, cov = np.zeros(d, np.float32), np.eye(d, dtype=np.float32)
    bound = jj_bound(paths, weights, offsets, x, cov)
    p = categorical_messages(
        bound.log_potential, np.ones(k, np.float32), np.ones(k, np.float32)
    )
    stat = statistics(paths, p.pair, weights, offsets, x, cov)
    for array in (*bound, *p, *stat):
        assert array.dtype == np.float32 and np.isfinite(array).all()
    assert_close(stat.exposure.sum(), 5.0)
    assert_close(stat.exposure[:, 0], p.pair.sum(axis=1))
    assert_close(stat.coefficient_precision[:, :, -1, -1], stat.exposure * bound.omega)


def test_zero_logit_and_invalid_inputs():
    paths = leaf_paths((0, 1))
    weights = np.zeros((2, 1, 1), np.float32)
    offsets = np.zeros((2, 1), np.float32)
    mean, cov = np.zeros(1, np.float32), np.eye(1, dtype=np.float32)
    assert np.all(jj_bound(paths, weights, offsets, mean, cov).omega == 0.25)
    subnormal = np.nextafter(np.float32(0), np.float32(1))
    assert np.all(
        jj_bound(paths, weights, offsets + subnormal, mean, cov).omega == 0.25
    )
    # Uncertain likely branches retain the JJ penalty even if xi and |mu| round alike.
    large_offsets = np.array([[1e4], [-1e4]], np.float32)
    likely = jj_bound(paths, weights + 1, large_offsets, mean, cov).log_potential
    gap = 0.5 / (np.hypot(1e4, 1.0) + 1e4)
    np.testing.assert_allclose(
        [likely[0, 0], likely[1, 1]], -gap, rtol=4 * np.finfo(np.float32).eps
    )
    for tree in (0, (0, 0), (0, 2), (0, 1, 2), ((0,), 1)):
        with pytest.raises(ValueError):
            leaf_paths(tree)
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="non-finite"):
            log_transition(paths, weights, offsets + invalid, mean)
    with pytest.raises(ValueError, match="negative"):
        jj_bound(paths, weights, offsets, mean, -cov)
    with pytest.raises(ValueError, match="support"):
        categorical_messages(np.zeros((2, 2)), np.ones(2), np.zeros(2))
    with pytest.raises(ValueError, match="support"):
        statistics(paths, np.zeros((2, 2)), weights, offsets, mean, cov)


def test_decision_list_matches_current_d9_pair_and_drive_rule():
    import jax

    from cffg.structured_recurrence import (
        structured_gaussian_message,
        structured_pair_belief,
        structured_pair_potential,
    )

    k = 4
    paths = leaf_paths(chain(tuple(range(k))))
    rng = np.random.default_rng(704)
    weights = rng.normal(size=(k, k - 1, 2))
    offsets = rng.normal(size=(k, k - 1))
    x, cov = np.array([-0.4, 0.3]), np.array([[1.0, 0.4], [0.4, 0.7]])
    left, right = np.array([0.5, 0.0, 0.2, 0.3]), np.array([0.0, 0.6, 0.3, 0.1])
    b = jj_bound(paths, weights, offsets, x, cov)
    p = categorical_messages(b.log_potential, left, right)
    stat = statistics(paths, p.pair, weights, offsets, x, cov)

    @jax.jit
    def current_rule(x, cov, weights, offsets, left_log, right_log):
        native = structured_pair_potential(x, cov, weights, offsets)
        actual = structured_pair_belief(left_log, right_log, native.log_potential)
        drive = structured_gaussian_message(
            actual.probability, native.omega, weights, offsets
        )
        return native, actual, drive

    with np.errstate(divide="ignore"):
        native, actual, drive = current_rule(
            x, cov, weights, offsets, np.log(left), np.log(right)
        )
    assert native.proper and actual.proper
    assert_close(b.log_potential, native.log_potential)
    assert_close(p.pair, actual.probability)
    assert drive.proper
    assert_close(stat.exposure, drive.exposure)
    assert_close(stat.signed, drive.signed_success)
    assert_close(stat.drive_eta, drive.eta)
    assert_close(stat.drive_precision, drive.precision)
