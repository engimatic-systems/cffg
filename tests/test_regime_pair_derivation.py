"""Independent executable anchors for D11; no production binding is changed."""

import numpy as np
import pytest

from cffg.experiments.regime_pair import (
    auxiliary_target,
    learning_statistics,
    local_objective,
    logsumexp,
    omega,
    quadratic_bound,
    solve_region,
    tilted_region,
)
from regime_trajectory_reference import (
    enumerate_paths,
    quadrature_node,
    quadrature_objective,
    switched_prior_reference,
)


def tolerance(dtype, terms=4):
    """Comparison budget declared before fixtures or candidate evaluation."""
    return 256 * terms * np.finfo(dtype).eps


def fixture(dtype=np.float64):
    cast = lambda x: np.asarray(x, dtype=dtype)
    return dict(weights=cast([[[1.3]], [[-0.8]]]), offsets=cast([[-0.4], [0.7]]),
                c=cast([[1.1], [0.9]]), left=cast([-0.3, -0.8]), right=cast([-0.2, -1.2]),
                eta=cast([0.25]), precision=cast([[1.4]]))


def reference_args(args):
    return dict(left=args["left"], right=args["right"], weights=args["weights"][:, 0, 0],
                offsets=args["offsets"][:, 0], c=args["c"][:, 0],
                eta=args["eta"][0], precision=args["precision"][0, 0])


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_all_boundary_messages_moments_and_local_energy_match_quadrature(dtype):
    tol = tolerance(dtype)
    args = fixture(dtype)
    expected = quadrature_node(**reference_args(args))
    actual = tilted_region(**args)
    np.testing.assert_allclose(actual.probability, expected["pair"], atol=tol, rtol=tol)
    np.testing.assert_allclose(actual.probability * actual.mean[..., 0], expected["first"], atol=tol, rtol=tol)
    second = actual.probability * (actual.covariance[..., 0, 0] + actual.mean[..., 0] ** 2)
    np.testing.assert_allclose(second, expected["second"], atol=tol, rtol=tol)
    for field, cavity, axis in (("to_source", args["left"], 1), ("to_destination", args["right"], 0)):
        message = np.log(expected["pair"].sum(axis=axis)) - cavity
        message -= logsumexp(message)
        np.testing.assert_allclose(getattr(actual, field), message, atol=tol, rtol=tol)
    mean = expected["first"].sum()
    variance = expected["second"].sum() - mean ** 2
    np.testing.assert_allclose(actual.to_drive_eta, [mean / variance - args["eta"][0]], atol=tol, rtol=tol)
    np.testing.assert_allclose(actual.to_drive_precision, [[1 / variance - args["precision"][0, 0]]], atol=tol, rtol=tol)
    expected_energy = quadrature_objective(actual.probability, actual.mean[..., 0], actual.covariance[..., 0, 0], **reference_args(args))
    assert actual.free_energy == pytest.approx(expected_energy, abs=tol)
    assert actual.free_energy == pytest.approx(-expected["log_normalizer"], abs=tol)
    assert actual.probability.dtype == np.dtype(dtype)
    assert actual.mean.dtype == np.dtype(dtype)


def test_literal_logistic_reference_exposes_lost_d9_and_source_only_moments():
    args = fixture()
    literal_args = reference_args(args)
    del literal_args["c"]
    literal = quadrature_node(**literal_args)
    pair, first = literal["pair"], literal["first"]
    unconditional = first.sum()
    assert np.max(abs(first - pair * unconditional)) > 0.05
    conditional = first / pair
    assert np.min(abs(conditional[:, 0] - conditional[:, 1])) > 0.2
    actual = tilted_region(**args)
    assert np.max(abs(actual.probability * actual.mean[..., 0] - actual.probability * actual.drive_mean[0])) > 0.05
    # The approximation gap is measured; a non-Gaussian logistic tilt is not exact.
    assert np.max(abs(actual.probability - pair)) < 0.01
    assert actual.free_energy >= -literal["log_normalizer"]
    exact_energy = quadrature_objective(actual.probability, actual.mean[..., 0], actual.covariance[..., 0, 0], **reference_args(args), literal=True)
    assert actual.free_energy >= exact_energy >= -literal["log_normalizer"]


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_drive_independent_reduction_is_d9_exact(dtype):
    tol = tolerance(dtype)
    args = fixture(dtype)
    args["weights"][:] = 0
    args["c"] = abs(args["offsets"])
    actual = tilted_region(**args)
    exact_args = reference_args(args)
    del exact_args["c"]
    exact = quadrature_node(**exact_args)
    np.testing.assert_allclose(actual.probability, exact["pair"], atol=tol, rtol=tol)
    np.testing.assert_allclose(actual.mean[..., 0], args["eta"][0] / args["precision"][0, 0], atol=tol, rtol=tol)
    np.testing.assert_allclose(actual.to_drive_eta, 0, atol=tol)
    np.testing.assert_allclose(actual.to_drive_precision, 0, atol=tol)
    first, second = learning_statistics(actual)
    h = np.append(actual.drive_mean, 1)
    moment = np.outer(h, h)
    moment[:-1, :-1] += actual.drive_covariance
    exposure = actual.probability.sum(axis=1)
    signed = actual.probability[:, 0] - exposure / 2
    np.testing.assert_allclose(first[:, 0], signed[:, None] * h, atol=tol, rtol=tol)
    np.testing.assert_allclose(second[:, 0], exposure[:, None, None] * omega(args["c"])[:, :, None] * moment, atol=tol, rtol=tol)


def test_source_homogeneity_alone_does_not_imply_d9_reduction():
    args = fixture()
    args["weights"][:] = 1.3
    args["offsets"][:] = 0
    actual = tilted_region(**args)
    assert abs(actual.mean[0, 0, 0] - actual.mean[0, 1, 0]) > 0.5


def test_fixed_drive_and_multistick_learning_contract():
    weights = np.arange(12, dtype=np.float64).reshape(3, 2, 2) / 7 - 0.7
    offsets = np.arange(6, dtype=np.float64).reshape(3, 2) / 5 - 0.4
    x = np.array([0.3, -0.6])
    psi = weights @ x + offsets
    h, precision, constant = quadratic_bound(weights, offsets, abs(psi))
    actual = constant + h @ x - np.einsum("a,ijab,b->ij", x, precision, x) / 2
    expected = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            expected[i, j] = -np.logaddexp(0, psi[i, :j]).sum()
            if j < 2:
                expected[i, j] -= np.logaddexp(0, -psi[i, j])
    np.testing.assert_allclose(actual, expected, atol=tolerance(np.float64), rtol=0)
    region = tilted_region(weights, offsets, np.ones((3, 2)), np.zeros(3),
                           np.array([0, -np.inf, -0.2]), np.zeros(2), np.eye(2))
    first, second = learning_statistics(region)
    target = auxiliary_target(region, weights, offsets)
    for i in range(3):
        for stick in range(2):
            expected_first, expected_second, exposure, square = np.zeros(3), np.zeros((3, 3)), 0.0, 0.0
            for j in range(stick, 3):
                p, m, s = region.probability[i, j], region.mean[i, j], region.covariance[i, j]
                feature = np.append(m, 1)
                moment = np.outer(feature, feature)
                moment[:2, :2] += s
                expected_first += (float(j == stick) - 0.5) * p * feature
                expected_second += p * moment
                theta = np.append(weights[i, stick], offsets[i, stick])
                square += p * (theta @ moment @ theta)
                exposure += p
            np.testing.assert_allclose(first[i, stick], expected_first, atol=tolerance(np.float64))
            np.testing.assert_allclose(second[i, stick], omega(region.c)[i, stick] * expected_second, atol=tolerance(np.float64))
            assert target[i, stick] ** 2 == pytest.approx(square / exposure, abs=tolerance(np.float64))


def _chain(unary, weights, offsets, c, eta, diagonal):
    """Compose fixed-c regions with independent Gaussian drives on a chain."""
    time, card = unary.shape
    zeros = np.zeros(card, dtype=unary.dtype)
    args = [dict(weights=weights[t], offsets=offsets[t], c=c[t], left=zeros,
                 right=zeros, eta=eta[t:t + 1], precision=diagonal[t:t + 1, None])
            for t in range(time - 1)]
    regions = [tilted_region(**a) for a in args]
    scores = [np.log(r.probability) + r.log_normalizer for r in regions]
    forward, backward = np.zeros_like(unary), np.zeros_like(unary)
    forward[0] = unary[0]
    for t in range(time - 1):
        forward[t + 1] = unary[t + 1] + logsumexp(forward[t, :, None] + scores[t], axis=0)
    for t in range(time - 2, -1, -1):
        backward[t] = logsumexp(scores[t] + unary[t + 1] + backward[t + 1], axis=1)
    regions = [tilted_region(**{**a, "left": forward[t], "right": unary[t + 1] + backward[t + 1]}) for t, a in enumerate(args)]
    single = np.exp(forward + backward - logsumexp(forward[-1]))
    energy = sum(local_objective(r.probability, r.mean, r.covariance, **a) for r, a in zip(regions, args))
    energy -= np.sum(single * unary)
    energy -= np.sum(single[1:-1] * np.log(single[1:-1]))
    return regions, energy, float(logsumexp(forward[-1]))


def test_complete_chain_region_objective_and_moments_match_path_enumeration():
    time = 5
    a = fixture()
    unary = np.array([[-0.3, -0.8], [0.2, -0.4], [-0.1, 0.3], [0.4, -0.2], [-0.3, 0.1]])
    weights = np.broadcast_to(a["weights"], (time - 1, 2, 1, 1))
    offsets = np.broadcast_to(a["offsets"], (time - 1, 2, 1))
    c = np.broadcast_to(a["c"], (time - 1, 2, 1))
    eta, diagonal = np.full(time - 1, 0.25), np.full(time - 1, 1.4)
    tol = tolerance(np.float64, time)
    expected = enumerate_paths(unary, weights[..., 0, 0], offsets[..., 0], c[..., 0], eta, np.diag(diagonal))
    regions, energy, normalizer = _chain(unary, weights, offsets, c, eta, diagonal)
    assert energy == pytest.approx(-expected["log_normalizer"], abs=tol)
    assert normalizer == pytest.approx(expected["log_normalizer"], abs=tol)
    for t, region in enumerate(regions):
        np.testing.assert_allclose(region.probability, expected["pair"][t], atol=tol, rtol=tol)
        np.testing.assert_allclose(region.probability * region.mean[..., 0], expected["first"][t], atol=tol, rtol=tol)
        np.testing.assert_allclose(region.probability * (region.covariance[..., 0, 0] + region.mean[..., 0] ** 2), expected["second"][t], atol=tol, rtol=tol)


def _coordinate_objective(vector, args, quadrature=False):
    logits = np.append(vector[:3], 0)
    p = np.exp(logits - logsumexp(logits)).reshape(2, 2)
    m = vector[3:7].reshape(2, 2)
    variance = np.exp(vector[7:11]).reshape(2, 2)
    changed = {**args, "c": np.exp(vector[11:]).reshape(2, 1)}
    if quadrature:
        return quadrature_objective(p, m, variance, **reference_args(changed))
    return local_objective(p, m[..., None], variance[..., None, None], **changed)


def _gradient(function, vector):
    step = np.finfo(vector.dtype).eps ** (1 / 3)
    basis = np.eye(len(vector)) * step
    return np.array([(function(vector + e) - function(vector - e)) / (2 * step) for e in basis])


def test_complete_objective_gradient_and_stationary_auxiliary_match_quadrature():
    args = fixture()
    solve_args = {k: v for k, v in args.items() if k != "c"}
    region = solve_region(**solve_args)
    assert region.status == "converged"
    args["c"] = region.c
    logits = np.log(region.probability.ravel())
    vector = np.concatenate((logits[:3] - logits[-1], region.mean.ravel(),
                             np.log(region.covariance.ravel()), np.log(region.c.ravel())))
    # Central-difference roundoff/truncation is O(eps^(2/3)), independent of solver tolerance.
    tol = 2048 * np.finfo(np.float64).eps ** (2 / 3)
    for perturbation in (0, 0.07):
        trial = vector + perturbation * np.linspace(-1, 1, len(vector))
        analytic = _gradient(lambda v: _coordinate_objective(v, args), trial)
        numerical = _gradient(lambda v: _coordinate_objective(v, args, quadrature=True), trial)
        np.testing.assert_allclose(analytic, numerical, atol=tol, rtol=tol)
        if perturbation == 0:
            np.testing.assert_allclose(analytic, 0, atol=tol)
        else:
            assert np.max(abs(analytic)) > 0.01
    np.testing.assert_allclose(auxiliary_target(region, args["weights"], args["offsets"]), region.c, atol=tol)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_required_signed_gaussian_message_and_source_statistics(dtype):
    cast = lambda x: np.asarray(x, dtype=dtype)
    args = dict(weights=cast([[[4]], [[-4]]]), offsets=cast([[-4], [-4]]),
                left=cast([0, 0]), right=cast([0, -np.inf]), eta=cast([0]), precision=cast([[1]]))
    region = solve_region(**args)
    assert region.status == "converged"
    assert region.to_drive_precision[0, 0] < -0.4
    assert region.drive_covariance[0, 0] > 1.8
    np.testing.assert_array_equal(region.probability[:, 1], 0)
    first, second = learning_statistics(region)
    assert first[0, 0, 0] > 0.3 and first[1, 0, 0] < -0.3
    assert np.linalg.eigvalsh(second).min() > 0


def test_literal_zero_support_and_unexposed_auxiliaries_stay_zero():
    args = fixture()
    args["left"][1] = -np.inf
    region = solve_region(**{k: v for k, v in args.items() if k != "c"})
    assert region.status == "converged"
    first, second = learning_statistics(region)
    for value in (region.probability, first, second, region.c):
        np.testing.assert_array_equal(value[1], 0)
    assert np.isneginf(region.to_source[1])


def test_existing_mixture_collapse_requires_a_distinct_ec_binding():
    from scipy.integrate import quad
    from scipy.stats import norm

    def tilted(x):
        return (norm.pdf(x, -2, 1) + norm.pdf(x, 2, 1)) * norm.pdf(x) / 2

    z = quad(tilted, -np.inf, np.inf, epsabs=2e-11, epsrel=2e-11)[0]
    second = quad(lambda x: x * x * tilted(x), -np.inf, np.inf, epsabs=2e-11, epsrel=2e-11)[0] / z
    assert second == pytest.approx(1.5, abs=tolerance(np.float64))
    assert 1 / second - 1 == pytest.approx(-1 / 3, abs=tolerance(np.float64))
    collapse_first_variance = 1 / (1 / (1 + 2 ** 2) + 1)
    assert second - collapse_first_variance == pytest.approx(2 / 3, abs=tolerance(np.float64))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_separate_boundaries_erase_external_source_association(dtype):
    """Decisive falsifier: opposite learning signals have identical cavities."""
    tol = tolerance(dtype)
    positive = switched_prior_reference([2, -2])
    negative = switched_prior_reference([-2, 2])
    np.testing.assert_allclose(positive["pair"], negative["pair"], atol=tol)
    for reference in (positive, negative):
        assert reference["first"].sum() == pytest.approx(0, abs=tol)
        assert reference["second"].sum() == pytest.approx(5, abs=tol)
    # Both external models expose exactly these ordinary singleton cavities.
    cast = lambda x: np.asarray(x, dtype=dtype)
    region = tilted_region(
        cast([[[0]], [[0]]]), cast([[0], [0]]), cast([[0], [0]]),
        cast([0, 0]), cast([0, -np.inf]), cast([0]), cast([[1 / 5]]),
    )
    np.testing.assert_allclose(region.probability, positive["pair"], atol=tol)
    actual_first = region.probability * region.mean[..., 0]
    np.testing.assert_array_equal(actual_first, 0)
    np.testing.assert_allclose(positive["first"][:, 0], [1, -1], atol=tol)
    np.testing.assert_allclose(negative["first"], -positive["first"], atol=tol)
    # D11 local stationarity and exact JJ do not fix this information loss.
    np.testing.assert_allclose(auxiliary_target(region, cast([[[0]], [[0]]]), cast([[0], [0]])), 0, atol=tol)
    learning_first, _ = learning_statistics(region)
    np.testing.assert_array_equal(learning_first[..., 0], 0)
    np.testing.assert_allclose(positive["first"][:, 0] / 2, [0.5, -0.5], atol=tol)


def test_separate_boundaries_also_erase_source_conditioned_second_moments():
    tol = tolerance(np.float64)
    left = switched_prior_reference([0, 0], [1, 9])
    right = switched_prior_reference([0, 0], [9, 1])
    for reference in (left, right):
        assert reference["first"].sum() == pytest.approx(0, abs=tol)
        assert reference["second"].sum() == pytest.approx(5, abs=tol)
    args = fixture()
    args["weights"][:] = args["offsets"][:] = args["c"][:] = 0
    args["left"][:], args["right"][:] = 0, [0, -np.inf]
    args["eta"][:], args["precision"][:] = 0, 1 / 5
    region = tilted_region(**args)
    _, statistics = learning_statistics(region)
    np.testing.assert_allclose(statistics[:, 0, 0, 0], [0.625, 0.625], atol=tol)
    np.testing.assert_allclose(left["second"][:, 0] / 4, [0.125, 1.125], atol=tol)
    np.testing.assert_allclose(right["second"][:, 0] / 4, [1.125, 0.125], atol=tol)


@pytest.mark.parametrize("replacement,match", (
    ({"left": np.full(2, -np.inf)}, "empty supported"),
    ({"right": np.array([np.nan, 0])}, "categorical cavity"),
    ({"eta": np.array([np.inf])}, "non-finite"),
    ({"c": np.array([[-1], [0]])}, "nonnegative"),
    ({"precision": np.array([[-100.0]])}, "improper"),
    ({"eta": np.zeros(2)}, "shapes"),
))
def test_invalid_inputs_fail_loudly(replacement, match):
    with pytest.raises(ValueError, match=match):
        tilted_region(**{**fixture(), **replacement})


def test_vacuous_cavity_requires_a_proper_tilt_and_budget_is_not_convergence():
    args = fixture()
    args["eta"][:] = 0
    args["precision"][:] = 0
    assert np.isfinite(tilted_region(**args).free_energy)
    args["weights"][:] = 0
    with pytest.raises(ValueError, match="improper"):
        tilted_region(**args)
    result = solve_region(**{k: v for k, v in fixture().items() if k != "c"}, max_iterations=1)
    assert result.status == "budget_exhausted"


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_horizon_100_and_multivariate_regimes_use_only_local_arrays(dtype):
    time, card = 100, 8
    unary = np.zeros((time, card), dtype=dtype)
    weights = np.full((time - 1, card, card - 1, 1), 0.3, dtype=dtype)
    offsets, c = np.zeros(weights.shape[:-1], dtype=dtype), np.ones(weights.shape[:-1], dtype=dtype)
    regions, energy, normalizer = _chain(unary, weights, offsets, c,
                                        np.zeros(time - 1, dtype=dtype), np.ones(time - 1, dtype=dtype))
    tol = tolerance(dtype, time * card)
    assert np.isfinite(energy)
    assert energy == pytest.approx(-normalizer, abs=tol)
    assert len(regions) == time - 1
    assert all(r.covariance.shape == (card, card, 1, 1) for r in regions)
    # A separate d=10 fixture exercises dense local algebra without trajectory covariance.
    a = fixture(dtype)
    a["weights"] = np.broadcast_to(a["weights"], (2, 1, 10)).copy()
    a["eta"], a["precision"] = np.zeros(10, dtype=dtype), np.eye(10, dtype=dtype)
    result = tilted_region(**a)
    assert result.covariance.shape == (2, 2, 10, 10)
    with pytest.raises(ValueError, match="restricted"):
        enumerate_paths(unary, None, None, None, None, None)
