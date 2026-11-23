"""D12 derivation anchors; no production dispatch is introduced."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cffg.experiments.hybrid_separator import (
    Belief, Natural, canonical, chain_energy, evaluate, initial_messages, lift,
    marginal, normalize, quotient, solve, source_region, transition,
)
from cffg.experiments.regime_pair import auxiliary_target, learning_statistics
from hybrid_separator_reference import conditional_moments, enumerate_chain, recurrence_increments, reflected_mixture_gap
from regime_trajectory_reference import switched_prior_reference


def tolerance(dtype, terms=4):
    """Floating-point equality budget set before evaluating the candidate."""
    return 512 * terms * np.finfo(dtype).eps


def fixture(time=3, dtype=np.float64, *, independent=False):
    cast = lambda x: np.asarray(x, dtype=dtype)
    weights = np.tile(cast([0.35, -0.25]), (time - 1, 1))
    offsets = np.tile(cast([-0.2, 0.3]), (time - 1, 1))
    c = np.ones_like(offsets)
    matrix = np.tile(cast([0.65, 0.85]), (time - 1, 1))
    bias = np.tile(cast([0.1, -0.15]), (time - 1, 1))
    noise = np.tile(cast([0.8, 1.1]), (time - 1, 1))
    constant = np.tile(cast([-0.2, -0.4]), (time, 1))
    linear = np.tile(cast([0.2, -0.3]), (time, 1))
    precision = np.tile(cast([0.9, 1.2]), (time, 1))
    if independent:
        weights[:] = 0
        c = abs(offsets)
        for value in (matrix, bias, noise, linear, precision):
            value[:, 1] = value[:, 0]
    return dict(site_constant=constant, site_linear=linear, site_precision=precision,
                weights=weights, offsets=offsets, c=c, matrix=matrix, bias=bias, noise=noise)


def build(args):
    sites = [Natural(a.copy(), h[:, None].copy(), j[:, None, None].copy())
             for a, h, j in zip(args["site_constant"], args["site_linear"], args["site_precision"])]
    factors = [transition(r[:, None, None], b[:, None], c[:, None], a[:, None, None], d[:, None], q[:, None, None])
               for r, b, c, a, d, q in zip(*(args[k] for k in ("weights", "offsets", "c", "matrix", "bias", "noise")))]
    return factors, sites


def weighted(region):
    return region.probability, region.probability[..., None] * region.mean, region.probability[..., None, None] * region.second


def learning(region, c):
    source = source_region(region)
    return learning_statistics(SimpleNamespace(**vars(source), c=c[:, None]))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_one_group_all_moments_messages_objective_and_learning_match_reference(dtype):
    tol = tolerance(dtype)
    args = fixture(2, dtype)
    factors, sites = build(args)
    expected = enumerate_chain(**args)
    result = solve(factors, sites)
    assert result.status == "converged"
    for value, key in zip(weighted(result.regions[0]), ("pair", "first", "second")):
        np.testing.assert_allclose(value, expected[key][0], atol=tol, rtol=tol)
    for side in (0, 1):
        p, m, v = conditional_moments(*(expected[k][0] for k in ("pair", "first", "second")), side)
        message = result.messages[0][side]
        j = 1 / v - sites[side].precision[:, 0, 0]
        h = m / v - sites[side].linear[:, 0]
        a = np.log(p) - (np.log(2 * np.pi * v) + m * m / v) / 2 - sites[side].constant
        a -= a[0]
        for actual, truth in ((message.constant, a), (message.linear[:, 0], h), (message.precision[:, 0, 0], j)):
            np.testing.assert_allclose(actual, truth, atol=tol, rtol=tol)
    assert result.free_energy == pytest.approx(expected["free_energy"], abs=tol)
    assert result.dual_energy == pytest.approx(result.free_energy, abs=tol)
    actual = learning(result.regions[0], args["c"][0])
    expected_learning = recurrence_increments(*(expected[k][0] for k in ("pair", "first", "second")), args["c"][0])
    for value, truth in zip(actual, expected_learning):
        np.testing.assert_allclose(value[:, 0], truth, atol=tol, rtol=tol)
    assert result.regions[0].mean.dtype == np.dtype(dtype)


@pytest.mark.parametrize("means,variances", (((2, -2), (1, 1)), ((-2, 2), (1, 1)), ((0, 0), (1, 9)), ((0, 0), (9, 1))))
def test_d11_twins_cross_hybrid_boundary_with_opposite_learning(means, variances):
    tol = tolerance(np.float64)
    args = fixture(2)
    args["weights"][:] = args["offsets"][:] = args["c"][:] = 0
    means, variances = np.array(means), np.array(variances)
    args["site_constant"][0] = np.log(0.5) - (np.log(2 * np.pi * variances) + means * means / variances) / 2
    args["site_linear"][0] = means / variances
    args["site_precision"][0] = 1 / variances
    args["site_constant"][1] = [0, -np.inf]
    args["site_linear"][1] = 0
    args["site_precision"][1] = 0
    factors, sites = build(args)
    # Vacuous endpoint cavities are valid when the joint tilted region is proper.
    region = normalize(factors[0] + lift(*sites))
    expected = switched_prior_reference(means, variances)
    for actual, key in zip(weighted(source_region(region)), ("pair", "first", "second")):
        np.testing.assert_allclose(actual.squeeze(), expected[key], atol=tol, rtol=tol)
    linear, precision = learning(region, args["c"][0])
    np.testing.assert_allclose(linear[:, 0, 0], means / 4, atol=tol)
    np.testing.assert_allclose(precision[:, 0, 0, 0], (variances + means ** 2) / 8, atol=tol)
    assert region.probability[:, 1].tolist() == [0, 0]


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_temporal_decoupled_reduction_matches_exact_chain(dtype):
    tol = tolerance(dtype, 20)
    args = fixture(4, dtype, independent=True)
    factors, sites = build(args)
    expected = enumerate_chain(**args)
    actual = solve(factors, sites)
    assert actual.status == "converged"
    assert actual.free_energy == pytest.approx(expected["free_energy"], abs=tol)
    for t, region in enumerate(actual.regions):
        for value, key in zip(weighted(region), ("pair", "first", "second")):
            np.testing.assert_allclose(value, expected[key][t], atol=tol, rtol=tol)
        assert np.max(abs(region.covariance[..., 0, 1])) > 0.05


def test_temporal_coupled_case_reports_projection_gap_separately_from_stationarity():
    # Accuracy limits are declared before evaluation; they are not exactness tolerances.
    probability_gap, moment_gap, energy_gap, learning_gap = 0.002, 0.01, 0.002, 0.0001
    args = fixture(3)
    factors, sites = build(args)
    exact = enumerate_chain(**args)
    result = solve(factors, sites)
    assert result.status == "converged"
    errors = []
    for t, region in enumerate(result.regions):
        errors.append([np.max(abs(value - exact[key][t])) for value, key in zip(weighted(region), ("pair", "first", "second"))])
        increments = recurrence_increments(*(exact[k][t] for k in ("pair", "first", "second")), args["c"][t])
        for actual, expected in zip(learning(region, args["c"][t]), increments):
            assert np.max(abs(actual[:, 0] - expected)) < learning_gap
    assert np.max(np.array(errors)[:, 0]) < probability_gap
    assert np.max(np.array(errors)[:, 1:]) < moment_gap
    assert tolerance(np.float64) < abs(result.free_energy - exact["free_energy"]) < energy_gap
    assert result.free_energy == pytest.approx(result.dual_energy, abs=1e-7)
    for t, region in enumerate(result.regions):
        for side in (0, 1):
            projected = marginal(region, side)
            for value, truth in zip(weighted(projected), weighted(result.separators[t + side])):
                np.testing.assert_allclose(value, truth, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("coordinate", ("constant", "linear", "precision"))
def test_dual_gradient_is_incidence_moment_residual(coordinate):
    args = fixture(3)
    factors, sites = build(args)
    messages = initial_messages(sites)
    regions, separators, _, _, _ = evaluate(factors, sites, messages)
    step = np.finfo(np.float64).eps ** (1 / 3)
    # Message from region 0 to the interior separator enters region 1's cavity.
    q, region = separators[1], regions[1]
    if coordinate == "constant":
        expected = q.probability[1] - region.probability[1].sum()
        index = (1,)
    elif coordinate == "linear":
        expected = q.probability[1] * q.mean[1, 0] - np.sum(region.probability[1] * region.mean[1, :, 0])
        index = (1, 0)
    else:
        expected = -(q.probability[1] * q.second[1, 0, 0] - np.sum(region.probability[1] * region.second[1, :, 0, 0])) / 2
        index = (1, 0, 0)
    values = []
    for sign in (-1, 1):
        changed = list(messages)
        msg = messages[0][1]
        value = getattr(msg, coordinate).copy()
        value[index] += sign * step
        changed[0] = (changed[0][0], replace(msg, **{coordinate: value}))
        values.append(evaluate(factors, sites, changed)[4])
    assert (values[1] - values[0]) / (2 * step) == pytest.approx(expected, abs=128 * step ** 2)


def test_hybrid_normalizers_keep_regime_mass_and_signed_quotients():
    belief = Belief(np.array([0.4, 0.6]), np.array([[2.0], [-1.0]]), np.array([[[0.5]], [[3.0]]]), 0)
    restored = normalize(canonical(belief))
    np.testing.assert_allclose(restored.probability, belief.probability)
    cavity = Natural(np.zeros(2), np.zeros((2, 1)), np.ones((2, 1, 1)))
    message = quotient(belief, cavity)
    assert message.precision[1, 0, 0] < 0
    np.testing.assert_allclose(normalize(message + cavity).probability, belief.probability)


def test_inactive_auxiliaries_and_support_rebuild():
    args = fixture(2)
    args["site_constant"][0, 1] = -np.inf
    factors, sites = build(args)
    region = normalize(factors[0] + lift(*sites))
    target = auxiliary_target(source_region(region), args["weights"][0, :, None, None], args["offsets"][0, :, None])
    assert target[1, 0] == 0
    args["site_constant"][0, 1] = 0
    _, changed = build(args)
    expanded = normalize(factors[0] + lift(*changed))
    with pytest.raises(ValueError, match="support changed"):
        quotient(marginal(expanded, 0), sites[0])


@pytest.mark.parametrize("failure", ("empty", "improper", "nonfinite", "shape", "asymmetric"))
def test_hostile_coordinates_fail_loudly(failure):
    natural = Natural(np.zeros(2), np.zeros((2, 2)), np.broadcast_to(np.eye(2), (2, 2, 2)).copy())
    if failure == "empty":
        natural.constant[:] = -np.inf
    elif failure == "improper":
        natural.precision[:] = 0
    elif failure == "nonfinite":
        natural.linear[0, 0] = np.inf
    elif failure == "shape":
        natural = replace(natural, linear=np.zeros((2, 1)))
    else:
        natural.precision[0, 0, 1] = 1
    with pytest.raises(ValueError):
        normalize(natural)


def test_budget_exhaustion_is_not_convergence():
    factors, sites = build(fixture(3))
    assert solve(factors, sites, max_iterations=1).status == "budget_exhausted"


def test_hundred_slice_inventory_has_no_trajectory_covariance():
    factors, sites = build(fixture(100, np.float32))
    regions, separators, _, energy, _ = evaluate(factors, sites, initial_messages(sites))
    assert len(regions) == 99 and len(separators) == 100 and np.isfinite(energy)
    assert all(b.covariance.shape == (2, 2, 2, 2) for b in regions)
    assert sum(b.covariance.size for b in regions) == 99 * 16


def test_reflected_mixtures_expose_finite_family_information_loss():
    mean, probability = reflected_mixture_gap(1)
    reflected_mean, reflected_probability = reflected_mixture_gap(-1)
    assert mean < -0.01
    assert reflected_mean == pytest.approx(-mean)
    np.testing.assert_allclose(reflected_probability, probability[::-1])
    assert abs(probability[0] - 0.5) > 0.019


def test_ordinary_gaussian_adapter_uses_full_tilt_and_signed_quotient():
    from cffg.experiments.hybrid_separator import gaussian_adapter

    prior = canonical(Belief(np.array([0.5, 0.5]), np.array([[2.0], [-2.0]]), np.ones((2, 1, 1)), 0))
    cavity = Natural(np.array(0.0), np.zeros(1), np.ones((1, 1)))
    belief, to_hybrid, to_gaussian = gaussian_adapter(prior, cavity)
    np.testing.assert_allclose(belief.mean[:, 0], [1, -1])
    np.testing.assert_allclose(belief.covariance[:, 0, 0], 0.5)
    np.testing.assert_allclose(to_hybrid.linear, 0, atol=tolerance(np.float64))
    np.testing.assert_allclose(to_hybrid.precision, 1)
    assert to_gaussian.precision[0, 0] == pytest.approx(-1 / 3)
    # Extra hybrid incidence adds +H(q_y); the adapter contributes -H(q_y).
    assert -belief.entropy() + normalize(prior + to_hybrid).entropy() == pytest.approx(0)


def test_auxiliary_xi_gradient_and_false_zero_at_c_zero():
    from cffg.experiments.hybrid_separator import auxiliary_residual

    args = fixture(2)
    args["c"][:] = 0
    factors, sites = build(args)
    region = normalize(factors[0] + lift(*sites))
    r = auxiliary_residual([region], args["weights"][:, :, None, None], args["offsets"][:, :, None], args["c"][:, :, None])
    assert np.max(abs(r)) > 0.02
    # Hold the belief and vary xi; dF/dxi = exposure * residual / 96 at xi=0.
    step = np.finfo(np.float64).eps ** (1 / 3)
    changed = {k: v.copy() for k, v in args.items()}
    changed["c"][0, 0] = np.sqrt(step)
    altered, _ = build(changed)
    derivative = (-region.expectation(altered[0]) + region.expectation(factors[0])) / step
    assert derivative == pytest.approx(region.probability[0].sum() * r[0, 0, 0] / 96, abs=32 * step)


@pytest.mark.parametrize("card,dim", ((20, 1), (2, 10)))
def test_polynomial_regime_and_dimension_inventory(card, dim):
    dtype = np.float32
    weights = np.full((card, card - 1, dim), 0.02, dtype=dtype)
    offsets, c = np.zeros((card, card - 1), dtype=dtype), np.ones((card, card - 1), dtype=dtype)
    matrix = np.broadcast_to(np.eye(dim, dtype=dtype) / 2, (card, dim, dim)).copy()
    noise = np.broadcast_to(np.eye(dim, dtype=dtype), matrix.shape).copy()
    factor = transition(weights, offsets, c, matrix, np.zeros((card, dim), dtype=dtype), noise)
    site = Natural(np.zeros(card, dtype=dtype), np.zeros((card, dim), dtype=dtype), noise)
    region = normalize(factor + lift(site, site))
    assert region.covariance.shape == (card, card, 2 * dim, 2 * dim)
    assert np.isfinite(region.entropy())


def test_signed_cavity_can_be_repaired_by_local_factor():
    factors, sites = build(fixture(2))
    signed = replace(sites[1], precision=np.full((2, 1, 1), -0.01))
    with pytest.raises(ValueError, match="improper"):
        normalize(signed)
    region = normalize(factors[0] + lift(sites[0], signed))
    assert np.isfinite(region.log_normalizer)


def test_alternating_scopes_and_grouped_elimination_have_identical_energy():
    from cffg.experiments.hybrid_separator import collapse_label

    args = fixture(2)
    factors, sites = build(args)
    grouped = normalize(factors[0] + lift(*sites))
    recurrence = source_region(grouped)
    intermediate = collapse_label(recurrence, 0)
    dynamics = collapse_label(grouped, 0)
    # The conditional y|x,j is common across source i. Gaussian projection of
    # x|j therefore commutes with this affine channel at the returned cavity.
    conditional_variance = dynamics.covariance[:, 1, 1] - dynamics.covariance[:, 1, 0] ** 2 / dynamics.covariance[:, 0, 0]
    expected = 1 / (1 / args["noise"][0] + sites[1].precision[:, 0, 0])
    np.testing.assert_allclose(conditional_variance, expected)
    assert grouped.entropy() - recurrence.entropy() == pytest.approx(dynamics.entropy() - intermediate.entropy())
    # Dynamics energy is quadratic, so projection preserves its expectation.
    zero = np.zeros_like(args["weights"][0, :, None, None])
    d_factor = transition(zero, np.zeros((2, 1)), np.zeros((2, 1)),
                          args["matrix"][0, :, None, None], args["bias"][0, :, None], args["noise"][0, :, None, None])
    # Remove the uniform recurrence constant, leaving the normalized channel.
    d_factor.constant[:] += np.log(2)
    channel = Natural(*(x[0] for x in d_factor.arrays()))
    assert grouped.expectation(d_factor) == pytest.approx(dynamics.expectation(channel))


def test_reflected_mixture_is_stationary_even_with_returning_cavities():
    from cffg.experiments.hybrid_separator import collapse_label

    prior_probability = np.exp(1) / (np.exp(1) + 2 * np.exp(0.25))
    messages = []
    for sign in (-1, 1):
        means = sign * np.array([2.0, -1.0])
        a = np.log([prior_probability, 1 - prior_probability]) - (np.log(2 * np.pi) + means ** 2) / 2 - np.log(2)
        left_factor = Natural(np.broadcast_to(a[:, None], (2, 2)).copy(), np.broadcast_to(means[:, None, None], (2, 2, 1)).copy(), np.ones((2, 2, 1, 1)))
        left_tilt = left_factor + Natural(np.zeros((2, 2)), np.zeros((2, 2, 1)), np.ones((2, 2, 1, 1)))
        left = normalize(left_tilt)
        hybrid = collapse_label(left, 0)
        np.testing.assert_allclose(hybrid.probability, 0.5)
        np.testing.assert_allclose(hybrid.mean, 0, atol=tolerance(np.float64))
        np.testing.assert_allclose(hybrid.covariance, 1)
        to_left = Natural(np.zeros(2), np.zeros((2, 1)), np.ones((2, 1, 1)))
        messages.append(quotient(hybrid, to_left))
    for a, b in zip(messages[0].arrays(), messages[1].arrays()):
        np.testing.assert_allclose(a, b, atol=tolerance(np.float64))
    b = np.array([1, -1]) / np.sqrt(2)
    right = Belief(np.full((2, 2), 0.25), np.broadcast_to(b[None, :, None], (2, 2, 1)), np.full((2, 2, 1, 1), 0.5), 0)
    q = collapse_label(right, 1)
    np.testing.assert_allclose(q.mean, 0, atol=tolerance(np.float64))
    np.testing.assert_allclose(q.covariance, 1)
    # Exact integration distinguishes destinations; both stationary EC points do not.
    for sign in (-1, 1):
        exact_mean, exact_destination = reflected_mixture_gap(sign)
        assert abs(exact_mean) > 0.01
        assert abs(exact_destination[0] - right.probability[:, 0].sum()) > 0.019


def test_local_jj_gap_is_separate_from_hybrid_projection_gap():
    # Fixed before evaluation; quadrature refinement certifies the reference.
    quadrature_tolerance, moment_gap, energy_gap = 1e-9, 0.002, 0.01
    args = fixture(2)
    literal = enumerate_chain(**args, literal=True)
    refined = enumerate_chain(**args, literal=True, order=48)
    for key in literal:
        np.testing.assert_allclose(literal[key], refined[key], atol=quadrature_tolerance)
    factors, sites = build(args)
    result = solve(factors, sites)
    for actual, key in zip(weighted(result.regions[0]), ("pair", "first", "second")):
        assert np.max(abs(actual - literal[key][0])) < moment_gap
    assert 0 < result.free_energy - literal["free_energy"] < energy_gap


def test_vacuous_terminal_site_has_a_proper_solved_aggregate():
    args = fixture(3)
    args["site_constant"][1:] = 0
    args["site_linear"][1:] = 0
    args["site_precision"][1:] = 0
    factors, sites = build(args)
    result = solve(factors, sites)
    assert result.status == "converged"
    assert all(np.isfinite(q.covariance).all() for q in result.separators)


def test_ordinary_categorical_adapter_preserves_conditional_state_moments():
    from cffg.experiments.hybrid_separator import categorical_adapter

    prior = canonical(Belief(np.array([0.4, 0.6]), np.array([[2.0], [-1.0]]), np.ones((2, 1, 1)), 0))
    belief, to_hybrid, outgoing = categorical_adapter(prior, np.log(np.array([0.8, 0.2])))
    np.testing.assert_allclose(belief.probability, [8 / 11, 3 / 11])
    np.testing.assert_allclose(belief.mean[:, 0], [2, -1])
    np.testing.assert_allclose(np.exp(outgoing), [0.4, 0.6])
    np.testing.assert_allclose(to_hybrid.linear, 0, atol=tolerance(np.float64))
    np.testing.assert_allclose(to_hybrid.precision, 0, atol=tolerance(np.float64))


def test_temporal_mixture_learning_bias_and_precision_change_sign():
    exact, reflected = (reflected_mixture_gap(sign, full=True) for sign in (1, -1))
    np.testing.assert_allclose(exact["linear"], [0.183824595041, -0.009632252237], atol=1e-11)
    np.testing.assert_allclose(reflected["linear"], exact["linear"] * [1, -1])
    assert reflected["precision"][0, 1] == pytest.approx(-exact["precision"][0, 1])
    assert exact["precision"][0, 0] == pytest.approx(1.029089989129 / 8)


def test_nonfinite_message_residual_cannot_be_reported_as_zero():
    from cffg.experiments.hybrid_separator import message_distance

    _, sites = build(fixture(2))
    messages = initial_messages(sites)
    targets = initial_messages(sites)
    targets[0][0].linear[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        message_distance(messages, targets)
