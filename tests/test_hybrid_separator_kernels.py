"""D12 JAX mathematics against independent Gaussian and logistic references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import hybrid_natural as hn
from cffg import hybrid_separator as hs
from hybrid_kernel_cases import build, case, messages, natural, stack, tolerance, unstack, weighted
from hybrid_separator_reference import conditional_moments, enumerate_chain, recurrence_increments, reflected_mixture_gap
from regime_trajectory_reference import switched_prior_reference


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_isolated_tilt_messages_energy_and_learning_match_independent_reference(dtype):
    tol = tolerance(dtype)
    args = case(2, dtype)
    reference = enumerate_chain(**args)
    factors, sites = map(unstack, build(args))
    tilted = factors[0] + hn.lift(*sites)
    region = jax.jit(hn.normalize)(tilted)
    hn.raise_for_status(region.status)
    for actual, key in zip(weighted(region), ("pair", "first", "second")):
        np.testing.assert_allclose(actual, reference[key][0], atol=tol, rtol=tol)
    energy = -region.entropy() - region.expectation(tilted)
    assert energy == pytest.approx(reference["free_energy"], abs=tol)
    assert energy == pytest.approx(-float(region.log_normalizer), abs=tol)
    for side in (0, 1):
        p, m, v = conditional_moments(*(reference[k][0] for k in ("pair", "first", "second")), side)
        projected = hn.marginal(region, side)
        expected_j = 1 / v - np.asarray(sites[side].precision[:, 0, 0])
        expected_h = m / v - np.asarray(sites[side].linear[:, 0])
        expected_a = np.log(p) - (np.log(2 * np.pi * v) + m * m / v) / 2 - np.asarray(sites[side].constant)
        expected_a -= expected_a[0]
        result = hn.quotient(projected, sites[side])
        for actual, expected in zip(result.arrays(), (expected_a, expected_h[:, None], expected_j[:, None, None])):
            np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
    aux = hs.auxiliary_statistics(region, jnp.asarray(args["weights"][0, :, None, None]),
                                  jnp.asarray(args["offsets"][0, :, None]), jnp.asarray(args["c"][0, :, None] ** 2))
    hn.raise_for_status(aux.status)
    expected = recurrence_increments(*(reference[k][0] for k in ("pair", "first", "second")), args["c"][0])
    for actual, truth in zip((aux.linear[:, 0], aux.precision[:, 0]), expected):
        np.testing.assert_allclose(actual, truth, atol=tol, rtol=tol)


@pytest.mark.parametrize("means,variances", (((2, -2), (1, 1)), ((-2, 2), (1, 1)), ((0, 0), (1, 9)), ((0, 0), (9, 1))))
def test_d11_mean_and_variance_twins_keep_source_association(means, variances):
    means, variances = np.asarray(means), np.asarray(variances)
    args = case(2)
    args["weights"][:] = args["offsets"][:] = args["c"][:] = 0
    args["site_constant"][0] = np.log(0.5) - (np.log(2 * np.pi * variances) + means ** 2 / variances) / 2
    args["site_linear"][0] = means / variances
    args["site_precision"][0] = 1 / variances
    args["site_constant"][1] = [0, -np.inf]
    args["site_linear"][1] = args["site_precision"][1] = 0
    factors, sites = map(unstack, build(args))
    region = hn.normalize(factors[0] + hn.lift(*sites))
    expected = switched_prior_reference(means, variances)
    for actual, key in zip(weighted(hs.source_region(region)), ("pair", "first", "second")):
        np.testing.assert_allclose(actual.squeeze(), expected[key], atol=tolerance(np.float64))
    aux = hs.auxiliary_statistics(region, jnp.zeros((2, 1, 1)), jnp.zeros((2, 1)), jnp.zeros((2, 1)))
    np.testing.assert_allclose(aux.linear[:, 0, 0], means / 4, atol=tolerance(np.float64))
    np.testing.assert_allclose(aux.precision[:, 0, 0, 0], (variances + means ** 2) / 8)
    np.testing.assert_array_equal(region.probability[:, 1], 0)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_pure_chain_evaluation_matches_experiment_without_running_a_jax_solver(dtype):
    from cffg.experiments import hybrid_separator as experiment

    args = case(3, dtype)
    factors, sites = build(args)
    output = jax.jit(hs.evaluate)(factors, sites, messages(sites))
    reference_factors = [experiment.Natural(*(np.asarray(x) for x in n.arrays())) for n in unstack(factors)]
    reference_sites = [experiment.Natural(*(np.asarray(x) for x in n.arrays())) for n in unstack(sites)]
    expected = experiment.evaluate(reference_factors, reference_sites, experiment.initial_messages(reference_sites))
    hn.raise_for_status(output.status)
    for actual, truth in zip(unstack_beliefs(output.regions), expected[0]):
        for value, target in zip(weighted(actual), weighted(truth)):
            np.testing.assert_allclose(value, target, atol=tolerance(dtype), rtol=tolerance(dtype))
    assert output.free_energy == pytest.approx(expected[3], abs=tolerance(dtype))
    assert output.dual_energy == pytest.approx(expected[4], abs=tolerance(dtype))


def unstack_beliefs(value):
    return [jax.tree.map(lambda x: x[t], value) for t in range(value.probability.shape[0])]


@pytest.mark.parametrize("independent", (False, True))
def test_stationary_local_evaluation_preserves_temporal_gap_and_d9_reduction(independent):
    from cffg.experiments import hybrid_separator as experiment

    args = case(3, independent=independent)
    factors, sites = build(args)
    reference_factors = [experiment.Natural(*(np.asarray(x) for x in n.arrays())) for n in unstack(factors)]
    reference_sites = [experiment.Natural(*(np.asarray(x) for x in n.arrays())) for n in unstack(sites)]
    solved = experiment.solve(reference_factors, reference_sites)
    assert solved.status == "converged"
    outgoing = tuple(stack([natural(pair[side]) for pair in solved.messages]) for side in (0, 1))
    result = hs.evaluate(factors, sites, outgoing)
    hn.raise_for_status(result.status)
    assert result.moment_residual < 1e-7
    assert result.quotient_residual < 1e-7
    assert result.free_energy == pytest.approx(float(result.dual_energy), abs=1e-7)
    exact = enumerate_chain(**args)
    if independent:
        assert result.free_energy == pytest.approx(exact["free_energy"], abs=1e-7)
        for value, key in zip(weighted(result.regions), ("pair", "first", "second")):
            np.testing.assert_allclose(value, exact[key], atol=1e-7)
    else:
        assert 1e-6 < abs(result.free_energy - exact["free_energy"]) < 0.002


@pytest.mark.parametrize("coordinate", ("constant", "linear", "precision"))
def test_dual_autodiff_and_independent_difference_equal_incidence_residual(coordinate):
    factors, sites = build(case(3))
    outgoing = messages(sites)
    base = hs.evaluate(factors, sites, outgoing)
    q = jax.tree.map(lambda x: x[1], base.separators)
    b = hn.marginal(jax.tree.map(lambda x: x[1], base.regions), 0)
    expected = {"constant": q.probability[1] - b.probability[1],
                "linear": q.probability[1] * q.mean[1, 0] - b.probability[1] * b.mean[1, 0],
                "precision": -(q.probability[1] * q.second[1, 0, 0] - b.probability[1] * b.second[1, 0, 0]) / 2}[coordinate]
    index = {"constant": (0, 1), "linear": (0, 1, 0), "precision": (0, 1, 0, 0)}[coordinate]

    def dual(shift):
        changed = outgoing[1]._replace(**{coordinate: getattr(outgoing[1], coordinate).at[index].add(shift)})
        return hs.evaluate(factors, sites, (outgoing[0], changed)).dual_energy

    step = np.finfo(np.float64).eps ** (1 / 3)
    assert jax.grad(dual)(0.0) == pytest.approx(float(expected), abs=128 * step ** 2)
    assert (dual(step) - dual(-step)) / (2 * step) == pytest.approx(float(expected), abs=128 * step ** 2)


def test_xi_zero_has_nonzero_energy_gradient_and_complete_auxiliary_residual():
    args = case(2)
    args["c"][:] = 0
    factors, sites = map(unstack, build(args))
    region = hn.normalize(factors[0] + hn.lift(*sites))
    weights = jnp.asarray(args["weights"][0, :, None, None])
    offsets = jnp.asarray(args["offsets"][0, :, None])
    aux = hs.auxiliary_statistics(region, weights, offsets, jnp.zeros((2, 1)))
    matrix, bias, noise = (jnp.asarray(args[key][0]).reshape(shape) for key, shape in
                            (("matrix", (2, 1, 1)), ("bias", (2, 1)), ("noise", (2, 1, 1))))

    def energy(xi):
        factor = hs.transition(weights, offsets, xi, matrix, bias, noise)
        return -region.expectation(factor)

    derivative = jax.grad(energy)(jnp.zeros((2, 1)))
    np.testing.assert_allclose(derivative, aux.exposure * aux.residual / 96, atol=tolerance(np.float64))
    assert np.max(abs(derivative)) > 1e-4
    step = np.finfo(np.float64).eps ** (1 / 3)
    finite_difference = (energy(jnp.zeros((2, 1)).at[0, 0].set(step)) - energy(jnp.zeros((2, 1)))) / step
    assert finite_difference == pytest.approx(float(derivative[0, 0]), abs=32 * step)


def test_independent_symmetric_precision_coordinate_has_full_off_diagonal_derivative():
    factors, sites = build(case(3))
    sites = sites._replace(linear=jnp.concatenate((sites.linear, sites.linear / 2 + 0.1), axis=-1),
                           precision=sites.precision * jnp.eye(2))
    args = case(3)
    weights = jnp.stack((jnp.asarray(args["weights"]), jnp.full((2, 2), 0.15)), axis=-1)[:, :, None]
    factors = jax.vmap(hs.transition)(weights, jnp.asarray(args["offsets"])[..., None], jnp.ones((2, 2, 1)),
                                      jnp.asarray(args["matrix"])[..., None, None] * jnp.eye(2),
                                      jnp.zeros((2, 2, 2)), jnp.asarray(args["noise"])[..., None, None] * jnp.eye(2))
    outgoing = messages(sites)
    result = hs.evaluate(factors, sites, outgoing)
    q = jax.tree.map(lambda x: x[1], result.separators)
    b = hn.marginal(jax.tree.map(lambda x: x[1], result.regions), 0)
    expected = -(q.probability[1] * q.second[1, 0, 1] - b.probability[1] * b.second[1, 0, 1])

    def dual(shift):
        matrix = outgoing[1].precision.at[0, 1, 0, 1].add(shift).at[0, 1, 1, 0].add(shift)
        return hs.evaluate(factors, sites, (outgoing[0], outgoing[1]._replace(precision=matrix))).dual_energy

    step = np.finfo(np.float64).eps ** (1 / 3)
    assert jax.grad(dual)(0.0) == pytest.approx(float(expected), abs=128 * step ** 2)
    assert (dual(step) - dual(-step)) / (2 * step) == pytest.approx(float(expected), abs=128 * step ** 2)


def test_grouped_and_alternating_scopes_preserve_energy_and_cross_moments():
    factors, sites = map(unstack, build(case(2)))
    group = hn.normalize(factors[0] + hn.lift(*sites))
    recurrence = hs.source_region(group)
    intermediate = hn.collapse_label(recurrence, 0)
    dynamics = hn.collapse_label(group, 0)
    conditional = dynamics.covariance[:, 1, 1] - dynamics.covariance[:, 1, 0] ** 2 / dynamics.covariance[:, 0, 0]
    args = case(2)
    np.testing.assert_allclose(conditional, 1 / (1 / args["noise"][0] + sites[1].precision[:, 0, 0]))
    assert group.entropy() - recurrence.entropy() == pytest.approx(float(dynamics.entropy() - intermediate.entropy()))
    polynomial = hs.recurrence_potential(jnp.asarray(args["weights"][0, :, None, None]),
                                        jnp.asarray(args["offsets"][0, :, None]), jnp.ones((2, 1)))
    embedded = hn.Natural(polynomial.constant, jnp.pad(polynomial.linear, ((0, 0), (0, 0), (0, 1))),
                          jnp.pad(polynomial.precision, ((0, 0), (0, 0), (0, 1), (0, 1))))
    channel = hn.Natural(*(x - y for x, y in zip(factors[0].arrays(), embedded.arrays())))
    by_destination = hn.Natural(*(x[0] for x in channel.arrays()))
    assert group.expectation(channel) == pytest.approx(float(dynamics.expectation(by_destination)))


def test_gaussian_and_categorical_adapters_project_full_tilt_then_divide():
    prior = hn.canonical(hn.from_moments(jnp.array([0.5, 0.5]), jnp.array([[2.0], [-2.0]]), jnp.ones((2, 1, 1))))
    cavity = hn.Natural(jnp.array(0.0), jnp.zeros(1), jnp.ones((1, 1)))
    belief, hybrid, gaussian = jax.jit(hn.gaussian_adapter)(prior, cavity)
    np.testing.assert_allclose(belief.mean[:, 0], [1, -1])
    np.testing.assert_allclose(belief.covariance[:, 0, 0], 0.5)
    np.testing.assert_allclose(hybrid.precision, 1)
    assert gaussian.precision[0, 0] == pytest.approx(-1 / 3)
    assert -belief.entropy() + hn.normalize(prior + hybrid).entropy() == pytest.approx(0)
    categorical, hybrid, outgoing = jax.jit(hn.categorical_adapter)(prior, jnp.log(jnp.array([0.8, 0.2])))
    np.testing.assert_allclose(categorical.probability, [0.8, 0.2])
    np.testing.assert_allclose(categorical.mean, [[2], [-2]])
    np.testing.assert_allclose(jnp.exp(outgoing), [0.5, 0.5])
    np.testing.assert_allclose(hybrid.precision, 0, atol=tolerance(np.float64))


def test_temporal_falsifier_retains_equal_stationary_messages_and_opposite_learning_error():
    prior_p = np.exp(1) / (np.exp(1) + 2 * np.exp(0.25))
    cavity = hn.Natural(jnp.zeros(2), jnp.zeros((2, 1)), jnp.ones((2, 1, 1)))
    outputs = []
    for sign in (1, -1):
        means = sign * jnp.array([2.0, -1.0])
        constant = jnp.log(jnp.array([prior_p, 1 - prior_p])) - (jnp.log(2 * jnp.pi) + means ** 2) / 2 - jnp.log(2.0)
        tilted = hn.Natural(jnp.broadcast_to(constant[:, None], (2, 2)),
                            jnp.broadcast_to(means[:, None, None], (2, 2, 1)), jnp.full((2, 2, 1, 1), 2.0))
        projected = hn.collapse_label(hn.normalize(tilted), 0)
        np.testing.assert_allclose(projected.mean, 0, atol=tolerance(np.float64))
        np.testing.assert_allclose(projected.covariance, 1)
        outputs.append(hn.quotient(projected, cavity))
        reference = reflected_mixture_gap(sign, full=True)
        assert abs(reference["mean"]) > 0.01
        assert abs(reference["destination"][0] - 0.5) > 0.019
        assert np.sign(reference["linear"][1]) == -sign
    for actual, expected in zip(outputs[0].arrays(), outputs[1].arrays()):
        np.testing.assert_allclose(actual, expected, atol=tolerance(np.float64))


def test_fixed_drive_reduction_is_exact_categorical_stick_table():
    # A fixed drive is conditioned out; no singular Gaussian entropy is used.
    weights = jnp.array([[[0.7], [-0.2]], [[-0.5], [0.6]], [[0.4], [0.3]]])
    offsets = jnp.array([[-0.2, 0.3], [0.1, -0.4], [0.2, 0.5]])
    x = jnp.array([1.2])
    logits = jnp.einsum("ild,d->il", weights, x) + offsets
    polynomial = hs.recurrence_potential(weights, offsets, logits ** 2)
    scores = polynomial.constant + jnp.einsum("ijd,d->ij", polynomial.linear, x)
    scores -= jnp.einsum("ijab,a,b->ij", polynomial.precision, x, x) / 2
    expected = np.empty((3, 3))
    for i in range(3):
        remainder = 1.0
        for k in range(2):
            success = 1 / (1 + np.exp(-float(logits[i, k])))
            expected[i, k] = remainder * success
            remainder *= 1 - success
        expected[i, 2] = remainder
    np.testing.assert_allclose(jnp.exp(scores), expected, atol=tolerance(np.float64))


def test_multistick_auxiliary_and_learning_match_independent_destination_sums():
    weights = jnp.arange(12, dtype=jnp.float64).reshape(3, 2, 2) / 10 - 0.3
    offsets = jnp.array([[-0.2, 0.3], [0.1, -0.4], [0.2, 0.5]])
    xi = jnp.array([[0.0, 0.8], [0.3, 1.2], [0.7, 1.5]])
    noise = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
    factor = hs.transition(weights, offsets, xi, noise / 2, jnp.zeros((3, 2)), noise)
    site = hn.Natural(jnp.zeros(3), jnp.array([[0.2, -0.1], [-0.3, 0.4], [0.1, 0.3]]), noise)
    region = hn.normalize(factor + hn.lift(site, site))
    result = hs.auxiliary_statistics(region, weights, offsets, xi)
    expected_h, expected_j, target = np.zeros((3, 2, 3)), np.zeros((3, 2, 3, 3)), np.zeros((3, 2))
    for i in range(3):
        for stick in range(2):
            c = np.sqrt(float(xi[i, stick]))
            omega = 0.25 if c == 0 else np.tanh(c / 2) / (2 * c)
            theta = np.r_[weights[i, stick], offsets[i, stick]]
            exposure = 0.0
            for destination in range(stick, 3):
                p = float(region.probability[i, destination])
                mean = np.r_[region.mean[i, destination, :2], 1]
                second = np.outer(mean, mean)
                second[:2, :2] += np.asarray(region.covariance[i, destination, :2, :2])
                expected_h[i, stick] += (0.5 if destination == stick else -0.5) * p * mean
                expected_j[i, stick] += omega * p * second
                target[i, stick] += p * theta @ second @ theta
                exposure += p
            target[i, stick] /= exposure
    np.testing.assert_allclose(result.linear, expected_h, atol=tolerance(np.float64))
    np.testing.assert_allclose(result.precision, expected_j, atol=tolerance(np.float64))
    np.testing.assert_allclose(result.target, target, atol=tolerance(np.float64))
    np.testing.assert_allclose(result.residual, xi - target, atol=tolerance(np.float64))
