"""Retained-coordinate regressions for complete local D12 message paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import hybrid_natural as hn
from cffg import hybrid_separator as hs
import hybrid_centering_reference as ref
from hybrid_kernel_cases import stack, tolerance


CASES = ((np.float32, 100, 1e-4), (np.float32, 10000, 1), (np.float64, 1e8, 1))


def scalar_case(dtype, origin, variance, *, sparse=False):
    p = jnp.array([1, 0] if sparse else [.4, .6], dtype)
    mean = jnp.full((2, 1), origin, dtype)
    left = hn.gaussian_site(jnp.log(p), mean, jnp.full((2, 1, 1), variance, dtype))
    right = hn.Natural(jnp.zeros(2, dtype), jnp.zeros((2, 1), dtype), jnp.zeros((2, 1, 1), dtype))
    params = hs.TransitionParameters(jnp.ones((2, 1, 1), dtype), jnp.full((2, 1), -origin, dtype),
                                    jnp.zeros((2, 1), dtype), jnp.zeros((2, 1, 1), dtype),
                                    jnp.zeros((2, 1), dtype), jnp.ones((2, 1, 1), dtype))
    return params, left, right


def local(params, left, right):
    factor = hs.transition(*params, left_center=left.origin, right_center=right.origin)
    region = hn.normalize(factor + hn.lift(left, right))
    outgoing = tuple(hn.quotient(hn.marginal(region, side), cavity)
                     for side, cavity in enumerate((left, right)))
    energy, status = hs.transition_expectation(region, params)
    aux = hs.auxiliary_statistics(region, params.weights, params.offsets, params.xi)
    return region, outgoing, energy, status, aux


@pytest.mark.parametrize("dtype,origin,variance", CASES)
@pytest.mark.parametrize("sparse", (False, True))
def test_translated_tilt_retains_probabilities_means_and_full_normalizer(dtype, origin, variance, sparse):
    budget = tolerance(dtype)
    baseline = local(*scalar_case(dtype, 0, variance, sparse=sparse))
    inputs = scalar_case(dtype, origin, variance, sparse=sparse)
    for evaluate in (local, jax.jit(local)):
        region, outgoing, energy, status, aux = evaluate(*inputs)
        hn.raise_for_status(region.status | status | aux.status)
        np.testing.assert_allclose(region.probability, baseline[0].probability, atol=budget, rtol=0)
        np.testing.assert_allclose(region.displacement, baseline[0].displacement, atol=budget, rtol=budget)
        np.testing.assert_allclose(region.covariance, baseline[0].covariance, atol=budget, rtol=budget)
        expected_logz = -np.log1p(variance / 4) / 2 + variance / (8 * (1 + variance / 4))
        assert region.log_normalizer == pytest.approx(expected_logz, abs=budget)
        assert region.entropy() == pytest.approx(float(baseline[0].entropy()), abs=budget)
        assert energy == pytest.approx(float(baseline[2]), abs=budget)
        for actual, expected in zip(outgoing, baseline[1]):
            for a, b in zip(actual.arrays(), expected.arrays()):
                np.testing.assert_allclose(a, b, atol=budget, rtol=budget)
        for a, b in zip(aux[:5], baseline[4][:5]):
            np.testing.assert_allclose(a, b, atol=budget, rtol=budget)
        # World readouts have dtype spacing at the translated coordinate;
        # internal displacement and label checks retain the original budget.
        world_budget = 4 * np.finfo(dtype).eps * (1 + abs(origin)) + budget
        active = np.asarray(region.probability > 0)
        np.testing.assert_allclose(np.asarray(region.mean[..., 0])[active] - origin,
                                   np.asarray(baseline[0].mean[..., 0])[active], atol=world_budget, rtol=0)


def two_dimensional_case(dtype, shift):
    mean = np.array([[[.25, -.5], [-.5, .75]], [[.25, .5], [-.5, -.25]]])
    covariance = np.array([[[.8, .2], [.2, 1.1]], [[1.2, -.15], [-.15, .7]]])
    weights = np.array([[[.5, -.25]], [[-.25, .5]]])
    matrix = np.array([[[.5, .25], [-.25, .5]], [[.25, -.25], [.5, .25]]])
    offset = np.array([[-.125], [.25]]) - np.einsum("ild,d->il", weights, shift)
    bias = np.array([[.25, -.125], [-.25, .375]]) + shift - np.einsum("jab,b->ja", matrix, shift)
    params = hs.TransitionParameters(*(jnp.asarray(x, dtype) for x in
        (weights, offset, [[0], [.5]], matrix, bias, covariance)))
    sites = tuple(hn.gaussian_site(jnp.log(jnp.array(p, dtype)), jnp.asarray(m + shift, dtype),
                                   jnp.asarray(covariance, dtype))
                  for p, m in zip(([.35, .65], [.6, .4]), mean))
    return params, *sites


def assert_reference(result, expected, budget):
    region, outgoing, energy, status, aux = result
    expected_region, expected_outgoing, expected_energy, expected_aux = expected
    hn.raise_for_status(region.status | status | aux.status)
    for actual, truth in zip((region.probability, region.displacement, region.covariance,
                               region.log_normalizer), expected_region):
        np.testing.assert_allclose(actual, truth, atol=budget, rtol=budget)
    assert region.entropy() == pytest.approx(ref.entropy(expected_region), abs=budget)
    assert energy == pytest.approx(expected_energy, abs=budget)
    for actual, truth in zip(outgoing, expected_outgoing):
        for a, b in zip(actual.arrays(), truth):
            np.testing.assert_allclose(a, b, atol=budget, rtol=budget)
    for actual, truth in zip(aux[:5], expected_aux):
        np.testing.assert_allclose(actual, truth, atol=budget, rtol=budget)


@pytest.mark.parametrize("dtype,shift", ((np.float32, [10000, -10000]), (np.float64, [1e8, -1e8])))
def test_two_dimensional_repeated_projection_quotient_and_statistics_match_independent_reference(dtype, shift):
    budget = tolerance(dtype)
    params, left, right = two_dimensional_case(dtype, np.asarray(shift))
    zero_params, zero_left, zero_right = two_dimensional_case(dtype, np.zeros(2))
    factor = ref.factor(zero_params, np.asarray(zero_left.origin), np.asarray(zero_right.origin))
    reference_covariance = np.array([[[.8, .2], [.2, 1.1]], [[1.2, -.15], [-.15, .7]]])
    reference_cavities = [ref.gaussian(np.log(p), np.asarray(s.origin), reference_covariance)
                          for p, s in zip(([.35, .65], [.6, .4]), (zero_left, zero_right))]
    for _ in range(3):
        region = ref.normalize(ref.add(factor, ref.lift(*reference_cavities)))
        outgoing = tuple(ref.quotient(ref.project(region, side), cavity)
                         for side, cavity in enumerate(reference_cavities))
        expected = region, outgoing, ref.expectation(region, factor), ref.statistics(region, zero_params, zero_left.origin)
        result = jax.jit(local)(params, left, right)
        assert_reference(result, expected, budget)
        left, right = (cavity + message for cavity, message in zip((left, right), result[1]))
        reference_cavities = [ref.add(cavity, message) for cavity, message in zip(reference_cavities, outgoing)]


@pytest.mark.parametrize("dtype,origin,variance", CASES)
def test_scalar_projection_quotient_reuse_is_translation_invariant(dtype, origin, variance):
    centered = scalar_case(dtype, 0, variance)
    translated = scalar_case(dtype, origin, variance)
    for _ in range(3):
        results = [jax.jit(local)(*values) for values in (centered, translated)]
        for result in results:
            hn.raise_for_status(result[0].status | result[3] | result[4].status)
        for a, b in zip((results[0][0].probability, results[0][0].displacement, results[0][0].covariance),
                        (results[1][0].probability, results[1][0].displacement, results[1][0].covariance)):
            np.testing.assert_allclose(a, b, atol=tolerance(dtype), rtol=tolerance(dtype))
        centered, translated = [(values[0], *(cavity + message for cavity, message in zip(values[1:], result[1])))
                                 for values, result in zip((centered, translated), results)]


def chain_case(dtype, shift, time=3):
    params, left, right = two_dimensional_case(dtype, np.asarray(shift))
    sites = stack([left if t % 2 == 0 else right for t in range(time)])
    params = stack([params] * (time - 1))
    outgoing = tuple(hn.Natural(jnp.zeros_like(sites.constant[:-1]), jnp.zeros_like(sites.linear[:-1]),
                                jnp.zeros_like(sites.precision[:-1]), jnp.zeros(time - 1, jnp.int32),
                                sites.origin[side:side + time - 1]) for side in (0, 1))
    return params, sites, outgoing


@pytest.mark.parametrize("dtype,shift", ((np.float32, [10000, -10000]), (np.float64, [1e8, -1e8])))
def test_chain_repeated_evaluation_preserves_objective_and_all_message_blocks(dtype, shift):
    values = [chain_case(dtype, shift), chain_case(dtype, [0, 0])]
    evaluate = jax.jit(hs.evaluate)
    for _ in range(3):
        outputs = [evaluate(*v) for v in values]
        for result in outputs:
            hn.raise_for_status(result.status)
        for name in ("free_energy", "dual_energy", "moment_residual", "quotient_residual"):
            np.testing.assert_allclose(getattr(outputs[0], name), getattr(outputs[1], name),
                                       atol=tolerance(dtype), rtol=tolerance(dtype))
        for name in ("regions", "separators"):
            a, b = (getattr(out, name) for out in outputs)
            for field in ("probability", "displacement", "covariance", "log_normalizer"):
                np.testing.assert_allclose(getattr(a, field), getattr(b, field),
                                           atol=tolerance(dtype), rtol=tolerance(dtype))
        for a, b in zip(outputs[0].targets, outputs[1].targets):
            for x, y in zip(a.arrays(), b.arrays()):
                np.testing.assert_allclose(x, y, atol=tolerance(dtype), rtol=tolerance(dtype))
        values = [(p, s, output.targets) for (p, s, _), output in zip(values, outputs)]


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_adapters_keep_origins_signed_messages_and_label_scales(dtype):
    def adapt(shift):
        mean = jnp.array([[.25], [-.5]], dtype) + shift
        prior = hn.gaussian_site(jnp.log(jnp.array([.4, .6], dtype)), mean, jnp.array([[[.5]], [[.75]]], dtype))
        gaussian = hn.gaussian_site(jnp.array(.25, dtype), jnp.array([shift], dtype), jnp.array([[.125]], dtype))
        belief, hybrid, ordinary = hn.gaussian_adapter(prior, gaussian)
        reconstructed = hn.normalize(gaussian + ordinary)
        categorical = hn.categorical_adapter(prior, jnp.log(jnp.array([.25, .75], dtype)))
        return belief, hybrid, ordinary, reconstructed, categorical

    baseline = adapt(0)
    for function in (adapt, jax.jit(adapt)):
        result = function(jnp.asarray(10000, dtype))
        for value in (result[0], result[1], result[2], result[3], result[4][0]):
            hn.raise_for_status(value.status)
        for index in (0, 3):
            for field in ("probability", "displacement", "covariance", "log_normalizer"):
                np.testing.assert_allclose(getattr(result[index], field), getattr(baseline[index], field),
                                           atol=tolerance(dtype), rtol=tolerance(dtype))
        for index in (1, 2):
            for a, b in zip(result[index].arrays(), baseline[index].arrays()):
                np.testing.assert_allclose(a, b, atol=tolerance(dtype), rtol=tolerance(dtype))
        np.testing.assert_allclose(result[4][2], baseline[4][2], atol=tolerance(dtype))
    assert np.all(np.asarray(result[2].precision) > 0)


def test_raw_expanded_sites_have_stored_coefficient_semantics_and_lost_inputs_are_not_inferred():
    means, variance = np.full((2, 1), 100, np.float32), np.float32(1e-4)
    raw = []
    for p in ([.4, .6], [.5, .5]):
        a = np.asarray(np.log(p) - (np.log(2 * np.pi * float(variance)) + 100 ** 2 / float(variance)) / 2, np.float32)
        raw.append(hn.Natural(jnp.asarray(a), jnp.asarray(means / variance),
                               jnp.full((2, 1, 1), 1 / variance)))
    for a, b in zip(raw[0].arrays(), raw[1].arrays()):
        np.testing.assert_array_equal(a, b)
    for value in raw:
        result = hn.normalize(value)
        hn.raise_for_status(result.status)
        np.testing.assert_array_equal(result.center, 0)
        np.testing.assert_array_equal(result.probability, [.5, .5])


def test_independent_scalar_reference_jit_and_vmap_remains_an_executable_anchor():
    probability, logz = jax.vmap(ref.anchored_tilt, in_axes=(0, None))(
        jnp.array([0, 100, 10000], jnp.float32), jnp.array([.4, .6], jnp.float32))
    np.testing.assert_allclose(probability, np.broadcast_to([[.2, .2], [.3, .3]], probability.shape), atol=1e-6)
    expected = -np.log1p(1e-4 / 4) / 2 + 1e-4 / (8 * (1 + 1e-4 / 4))
    np.testing.assert_allclose(logz, expected, atol=5e-7)


@pytest.mark.parametrize("dtype,origin,variance", CASES)
@pytest.mark.parametrize("sparse", (False, True))
def test_retained_xi_derivative_matches_analytic_identity_and_finite_difference(dtype, origin, variance, sparse):
    params, left, right = scalar_case(dtype, origin, variance, sparse=sparse)
    region, _, _, _, aux = local(params, left, right)

    def energy(xi):
        return -hs.transition_expectation(region, params._replace(xi=xi))[0]

    derivative = jax.grad(energy)(params.xi)
    np.testing.assert_allclose(derivative, aux.exposure * aux.residual / 96,
                               atol=64 * np.finfo(dtype).eps * variance, rtol=tolerance(dtype))
    step = np.finfo(dtype).eps ** (1 / 3)
    difference = (-3 * energy(params.xi) + 4 * energy(params.xi.at[0, 0].add(step))
                  - energy(params.xi.at[0, 0].add(2 * step))) / (2 * step)
    assert difference == pytest.approx(float(derivative[0, 0]), abs=32 * step ** 2)

    def dual(xi):
        factor = hs.transition(*params._replace(xi=xi), left_center=left.origin, right_center=right.origin)
        return -hn.normalize(factor + hn.lift(left, right)).log_normalizer

    np.testing.assert_allclose(jax.grad(dual)(params.xi), derivative,
                               atol=64 * np.finfo(dtype).eps * variance, rtol=tolerance(dtype))


@pytest.mark.parametrize("coordinate", ("constant", "linear", "precision"))
def test_translated_chain_dual_derivatives_use_retained_incidence_moments(coordinate):
    params, sites, messages = chain_case(np.float64, [1e8, -1e8])
    result = hs.evaluate(params, sites, messages)
    separator = jax.tree.map(lambda a: a[1], result.separators)
    projection = hn.marginal(jax.tree.map(lambda a: a[1], result.regions), 0)
    moment = lambda b: b.covariance[1, 0, 0] + b.displacement[1, 0] ** 2
    expected = {"constant": separator.probability[1] - projection.probability[1],
                "linear": separator.probability[1] * separator.displacement[1, 0]
                          - projection.probability[1] * projection.displacement[1, 0],
                "precision": -(separator.probability[1] * moment(separator)
                               - projection.probability[1] * moment(projection)) / 2}[coordinate]
    index = {"constant": (0, 1), "linear": (0, 1, 0), "precision": (0, 1, 0, 0)}[coordinate]

    def dual(change):
        outgoing = messages[1]._replace(**{coordinate: getattr(messages[1], coordinate).at[index].add(change)})
        return hs.evaluate(params, sites, (messages[0], outgoing)).dual_energy

    step = np.finfo(np.float64).eps ** (1 / 3)
    assert jax.grad(dual)(0.) == pytest.approx(float(expected), abs=128 * step ** 2)
    assert (dual(step) - dual(-step)) / (2 * step) == pytest.approx(float(expected), abs=128 * step ** 2)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_rebasing_preserves_stored_polynomial_projection_and_physical_statistics(dtype):
    params, left, right = two_dimensional_case(dtype, np.array([10000, -10000]))
    region, _, _, _, aux = local(params, left, right)
    projected = hn.marginal(region, 0)
    density = hn.canonical(projected)
    center = density.center + jnp.array([[.125, -.25], [-.25, .125]], dtype)
    changed = hn.rebase(density, center)
    reconstructed = hn.normalize(changed)
    hn.raise_for_status(reconstructed.status)
    np.testing.assert_allclose(reconstructed.probability, projected.probability, atol=tolerance(dtype))
    np.testing.assert_allclose(reconstructed.relative_mean(projected.center), projected.displacement,
                               atol=tolerance(dtype), rtol=tolerance(dtype))
    np.testing.assert_allclose(reconstructed.covariance, projected.covariance, atol=tolerance(dtype))
    assert reconstructed.entropy() == pytest.approx(float(projected.entropy()), abs=tolerance(dtype))
    assert projected.expectation(changed) == pytest.approx(-float(projected.entropy()), abs=tolerance(dtype))
    for a, b in zip(hn.rebase(changed, density.center).arrays(), density.arrays()):
        np.testing.assert_allclose(a, b, atol=tolerance(dtype), rtol=tolerance(dtype))
    world_region = hn.from_moments(region.probability, region.mean, region.covariance)
    physical = hs.auxiliary_statistics(world_region, params.weights, params.offsets, params.xi)
    np.testing.assert_allclose(aux.physical_linear, physical.linear, atol=tolerance(dtype), rtol=tolerance(dtype))
    np.testing.assert_allclose(aux.physical_precision, physical.precision, atol=tolerance(dtype), rtol=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("kind,flag", (("center", hn.Status.NONFINITE), ("mean", hn.Status.NONFINITE),
                                      ("covariance", hn.Status.IMPROPER), ("empty", hn.Status.EMPTY_SUPPORT),
                                      ("underflow", hn.Status.UNDERFLOW), ("mass", hn.Status.NONFINITE)))
def test_retained_gaussian_invalid_inputs_fail_on_host_and_jit(dtype, kind, flag):
    log_mass = jnp.log(jnp.array([.4, .6], dtype))
    mean, covariance = jnp.full((2, 1), 100, dtype), jnp.ones((2, 1, 1), dtype)
    center = mean
    if kind == "center":
        center = center.at[0, 0].set(jnp.nan)
    elif kind == "mean":
        mean = mean.at[0, 0].set(jnp.inf)
    elif kind == "covariance":
        covariance = covariance.at[0, 0, 0].set(-1)
    elif kind == "empty":
        log_mass = jnp.full(2, -jnp.inf, dtype)
    elif kind == "underflow":
        log_mass = log_mass.at[1].set(2 * np.log(np.finfo(dtype).tiny))
    else:
        log_mass = log_mass.at[1].set(jnp.inf)

    def construct(a, m, v, c):
        return hn.normalize(hn.gaussian_site(a, m, v, center=c))

    for function in (construct, jax.jit(construct)):
        result = function(log_mass, mean, covariance, center)
        assert result.status & flag
        assert not np.isfinite(result.log_normalizer)
        assert np.all(np.isnan(np.asarray(result.mean)[np.isfinite(log_mass)]))


def test_center_shapes_and_dtypes_are_validated_before_arithmetic():
    params, left, right = two_dimensional_case(np.float32, np.zeros(2))
    for center in (jnp.zeros((2, 1), jnp.float32), jnp.zeros((2, 2), jnp.float64)):
        with pytest.raises(ValueError, match="center.*shapes|center.*dtypes"):
            hn.normalize(left._replace(center=center))
        with pytest.raises(ValueError, match="center.*shapes|center.*dtypes"):
            hs.transition(*params, left_center=center, right_center=right.origin)


def test_retained_origins_are_polynomial_metadata_in_a_moderate_float32_chain():
    params, sites, messages = chain_case(np.float32, [10000, -10000], time=100)
    result = jax.jit(hs.evaluate)(params, sites, messages)
    hn.raise_for_status(result.status)
    assert result.regions.covariance.shape == (99, 2, 2, 4, 4)
    assert result.separators.covariance.shape == (100, 2, 2, 2)
    center_bytes = sum(x.center.nbytes for x in (sites, *messages, result.regions, result.separators, *result.targets))
    expected_centers = 100 * 2 * 2 + 4 * 99 * 2 * 2 + 99 * 2 * 2 * 4 + 100 * 2 * 2
    assert center_bytes == expected_centers * 4
    total_bytes = sum(x.nbytes for x in jax.tree.leaves((params, sites, messages, result)))
    assert total_bytes < 100 * 2 ** 2 * 4 ** 2 * 4 * 5
    for message in messages:
        assert len(message.arrays()) == 3
        assert sum(x.size for x in message.arrays()) == 99 * 2 * (1 + 2 + 2 ** 2)


def test_anchored_single_vmap_and_jit_match_every_local_output():
    inputs = stack([two_dimensional_case(np.float32, np.asarray(shift)) for shift in
                    ([0, 0], [10000, -10000], [-10000, 10000])])
    single = stack([local(*jax.tree.map(lambda a: a[t], inputs)) for t in range(3)])
    for result in (jax.vmap(local)(*inputs), jax.jit(jax.vmap(local))(*inputs)):
        for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(single)):
            np.testing.assert_allclose(a, b, atol=tolerance(np.float32), rtol=tolerance(np.float32))


@pytest.mark.parametrize("precision", (-.01, 0))
def test_anchored_signed_and_vacuous_cavities_and_direct_recurrence(precision):
    params, left, right = two_dimensional_case(np.float32, np.array([10000, -10000]))
    right = hn.Natural(jnp.zeros_like(right.constant), jnp.zeros_like(right.linear),
                       jnp.broadcast_to(jnp.eye(2, dtype=jnp.float32) * precision, right.precision.shape),
                       center=right.center)
    assert hn.normalize(right).status & hn.Status.IMPROPER
    region, outgoing, _, status, aux = jax.jit(local)(params, left, right)
    hn.raise_for_status(region.status | status | aux.status)
    reconstructed = hn.normalize(left + outgoing[0])
    expected = hn.marginal(region, 0)
    np.testing.assert_allclose(reconstructed.probability, expected.probability, atol=tolerance(np.float32))
    np.testing.assert_allclose(reconstructed.displacement, expected.displacement, atol=tolerance(np.float32))
    base_params, base_left, _ = two_dimensional_case(np.float32, np.zeros(2))
    direct = hs.recurrence_potential(params.weights, params.offsets, params.xi, center=left.center)
    reference = hs.recurrence_potential(base_params.weights, base_params.offsets, base_params.xi,
                                       center=base_left.center)
    for a, b in zip(direct.arrays(), reference.arrays()):
        np.testing.assert_allclose(a, b, atol=tolerance(np.float32))


def test_rebased_message_residual_removes_only_one_common_gauge():
    _, left, _ = two_dimensional_case(np.float64, np.array([10000, -10000]))
    target = hn.rebase(left, left.center + jnp.array([[.125, -.25], [-.25, .125]]))
    target = target._replace(constant=target.constant + 3)
    residual, status = hs.natural_residual(left, target)
    hn.raise_for_status(status)
    assert residual < tolerance(np.float64)
    target = target._replace(constant=target.constant.at[1].add(.1))
    residual, status = hs.natural_residual(left, target)
    hn.raise_for_status(status)
    assert residual > .01


def test_finite_constructor_inputs_cannot_overflow_into_literal_log_zero_support():
    dtype = jnp.float32
    mean = jnp.array([[2e20], [0]], dtype)
    covariance = jnp.ones((2, 1, 1), dtype)
    site = hn.gaussian_site(jnp.log(jnp.array([.4, .6], dtype)), mean, covariance,
                            center=jnp.zeros_like(mean))
    assert site.status & hn.Status.NONFINITE
    base = hn.Natural(jnp.zeros(2, dtype), jnp.zeros_like(mean), covariance)
    changed = hn.rebase(base, mean)
    assert changed.status & hn.Status.NONFINITE
    large = base._replace(constant=jnp.array([-3e38, 0], dtype))
    multiplied = large + large
    assert multiplied.status & hn.Status.NONFINITE
    lifted = hn.lift(large, large)
    assert lifted.status & hn.Status.NONFINITE
    assert np.isnan(lifted.constant[0, 0])
    for natural in (site, changed, multiplied):
        belief = hn.normalize(natural)
        assert belief.status & hn.Status.NONFINITE
        assert not np.isneginf(natural.constant[0])
        assert np.isnan(belief.mean[0, 0])
    params, _, _ = scalar_case(np.float32, 0, 1)
    recurrence = hs.recurrence_potential(params.weights, params.offsets, params.xi, center=mean)
    factor = hs.transition(*params, left_center=jnp.zeros_like(mean), right_center=mean)
    for natural in (recurrence, factor):
        assert natural.status & hn.Status.NONFINITE
        assert not np.any(np.isneginf(natural.constant))
