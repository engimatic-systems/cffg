"""Small message increments survive D12 covariance projection and division."""

import json
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import hybrid_covariance as hc, hybrid_natural as hn, hybrid_separator as hs
from hybrid_kernel_cases import stack, tolerance
import hybrid_projection_reference as ref
from test_hybrid_affine import case
from test_hybrid_centering import local, chain_case


def stored_case(dtype):
    params, left, right = case(dtype, 10000, 1e-4, [6.7], -67000)
    factor = hs.transition(*params, left_center=left.origin, right_center=right.origin)
    return hn.normalize(factor + hn.lift(left, right)), left, right


def both_messages(region, left, right):
    return tuple(hn.quotient(hn.marginal(region, side), cavity)
                 for side, cavity in enumerate((left, right)))


def assert_messages(actual, expected, dtype):
    hn.raise_for_status(actual.status)
    for value, truth in zip(actual.arrays(), expected):
        np.testing.assert_allclose(value, np.asarray(truth, float), atol=tolerance(dtype), rtol=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_identical_stored_region_has_exact_scalar_reference_and_transformed_parity(dtype):
    region, left, right = stored_case(dtype)
    expected_precision = ref.scalar_precision(region.probability, region.displacement[..., :1],
                                              ref.represented(region.covariance, region.correction)[..., :1, :1], left.precision)
    references = []
    for side, cavity in enumerate((left, right)):
        s = slice(side, side + 1)
        moments = ref.project(region.probability, region.displacement[..., s],
                              ref.represented(region.covariance, region.correction)[..., s, s], 1 - side)
        references.append(ref.quotient(moments, ref.natural(cavity)))
    batch = stack([region, region]), stack([left, left]), stack([right, right])
    baseline = both_messages(region, left, right)
    for evaluate, inputs in ((both_messages, (region, left, right)),
                              (jax.jit(both_messages), (region, left, right)),
                              (jax.vmap(both_messages), batch), (jax.jit(jax.vmap(both_messages)), batch)):
        outputs = evaluate(*inputs)
        if outputs[0].constant.ndim == 2:
            outputs = jax.tree.map(lambda x: x[0], outputs)
        for actual, expected, prior in zip(outputs, references, baseline):
            assert_messages(actual, expected, dtype)
            assert_messages(actual, prior.arrays(), dtype)
        np.testing.assert_allclose(outputs[0].precision[:, 0, 0], expected_precision,
                                   atol=tolerance(dtype), rtol=tolerance(dtype))
    projection = hn.marginal(region, 0)
    assert np.any(np.asarray(projection.covariance_correction) != 0)
    represented = ref.array(projection.covariance) + ref.array(projection.correction)
    exact = ref.project(region.probability, region.displacement[..., :1], ref.represented(region.covariance, region.correction)[..., :1, :1], 1)[2]
    error = np.asarray(abs(represented - exact), float)
    rounded_error = np.asarray(abs(ref.array(projection.covariance) - exact), float)
    assert np.all(error <= rounded_error + 4 * np.finfo(dtype).eps ** 2 * abs(np.asarray(exact, float)))


def matrix_case(dtype, card=3, dim=2):
    rng = np.random.default_rng(72)
    p = rng.uniform(.2, 1, (card, card)); p /= p.sum()
    mean = rng.normal(size=(card, card, 2 * dim)) * 1e-5
    base = np.eye(dim) * 1e-4
    if dim > 1:
        base[0, 1] = base[1, 0] = 2e-5
    covariance = np.broadcast_to(np.kron(np.eye(2), base), (card, card, 2 * dim, 2 * dim)).copy()
    for i in range(card):
        for j in range(card):
            covariance[i, j] *= 1 + (i + j) * 1e-6
    region = hn.from_moments(*(jnp.asarray(x, dtype) for x in (p, mean, covariance)))
    precision = ref.inverse_logdet(np.asarray(base, dtype))[0]
    cavity = hn.Natural(jnp.asarray(np.linspace(-.2, .3, card), dtype),
                       jnp.asarray(rng.normal(size=(card, dim)) * .01, dtype),
                       jnp.broadcast_to(jnp.asarray(np.asarray(precision, float), dtype), (card, dim, dim)))
    return region, cavity


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_offdiagonal_both_marginals_all_coefficients_and_canonical_match_stored_reference(dtype):
    region, cavity = matrix_case(dtype)
    for side in (0, 1):
        s = slice(2 * side, 2 * side + 2)
        moments = ref.project(region.probability, region.displacement[..., s], ref.represented(region.covariance, region.correction)[..., s, s], 1 - side)
        expected = ref.quotient(moments, ref.natural(cavity))
        for evaluate in (lambda b, c: hn.quotient(hn.marginal(b, side), c),
                         jax.jit(lambda b, c: hn.quotient(hn.marginal(b, side), c))):
            assert_messages(evaluate(region, cavity), expected, dtype)
        projected = hn.marginal(region, side)
        for value, truth in zip((projected.probability, projected.displacement, projected.full_covariance), moments):
            np.testing.assert_allclose(value, np.asarray(truth, float), atol=tolerance(dtype), rtol=tolerance(dtype))
        canonical = hn.canonical(projected)
        for value, truth in zip(canonical.arrays(), ref.canonical(*moments)):
            np.testing.assert_allclose(value, np.asarray(truth, float), atol=tolerance(dtype), rtol=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_adapters_divide_the_identical_stored_tilt_including_small_increments(dtype):
    region, cavity = matrix_case(dtype)
    hybrid = hn.Natural(cavity.constant, cavity.linear, cavity.precision * 1e-6)
    gaussian = cavity._replace(constant=cavity.constant[0], linear=cavity.linear[0], precision=cavity.precision[0])
    for evaluate in (hn.gaussian_adapter, jax.jit(hn.gaussian_adapter)):
        belief, hybrid_message, ordinary_message = evaluate(hybrid, gaussian)
        p, m, c = ref.belief(belief)
        assert_messages(hybrid_message, ref.quotient((p, m, c), hybrid.arrays()), dtype)
        assert_messages(ordinary_message, ref.quotient(ref.project(p, m, c, 0), gaussian.arrays()), dtype)
    for evaluate in (hn.categorical_adapter, jax.jit(hn.categorical_adapter)):
        belief, message, categorical = evaluate(cavity, jnp.log(jnp.array([.2, .3, .5], dtype)))
        expected = ref.quotient(ref.belief(belief), ref.natural(cavity))
        assert_messages(message, expected, dtype)
        assert np.isfinite(categorical).all()


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_retained_covariance_derivatives_include_low_word_and_match_matrix_identities(dtype):
    base = jnp.array([[1, .2], [.2, 2]], dtype)
    low = base * (np.finfo(dtype).eps / 4)
    rhs = jnp.array([[.7], [-.3]], dtype)
    direction = jnp.array([[.3, -.1], [-.1, .2]], dtype)
    inverse = np.asarray(ref.inverse_logdet(ref.array(base) + ref.array(low))[0], float)
    solution = inverse @ np.asarray(rhs)
    expected = -inverse @ np.asarray(direction) @ solution
    for arg in (0, 1):
        tangent = (direction, jnp.zeros_like(direction)) if arg == 0 else (jnp.zeros_like(direction), direction)
        actual = jax.jvp(lambda a, b: hc.solve(a, b, rhs), (base, low), tangent)[1]
        np.testing.assert_allclose(actual, expected, atol=tolerance(dtype), rtol=tolerance(dtype))
        for derivative in (jax.jacfwd(hc.logdet, argnums=arg), jax.jacrev(hc.logdet, argnums=arg)):
            np.testing.assert_allclose(derivative(base, low), inverse.T, atol=tolerance(dtype), rtol=tolerance(dtype))
    hessian = jax.hessian(lambda a: hc.logdet(base, a))(jnp.zeros_like(base))
    assert np.isfinite(hessian).all()
    scalar_second = jax.grad(jax.grad(lambda t: hc.solve(base, t * direction, rhs).sum()))(jnp.array(0, dtype))
    expected_second = (2 * inverse @ np.asarray(direction) @ inverse @ np.asarray(direction) @ solution).sum()
    np.testing.assert_allclose(scalar_second, expected_second, atol=tolerance(dtype), rtol=tolerance(dtype))
    eps = 1e-3 if dtype == np.float32 else 1e-4
    finite = (hc.solve(base + eps * direction, low, rhs) - hc.solve(base - eps * direction, low, rhs)) / (2 * eps)
    np.testing.assert_allclose(finite, expected, atol=8e-5 if dtype == np.float32 else 1e-8, rtol=1e-6)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_projected_quotient_derivative_agrees_with_independent_finite_difference(dtype):
    region, cavity = matrix_case(dtype)
    direction = jnp.ones_like(region.displacement) * .001
    def output(t):
        moved = region._replace(displacement=region.displacement + t * direction)
        return jnp.concatenate([x.ravel() for x in hn.quotient(hn.marginal(moved, 0), cavity).arrays()])
    derivative = jax.jacfwd(output)(jnp.array(0, dtype))
    step = ref.number('0.0001')
    references = []
    for t in (-step, step):
        mean = ref.array(region.displacement) + t * ref.array(direction)
        moments = ref.project(region.probability, mean[..., :2], region.covariance[..., :2, :2], 1)
        references.append(np.concatenate([x.ravel() for x in ref.quotient(moments, ref.natural(cavity))]))
    expected = (references[1] - references[0]) / (2 * step)
    np.testing.assert_allclose(derivative, np.asarray(expected, float), atol=tolerance(dtype), rtol=tolerance(dtype))
    np.testing.assert_allclose(jax.jacrev(output)(jnp.array(0, dtype)), derivative, atol=tolerance(dtype), rtol=tolerance(dtype))


def test_float32_projection_counterexample_with_x64_disabled():
    script = textwrap.dedent('''
        import sys
        sys.path.insert(0, 'tests')
        import jax
        jax.config.update('jax_enable_x64', False)
        import numpy as np
        from test_hybrid_projection import (
            stored_case, both_messages, assert_messages, assert_solve_underflow,
            assert_second_moment_boundary, assert_small_mean_readout)
        import hybrid_projection_reference as ref
        assert_solve_underflow(np.float32)
        assert_second_moment_boundary(np.float32)
        assert_small_mean_readout(np.float32)
        region, left, right = stored_case(np.float32)
        expected = ref.scalar_precision(region.probability, region.displacement[..., :1], ref.represented(region.covariance, region.correction)[..., :1, :1], left.precision)
        for evaluate in (both_messages, jax.jit(both_messages)):
            outputs = evaluate(region, left, right)
            for message in outputs:
                assert all(x.dtype == np.float32 for x in message.arrays())
                assert int(message.status) == 0
            np.testing.assert_allclose(outputs[0].precision[:, 0, 0], expected, atol=512*4*np.finfo(np.float32).eps, rtol=512*4*np.finfo(np.float32).eps)
    ''')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_repeated_chain_targets_match_each_stored_region_with_unchanged_full_objective(dtype):
    import hybrid_centering_reference as centered_ref
    params, sites, messages = chain_case(dtype, [0, 0], time=3)
    evaluate = jax.jit(hs.evaluate)
    for _ in range(3):
        result = evaluate(params, sites, messages)
        hn.raise_for_status(result.status)
        references = []
        for time in range(2):
            region = jax.tree.map(lambda x: x[time], result.regions)
            parameter = jax.tree.map(lambda x: x[time], params)
            stored = tuple(np.asarray(x, float) for x in
                           (region.probability, region.displacement, region.full_covariance, region.log_normalizer))
            factor = centered_ref.factor(parameter, region.center[:, 0, :2], region.center[0, :, 2:])
            references.append(-centered_ref.entropy(stored) - centered_ref.expectation(stored, factor))
            auxiliary = hs.auxiliary_statistics(region, parameter.weights, parameter.offsets, parameter.xi)
            truth = centered_ref.statistics(stored, parameter, region.center[:, 0, :2])
            for value, expected in zip(auxiliary[:5], truth):
                np.testing.assert_allclose(value, expected, atol=tolerance(dtype), rtol=tolerance(dtype))
            for side in (0, 1):
                cavity = jax.tree.map(lambda x: x[time + side], sites)
                if side == 0 and time > 0:
                    cavity += jax.tree.map(lambda x: x[time - 1], messages[1])
                if side == 1 and time < 1:
                    cavity += jax.tree.map(lambda x: x[time + 1], messages[0])
                selection = slice(2 * side, 2 * side + 2)
                moments = ref.project(region.probability, region.displacement[..., selection],
                                      ref.represented(region.covariance, region.correction)[..., selection, selection], 1 - side)
                target = jax.tree.map(lambda x: x[time], result.targets[side])
                assert_messages(target, ref.quotient(moments, ref.natural(cavity)), dtype)
        expected_fe = sum(references)
        for time in range(3):
            separator = jax.tree.map(lambda x: x[time], result.separators)
            stored = tuple(np.asarray(x, float) for x in
                           (separator.probability, separator.displacement, separator.full_covariance, separator.log_normalizer))
            if time == 1:
                expected_fe += centered_ref.entropy(stored)
            site = jax.tree.map(lambda x: x[time], sites)
            expected_fe -= centered_ref.expectation(stored, site.arrays())
        assert result.free_energy == pytest.approx(expected_fe, abs=tolerance(dtype))
        messages = result.targets


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_covariance_correction_validation_and_sparse_projection_preserve_status(dtype):
    region, cavity = matrix_case(dtype)
    p = region.probability.at[:, 0].set(0)
    p /= p.sum()
    sparse = hn.from_moments(p, region.displacement, region.covariance)
    for evaluate in (hn.marginal, jax.jit(hn.marginal, static_argnums=1)):
        for side in (0, 1):
            projected = evaluate(sparse, side)
            message = hn.quotient(projected, cavity)
            hn.raise_for_status(projected.status | message.status)
            absent = projected.probability == 0
            assert np.all(np.isneginf(np.asarray(message.constant)[absent]))
            assert np.all(np.asarray(projected.correction)[absent] == 0)
    base = jnp.eye(2, dtype=dtype)
    options = ((base * np.nan, hn.Status.NONFINITE),
               (base * .1, hn.Status.INVALID),
               (jnp.array([[0, 1e-20], [0, 0]], dtype), hn.Status.INVALID),
               (jnp.asarray(np.eye(2, dtype=dtype) * np.nextafter(dtype(0), dtype(1))), hn.Status.NONFINITE))
    for low, status in options:
        result = hn.from_centered_moments(jnp.ones((), dtype), jnp.zeros(2, dtype), base,
                                          jnp.zeros(2, dtype), covariance_correction=low)
        assert result.status & status
    with pytest.raises(ValueError, match="correction shapes"):
        hn.from_centered_moments(jnp.ones((), dtype), jnp.zeros(2, dtype), base, jnp.zeros(2, dtype),
                                 covariance_correction=base[0])


def test_covariance_correction_storage_and_matrix_products_have_only_local_matrix_buffers():
    time, card, dim = 100, 20, 10
    region, _ = matrix_case(np.float32, card, dim)
    regions = stack([region] * time)
    projections = tuple(jax.eval_shape(jax.vmap(lambda b: hn.marginal(b, side)), regions) for side in (0, 1))
    assert all(p.covariance_correction.shape == (time, card, dim, dim) for p in projections)
    retained_bytes = sum(p.covariance_correction.size * p.covariance_correction.dtype.itemsize for p in projections)
    assert retained_bytes == 1_600_000
    matrix = jax.ShapeDtypeStruct((time, card, dim, dim), jnp.float32)
    graph = jax.make_jaxpr(hc.product)(matrix, matrix)
    def inspect(jaxpr):
        if hasattr(jaxpr, 'jaxpr'):
            jaxpr = jaxpr.jaxpr
        for equation in jaxpr.eqns:
            for value in equation.outvars:
                if hasattr(value.aval, 'shape'):
                    shape = value.aval.shape
                    assert np.prod(shape) <= time * card * dim * dim
            for value in equation.params.values():
                if hasattr(value, 'jaxpr') or hasattr(value, 'eqns'):
                    inspect(value)
    inspect(graph)


def test_default_and_inherited_statuses_are_int32_under_stacking_and_vmap():
    default = hn.Natural(jnp.zeros(2), jnp.zeros((2, 1)), jnp.ones((2, 1, 1)))
    invalid = default._replace(status=int(hn.Status.INVALID))
    batch = stack([default, invalid])
    for evaluate in (jax.vmap(hn.normalize), jax.jit(jax.vmap(hn.normalize))):
        beliefs = evaluate(batch)
        np.testing.assert_array_equal(beliefs.status, [0, hn.Status.INVALID])
        assert beliefs.status.dtype == jnp.int32
    region, cavity = matrix_case(np.float32)
    inherited = stack([region._replace(status=0), region._replace(status=int(hn.Status.INVALID))])
    for evaluate in (jax.vmap(lambda b: hn.marginal(b, 0)), jax.jit(jax.vmap(lambda b: hn.marginal(b, 0)))):
        projected = evaluate(inherited)
        np.testing.assert_array_equal(projected.status, [0, hn.Status.INVALID])
        assert projected.status.dtype == jnp.int32
    potential = hn.Natural(jnp.zeros_like(region.probability), jnp.zeros_like(region.displacement),
                           region.covariance)
    for evaluate in (jax.vmap(lambda b: b.expectation(potential)), jax.jit(jax.vmap(lambda b: b.expectation(potential)))):
        result = evaluate(inherited)
        assert np.isfinite(result[0]) and np.isnan(result[1])


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_small_increment_survives_three_composed_local_evaluations(dtype):
    params, left, right = case(dtype, 10000, 1e-4, [6.7], -67000)
    evaluate = jax.jit(local)
    for _ in range(3):
        region, messages, energy, status, auxiliary = evaluate(params, left, right)
        hn.raise_for_status(region.status | status | auxiliary.status)
        for side, cavity in enumerate((left, right)):
            selection = slice(side, side + 1)
            moments = ref.project(region.probability, region.displacement[..., selection],
                                  ref.represented(region.covariance, region.correction)[..., selection, selection], 1 - side)
            assert_messages(messages[side], ref.quotient(moments, ref.natural(cavity)), dtype)
        free_energy = -region.entropy() - energy - region.expectation(hn.lift(left, right))
        assert free_energy == pytest.approx(-float(region.log_normalizer), abs=tolerance(dtype))
        left, right = (cavity + message for cavity, message in zip((left, right), messages))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_retained_words_reach_entropy_moments_source_statistics_and_energy(dtype):
    import hybrid_centering_reference as centered_ref
    params, left, right = case(dtype, 0, 1e-4, [6.7], 0)
    factor = hs.transition(*params, left_center=left.origin, right_center=right.origin)
    region = hn.normalize(factor + hn.lift(left, right))
    correction = region.covariance * (np.finfo(dtype).eps / 4)
    held = hn.from_centered_moments(region.probability, region.displacement, region.covariance, region.center,
                                    covariance_correction=correction)
    represented = ref.array(held.covariance) + ref.array(held.correction)
    expected_h = float(ref.entropy(held.probability, held.displacement, represented))
    assert held.entropy() == pytest.approx(expected_h, abs=tolerance(dtype))
    expected_f = float(ref.expectation(held.probability, held.displacement, represented, factor.arrays()))
    assert held.expectation(factor) == pytest.approx(expected_f, abs=tolerance(dtype))
    np.testing.assert_allclose(held.second, np.asarray(represented, float) + np.asarray(held.mean)[..., :, None] * np.asarray(held.mean)[..., None, :],
                               atol=tolerance(dtype), rtol=tolerance(dtype))
    np.testing.assert_array_equal(hs.source_region(held).correction, correction[..., :1, :1])
    stored = held.probability, held.displacement, np.asarray(represented, float), 0
    auxiliary = hs.auxiliary_statistics(held, params.weights, params.offsets, params.xi)
    for value, truth in zip(auxiliary[:5], centered_ref.statistics(stored, params, left.origin)):
        np.testing.assert_allclose(value, truth, atol=tolerance(dtype), rtol=tolerance(dtype))
    energy, status = hs.transition_expectation(held, params)
    hn.raise_for_status(status | auxiliary.status)
    assert energy == pytest.approx(expected_f, abs=tolerance(dtype))
    derivative = jax.grad(lambda low: hs.auxiliary_statistics(
        held._replace(covariance_correction=low), params.weights, params.offsets, params.xi).target.sum())(correction)
    expected = np.zeros_like(derivative)
    weights = np.asarray(held.probability) / np.asarray(held.probability).sum(axis=1, keepdims=True)
    expected[..., 0, 0] = weights * float(params.weights[0, 0, 0]) ** 2
    np.testing.assert_allclose(derivative, expected, atol=tolerance(dtype), rtol=tolerance(dtype))


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_directly_supplied_malformed_correction_cannot_bypass_consumers(dtype):
    region, cavity = matrix_case(dtype)
    projected = hn.marginal(region, 0)
    malformed = projected._replace(covariance_correction=projected.covariance / 2)
    for evaluate in (hn.canonical, jax.jit(hn.canonical)):
        assert evaluate(malformed).status & hn.Status.INVALID
    assert hn.quotient(malformed, cavity).status & hn.Status.INVALID
    assert np.isnan(malformed.entropy()) and np.isnan(malformed.expectation(cavity))
    assert hs.moment_residual(malformed, projected)[1] & hn.Status.INVALID
    malformed_region = region._replace(covariance_correction=region.covariance / 2)
    assert hn.marginal(malformed_region, 0).status & hn.Status.INVALID
    assert np.isnan(hn.marginal(malformed_region, 0).probability).all()
    params = hs.TransitionParameters(jnp.zeros((3, 2, 2), dtype), jnp.zeros((3, 2), dtype),
                                    jnp.zeros((3, 2), dtype), jnp.broadcast_to(jnp.eye(2, dtype=dtype), (3, 2, 2)),
                                    jnp.zeros((3, 2), dtype), jnp.broadcast_to(jnp.eye(2, dtype=dtype), (3, 2, 2)))
    assert hs.auxiliary_statistics(malformed_region, params.weights, params.offsets, params.xi).status & hn.Status.INVALID



def assert_solve_underflow(dtype):
    from fractions import Fraction
    magnitude = 1e30 if dtype == np.float32 else 1e300
    base = jnp.array([[magnitude]], dtype)
    low = jnp.zeros_like(base)
    rhs = jnp.array([[1e-10]], dtype)
    exact = Fraction(float(rhs[0, 0])) / Fraction(float(base[0, 0]))
    assert 0 < exact < Fraction(float(np.finfo(dtype).tiny))
    for evaluate in (hc.solve, jax.jit(hc.solve)):
        assert np.isnan(evaluate(base, low, rhs)).all()
        np.testing.assert_array_equal(evaluate(base, low, jnp.zeros_like(rhs)), 0)
    batch = jnp.stack((base, base)), jnp.stack((low, low)), jnp.stack((rhs, jnp.zeros_like(rhs)))
    for evaluate in (jax.vmap(hc.solve), jax.jit(jax.vmap(hc.solve))):
        result = evaluate(*batch)
        assert np.isnan(result[0]).all()
        np.testing.assert_array_equal(result[1], 0)
    cavity = hn.Natural(jnp.zeros((), dtype), jnp.zeros(1, dtype), jnp.zeros((1, 1), dtype))
    for evaluate in (hn.quotient, jax.jit(hn.quotient)):
        for inherited in (0, hn.Status.INVALID):
            incoming = cavity._replace(status=int(inherited))
            belief = hn.from_moments(jnp.ones((), dtype), rhs[:, 0], base)
            message = evaluate(belief, incoming)
            assert message.status & hn.Status.NONFINITE
            assert int(message.status) & int(inherited) == int(inherited)
            assert all(np.isnan(x).all() for x in message.arrays())
        zero = belief._replace(displacement=jnp.zeros_like(belief.displacement))
        hn.raise_for_status(evaluate(zero, cavity).status)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_unrepresentable_solve_fails_and_exact_zero_remains_valid(dtype):
    assert_solve_underflow(dtype)


def _assert_moment_boundary_modes(construct, arguments, expected_second, dtype):
    size = arguments[0].shape[0]
    for evaluate, batched in ((construct, False), (jax.jit(construct), False),
                              (jax.vmap(construct), True), (jax.jit(jax.vmap(construct)), True)):
        if batched:
            beliefs, seconds = evaluate(*arguments)
        else:
            beliefs, seconds = stack([evaluate(*(x[i] for x in arguments)) for i in range(size)])
        np.testing.assert_array_equal(beliefs.status, [hn.Status.NONFINITE] + [0] * (size - 1))
        assert np.isnan(seconds[0]).all() and np.isnan(beliefs.covariance[0]).all()
        assert np.isfinite(seconds[1:]).all()
        np.testing.assert_allclose(seconds[1:], expected_second, atol=0, rtol=2*np.finfo(dtype).eps)


def assert_second_moment_boundary(dtype):
    maximum = dtype(np.finfo(dtype).max)
    fraction = dtype(.75 if dtype == np.float32 else .76)
    mean = dtype(np.sqrt(maximum * fraction))
    high = np.nextafter(dtype(maximum - dtype(mean * mean)), dtype(np.inf))
    low = dtype(high * dtype(np.finfo(dtype).eps))
    finite_mean = np.nextafter(mean, dtype(0))
    with np.errstate(over="ignore"):
        high_only = dtype(high + dtype(mean * mean))
        represented = dtype(high + low)
        overflowing = dtype(represented + dtype(mean * mean))
        finite_second = dtype(represented + dtype(finite_mean * finite_mean))
    assert np.isfinite(high_only) and np.isfinite(represented)
    assert np.isinf(overflowing) and np.isfinite(finite_second)

    def construct(covariance, correction, displacement):
        belief = hn.from_centered_moments(jnp.ones((), dtype), displacement, covariance,
                                          jnp.zeros(1, dtype), covariance_correction=correction)
        return belief, belief.second

    arguments = (jnp.full((3, 1, 1), high, dtype),
                 jnp.asarray([[[low]], [[low]], [[0]]], dtype),
                 jnp.asarray([[mean], [finite_mean], [mean]], dtype))
    expected_second = np.asarray([finite_second, high_only], dtype).reshape(2, 1, 1)
    _assert_moment_boundary_modes(construct, arguments, expected_second, dtype)

    root = dtype(np.sqrt(maximum))
    finite_center = np.nextafter(root, dtype(0))
    overflowing_center = np.nextafter(root, dtype(np.inf))
    def normalize(center):
        belief = hn.normalize(hn.Natural(jnp.zeros((), dtype), jnp.zeros(1, dtype),
                                          jnp.ones((1, 1), dtype), center=center))
        return belief, belief.second
    arguments = (jnp.asarray([[overflowing_center], [finite_center]], dtype),)
    expected_second = np.asarray([[[dtype(1) + dtype(finite_center * finite_center)]]], dtype)
    _assert_moment_boundary_modes(normalize, arguments, expected_second, dtype)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_second_moment_guard_uses_represented_covariance_at_overflow_boundary(dtype):
    assert_second_moment_boundary(dtype)


def assert_small_mean_readout(dtype):
    from fractions import Fraction
    from cffg.hybrid_affine import rounded_product

    small = dtype(1e-16 if dtype == np.float32 else 1e-150)
    tiny_mean = dtype(1e-20 if dtype == np.float32 else 1e-160)
    rounded = dtype(small * small)
    remainder = Fraction(float(small)) ** 2 - Fraction(float(rounded))
    assert rounded >= np.finfo(dtype).tiny
    assert 0 < abs(remainder) < Fraction(float(np.finfo(dtype).tiny))
    point = jnp.asarray([small], dtype)
    for product in (rounded_product, jax.jit(rounded_product)):
        np.testing.assert_array_equal(product(point, point), [rounded])
    for product in (hc.product, jax.jit(hc.product)):
        assert np.isnan(product(point[:, None], point[None, :])).all()

    def moments(mean):
        belief = hn.from_centered_moments(jnp.ones((), dtype), mean, jnp.ones((1, 1), dtype),
                                          jnp.zeros(1, dtype))
        return belief.status, belief.second
    def normalized(mean):
        belief = hn.normalize(hn.Natural(jnp.zeros((), dtype), mean, jnp.ones((1, 1), dtype)))
        return belief.status, belief.second
    values = jnp.asarray([[small], [tiny_mean], [0], [.125]], dtype)
    expected = np.asarray([1, 1, 1, 1.015625], dtype).reshape(4, 1, 1)
    for constructor in (moments, normalized):
        for evaluate, batched in ((constructor, False), (jax.jit(constructor), False),
                                  (jax.vmap(constructor), True), (jax.jit(jax.vmap(constructor)), True)):
            status, second = (evaluate(values) if batched else
                              stack([evaluate(value) for value in values]))
            np.testing.assert_array_equal(status, np.zeros(4, np.int32))
            np.testing.assert_array_equal(second, expected)
        scalar = lambda m: constructor(m)[1].sum()
        tangent = jax.jvp(scalar, (point,), (jnp.ones_like(point),))[1]
        np.testing.assert_allclose(tangent, 2 * small, atol=0, rtol=16*np.finfo(dtype).eps)
        for gradient in (jax.jacfwd(scalar), jax.jacrev(scalar), jax.jit(jax.grad(scalar))):
            np.testing.assert_allclose(gradient(point), 2 * point, atol=0, rtol=16*np.finfo(dtype).eps)
        np.testing.assert_allclose(jax.hessian(scalar)(point), [[2]], atol=0, rtol=16*np.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_second_moment_does_not_require_an_unused_subnormal_remainder(dtype):
    assert_small_mean_readout(dtype)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_second_moment_readout_retains_forward_reverse_and_second_derivatives(dtype):
    covariance = jnp.eye(2, dtype=dtype)
    low = covariance * (np.finfo(dtype).eps / 4)
    mean = jnp.array([.7, -.1], dtype)
    center = jnp.array([.2, -.3], dtype)
    direction = jnp.array([-.4, .3], dtype)
    weight = jnp.array([[.3, .1], [.1, .4]], dtype)
    def moments(m, correction):
        return hn.from_centered_moments(jnp.ones((), dtype), m, covariance, center,
                                         covariance_correction=correction).second
    def normalized(m, correction):
        del correction
        return hn.normalize(hn.Natural(jnp.zeros((), dtype), m, covariance, center=center)).second
    expected_tangent = jnp.outer(direction, mean + center) + jnp.outer(mean + center, direction)
    for readout in (moments, normalized):
        scalar = lambda m: jnp.sum(weight * readout(m, low))
        np.testing.assert_allclose(jax.jvp(lambda m: readout(m, low), (mean,), (direction,))[1],
                                   expected_tangent, atol=tolerance(dtype), rtol=tolerance(dtype))
        for gradient in (jax.jacfwd(scalar), jax.jacrev(scalar)):
            np.testing.assert_allclose(gradient(mean), 2 * weight @ (mean + center),
                                       atol=tolerance(dtype), rtol=tolerance(dtype))
        np.testing.assert_allclose(jax.hessian(scalar)(mean), 2 * weight,
                                   atol=tolerance(dtype), rtol=tolerance(dtype))
    np.testing.assert_allclose(jax.grad(lambda correction: jnp.sum(weight * moments(mean, correction)))(low),
                               weight, atol=tolerance(dtype), rtol=tolerance(dtype))


def compiled_inventory(time=100, card=20, dim=10):
    """Lowered local allocation; time counts independent pair regions, not states."""
    shape = lambda s: jax.ShapeDtypeStruct(s, jnp.float32)
    region = hn.Belief(shape((time, card, card)), shape((time, card, card, 2*dim)),
                       shape((time, card, card, 2*dim, 2*dim)), shape((time, card, card)),
                       shape((time,)), jax.ShapeDtypeStruct((time,), jnp.int32),
                       shape((time, card, card, 2*dim)), shape((time, card, card, 2*dim, 2*dim)))
    def projections(value):
        return tuple(jax.vmap(lambda b: hn.marginal(b, side))(value) for side in (0, 1))
    def incidences(value, left, right):
        projected = projections(value)
        targets = tuple(jax.vmap(hn.quotient)(b, c) for b, c in zip(projected, (left, right)))
        return projected, targets
    matrix = shape((time, card, dim, dim))
    cavity = hn.Natural(shape((time, card)), shape((time, card, dim)), matrix,
                        jax.ShapeDtypeStruct((time,), jnp.int32), shape((time, card, dim)))
    grouped = hn.Natural(region.probability, region.displacement, region.covariance,
                         region.status, region.center, region.covariance_correction)
    result = {"shape": {"T": time, "K": card, "d": dim}, "x64_enabled": jax.config.x64_enabled}
    for name, function, arguments in (("both_projections", projections, (region,)),
                                      ("both_projections_and_quotients", incidences, (region, cavity, cavity)),
                                      ("compensated_product", hc.product, (matrix, matrix)),
                                      ("corrected_solve", hc.solve, (matrix, matrix, matrix)),
                                      ("grouped_normalization", jax.vmap(hn.normalize), (grouped,)),
                                      ("grouped_second_moment", jax.vmap(lambda b: b.second), (region,))):
        memory = jax.jit(function).lower(*arguments).compile().memory_analysis()
        result[name] = {key: getattr(memory, key) for key in
                       ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
        result[name]["input_tree_bytes"] = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(arguments))
        result[name]["output_arrays"] = [{"shape": list(x.shape), "bytes": x.size * x.dtype.itemsize}
                                         for x in jax.tree.leaves(jax.eval_shape(function, *arguments))]
    result["retained_covariance_correction_bytes"] = 2 * time * card * dim * dim * 4
    result["matrix_accumulator_carry_bytes"] = 2 * time * card * dim * dim * 4
    result["product_high_and_low_bytes"] = 2 * time * card * dim * dim * 4
    result["grouped_matrix_accumulator_carry_bytes"] = 2 * time * card * card * (2 * dim) ** 2 * 4
    result["grouped_rounded_product_bytes"] = time * card * card * (2 * dim) ** 2 * 4
    result["grouped_product_remainder_bytes"] = 0
    result["grouped_integer_significand_limb_bytes"] = 2 * time * card * card * (2 * dim) ** 2 * 4
    result["grouped_new_correction_bytes"] = time * card * card * (2 * dim) ** 2 * 4
    result["assembled_precision_correction_bytes"] = result["grouped_new_correction_bytes"]
    result["separator_normalization_correction_bytes"] = time * card * dim * dim * 4
    result["message_new_correction_bytes"] = 0
    return result


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    print(json.dumps(compiled_inventory(), indent=2))
