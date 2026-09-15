"""Native conditional mixtures near binary exponent boundaries."""

from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from cffg import hybrid_natural as hn


ROOT = Path(__file__).resolve().parents[1]
ALLOWANCE = 512 * 4 * np.finfo(np.float32).eps


def inputs(kind, card=4):
    probability = np.arange(1, card * card + 1, dtype=np.float32).reshape(card, card)
    if kind in ("uniform", "rare"):
        probability[:] = 1
    if kind == "rare":
        probability[0] = 0
        probability[1] *= np.float32(1e-12)
    probability /= probability.sum()
    mean = (np.arange(card * card * 4, dtype=np.float32).reshape(card, card, 4)
            - card * card * 2) / (4 * card)
    if kind in ("below", "above"):
        mean = np.nextafter(mean, np.float32(-np.inf if kind == "below" else np.inf))
        # Keep exact zero: subnormal incoming words are a separate failure case.
        mean[abs(mean) < np.finfo(np.float32).tiny] = 0
    high = np.broadcast_to(np.eye(4, dtype=np.float32) * 1e5, (card, card, 4, 4)).copy()
    if kind == "unequal":
        high += np.arange(card, dtype=np.float32)[None, :, None, None] / 64
    low = high * (np.finfo(np.float32).eps / 4)
    center = np.broadcast_to(np.float32([32, -16, 8, -4]), mean.shape).copy()
    return tuple(jnp.asarray(x) for x in (probability, mean, high, low, center))


def construct(probability, mean, high, low, center):
    return hn.from_centered_moments(probability, mean, high, center,
                                    covariance_correction=low)


def project(*arguments):
    region = construct(*arguments)
    return tuple(hn.marginal(region, side) for side in (0, 1))


def reference(arguments, side):
    """Independent total-moment identity; float64 is only the NumPy oracle.

    Subtract the common covariance before averaging to expose retained-word
    error instead of hiding it behind the large physical diagonal.
    """
    p, mean, high, low, _ = (np.asarray(x, np.float64) for x in arguments)
    selection = slice(2 * side, 2 * side + 2)
    mean, high, low = mean[..., selection], high[..., selection, selection], low[..., selection, selection]
    axis = 1 - side
    mass = p.sum(axis=axis)
    weight = p / np.expand_dims(np.where(mass > 0, mass, 1), axis)
    mu = (weight[..., None] * mean).sum(axis=axis)
    base = np.take(high, 0, axis=axis)
    local = high - np.expand_dims(base, axis) + low
    second = local + mean[..., :, None] * mean[..., None, :]
    increment = (weight[..., None, None] * second).sum(axis=axis) - mu[..., :, None] * mu[..., None, :]
    return mass, mu, base, increment


def assert_projection(actual, arguments, side):
    assert int(actual.status) == 0
    mass, mean, base, increment = reference(arguments, side)
    active = mass > 0
    for word in (actual.covariance, actual.correction, actual.full_covariance):
        assert word.dtype == np.float32
        np.testing.assert_array_equal(np.asarray(word)[~active], 0)
    np.testing.assert_array_equal(np.isneginf(actual.log_probability), ~active)
    np.testing.assert_allclose(actual.probability, mass, atol=0, rtol=ALLOWANCE)
    np.testing.assert_allclose(actual.displacement, mean, atol=ALLOWANCE, rtol=ALLOWANCE)
    high, low = (np.asarray(x, np.float64) for x in (actual.covariance, actual.correction))
    retained = (high - base) + low
    np.testing.assert_allclose(retained[active], increment[active], atol=ALLOWANCE, rtol=ALLOWANCE)
    physical = np.asarray(base + increment, np.float32)
    np.testing.assert_array_equal(actual.full_covariance, np.asarray(high + low, np.float32))
    np.testing.assert_allclose(np.asarray(actual.full_covariance)[active], physical[active],
                               atol=ALLOWANCE, rtol=ALLOWANCE)
    assert np.any(low[active] != 0)


def check_modes():
    cases = [inputs(kind) for kind in ("uniform", "weighted", "rare", "below", "above", "unequal")]
    for evaluate in (project, jax.jit(project)):
        for arguments in cases:
            for side, result in enumerate(evaluate(*arguments)):
                assert_projection(result, arguments, side)
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *cases)
    for evaluate in (jax.vmap(project), jax.jit(jax.vmap(project))):
        results = evaluate(*batch)
        for index, arguments in enumerate(cases):
            for side, result in enumerate(results):
                assert_projection(jax.tree.map(lambda x: x[index], result), arguments, side)
    # Ten times the labels, one hundred times the pair components; no solver.
    for kind in ("weighted", "rare"):
        arguments = inputs(kind, card=40)
        for side, result in enumerate(jax.jit(project)(*arguments)):
            assert_projection(result, arguments, side)


def check_derivatives():
    arguments = inputs("weighted")
    p, mean, high, low, center = arguments
    matrix = np.array([[.3, -.2], [-.2, .7]], np.float64)

    def output(p, mean, low):
        result = hn.marginal(construct(p, mean, high, low, center), 0)
        return jnp.sum((result.covariance - high[:, 0, :2, :2] + result.correction)
                       * jnp.asarray(matrix, jnp.float32))

    mass, mu, base, increment = reference(arguments, 0)
    weight = np.asarray(p, np.float64) / mass[:, None]
    delta = np.asarray(mean, np.float64)[..., :2] - mu[:, None]
    dm = np.zeros(mean.shape)
    dm[..., :2] = 2 * weight[..., None] * (delta @ matrix)
    dl = np.zeros(low.shape)
    dl[..., :2, :2] = weight[..., None, None] * matrix
    local = np.asarray(high, np.float64)[..., :2, :2] - base[:, None]
    local += np.asarray(low, np.float64)[..., :2, :2]
    local += delta[..., :, None] * delta[..., None, :]
    dp = np.sum((local - increment[:, None]) * matrix, axis=(-2, -1)) / mass[:, None]
    expected = (dp, dm, dl)
    directions = tuple(jnp.asarray(np.linspace(-.1, .1, x.size).reshape(x.shape), jnp.float32)
                       for x in (p, mean, low))
    # Covariance directions are symmetric; the primitive shares one triangle.
    directions = (*directions[:2], (directions[2] + directions[2].swapaxes(-1, -2)) / 2)
    for evaluate in (output, jax.jit(output)):
        tangent = jax.jvp(evaluate, (p, mean, low), directions)[1]
        expected_tangent = sum(np.sum(a * np.asarray(b)) for a, b in zip(expected, directions))
        np.testing.assert_allclose(tangent, expected_tangent, atol=ALLOWANCE, rtol=ALLOWANCE)
        gradients = jax.grad(evaluate, argnums=(0, 1, 2))(p, mean, low)
        for actual, truth in zip(gradients[:2], expected[:2]):
            np.testing.assert_allclose(actual, truth, atol=ALLOWANCE, rtol=ALLOWANCE)
        # Compare on symmetric covariance directions, including the low word.
        gradient = (gradients[2] + gradients[2].swapaxes(-1, -2)) / 2
        np.testing.assert_allclose(gradient, dl, atol=ALLOWANCE, rtol=ALLOWANCE)


def check_status():
    arguments = inputs("weighted")
    valid = construct(*arguments)
    empty = construct(jnp.zeros_like(arguments[0]), *arguments[1:])
    nonfinite = valid._replace(displacement=valid.displacement.at[0, 0, 0].set(jnp.inf))
    smallest = np.nextafter(np.float32(0), np.float32(1))
    subnormal = valid._replace(covariance_correction=valid.correction.at[0, 0, 0, 0].set(smallest))
    natural = hn.Natural(jnp.zeros_like(valid.probability).at[0].set(-1000),
                         jnp.zeros_like(valid.displacement),
                         jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), valid.covariance.shape))
    underflow = hn.normalize(natural)
    assert int(underflow.status) & hn.Status.UNDERFLOW
    assert np.isfinite(underflow.log_probability).all()
    rare = inputs("rare")
    tiny_mass = construct(rare[0].at[1].set(np.float32(2.0 ** -124)), *rare[1:])
    assert int(tiny_mass.status) == 0
    assert np.all(np.asarray(tiny_mass.probability)[1] >= np.finfo(np.float32).tiny)
    # Positive normal masses can still produce unrepresentable weighted variance.
    cases = (empty, nonfinite, subnormal, underflow, tiny_mass)
    flags = (hn.Status.EMPTY_SUPPORT, hn.Status.NONFINITE, hn.Status.NONFINITE,
             hn.Status.UNDERFLOW, hn.Status.NONFINITE)
    for evaluate in (lambda b: hn.marginal(b, 0), jax.jit(lambda b: hn.marginal(b, 0))):
        for belief, flag in zip(cases, flags):
            result = evaluate(belief)
            assert int(result.status) & flag
            assert np.isnan(result.log_normalizer)
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *cases)
    for evaluate in (jax.vmap(lambda b: hn.marginal(b, 0)), jax.jit(jax.vmap(lambda b: hn.marginal(b, 0)))):
        results = evaluate(batch)
        for status, flag in zip(results.status, flags):
            assert int(status) & flag


def test_native_weighted_projection_words_transforms_derivatives_and_status():
    source = """
import sys
sys.path.insert(0, 'tools')
sys.path.insert(0, 'tests')
from d12_compile_scaling import native_jax
jax = native_jax()
from test_hybrid_weighted_projection import check_modes, check_derivatives, check_status
check_modes()
check_derivatives()
check_status()
assert not jax.config.x64_enabled
print('native f32: mixture words, physical rounding, transforms, derivatives, status, K40 passed')
"""
    result = subprocess.run([sys.executable, "-c", source], cwd=ROOT, text=True,
                            capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
