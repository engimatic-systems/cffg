"""Independent augmented least-squares and likelihood checks for AXIOM pixel fits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma

from cffg.frame_regression import Frame, Regression, fit_frame, score_pixels


def example(dtype):
    def a(x):
        return jnp.asarray(x, dtype)

    # The third input is an exact bias. The first two inputs can be correlated.
    model = Regression(
        a(np.arange(18).reshape(3, 2, 3) / 13),
        a([[[2, 0.2, -0.04], [0.2, 1, 0.1], [-0.04, 0.1, 0.5]]] * 3),
        a([[2, 3], [4, 2], [2, 2]]),
        a([[0.4, 0.8], [1.2, 0.7], [2, 1]]),
    )
    frame = Frame(
        a([[1, 2], [0.2, -1], [3, 2], [-2, 4]]),
        a([[1, 2, 1], [0.5, -0.4, 1], [0, 0, 1]]),
        a(
            [
                [[0.3, 0.15, 0], [0.15, 0.2, 0], [0, 0, 0]],
                [[0.1, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        ),
        jnp.array([True, True, False]),
    )
    return model, frame, a([[1, 0, 0], [0.25, 0.75, 0], [0.4, 0.3, 0], [0, 0, 0]])


def oracle(prior, frame, weights):
    m, p, a, b = (np.asarray(v, dtype=float) for v in prior)
    pixels, design, cov = (np.asarray(v, dtype=float) for v in frame[:3])
    w = np.asarray(weights, dtype=float)
    results = [v.copy() for v in (m, p, a, b)]
    scores = np.full((len(w), len(m)), -np.inf)
    for k in np.flatnonzero(frame.active):
        # Symmetric cubature reproduces all input moments of this quadratic model.
        values, vectors = np.linalg.eigh(cov[k])
        offsets = (vectors * np.sqrt(np.maximum(values, 0) * len(values))).T
        points = np.concatenate((design[k] + offsets, design[k] - offsets))
        prior_root = np.linalg.cholesky(p[k]).T
        features = [prior_root]
        targets = [prior_root @ m[k].T]
        for n, pix in enumerate(pixels):
            weight = np.sqrt(w[n, k] / len(points))
            features.append(weight * points)
            targets.append(weight * np.tile(pix, (len(points), 1)))
            errs = (pix - points @ m[k].T) ** 2
            uncertainty = np.sum((points @ np.linalg.inv(p[k])) * points, axis=1)
            scores[n, k] = 0.5 * np.sum(
                digamma(a[k])
                - np.log(2 * np.pi * b[k])
                - errs.mean(axis=0) * a[k] / b[k]
                - uncertainty.mean()
            )
        X, Y = np.vstack(features), np.vstack(targets)
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        results[0][k] = beta.T
        results[1][k] = X.T @ X
        results[2][k] += w[:, k].sum() / 2
        results[3][k] += np.sum((Y - X @ beta) ** 2, axis=0) / 2
    return scores, results


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_frame_uses_uncertain_input_and_independent_output_precisions(dtype):
    prior, frame, w = example(dtype)
    expected, posterior = oracle(prior, frame, w)
    scores, valid = jax.jit(score_pixels)(prior, frame)
    assert valid
    np.testing.assert_allclose(
        scores, expected, atol=64 * np.finfo(dtype).eps, rtol=64 * np.finfo(dtype).eps
    )
    for fit in (fit_frame, jax.jit(fit_frame)):
        actual, valid = fit(prior, frame, w)
        assert valid
        for x, y, p in zip(actual, posterior, prior, strict=True):
            np.testing.assert_array_equal(x[2], p[2])
            np.testing.assert_allclose(
                x, y, atol=64 * np.finfo(dtype).eps, rtol=64 * np.finfo(dtype).eps
            )


def test_frame_ignores_missing_pixels_only_at_zero_weight_and_rejects_improper_prior():
    prior, frame, w = example(jnp.float32)
    missing = frame._replace(pixels=frame.pixels.at[-1, :].set(jnp.nan))
    correct, ok = fit_frame(prior, frame, w)
    assert ok
    result, ok = fit_frame(prior, missing, w)
    assert ok
    for x, y in zip(result, correct, strict=True):
        np.testing.assert_array_equal(x, y)
    for p, f, weights in [
        (prior, missing, w.at[-1, 0].set(1)),
        (prior, frame, w.at[1, 2].set(0.2)),
        (prior._replace(precision=prior.precision.at[0, 0, 0].set(-1)), frame, w),
    ]:
        result, ok = fit_frame(p, f, weights)
        assert not ok
        for x, y in zip(result, p, strict=True):
            np.testing.assert_array_equal(x, y)


def test_float32_collinear_design_and_translated_bias_preserve_noise_learning():
    dtype = jnp.float32
    prior = Regression(
        jnp.array([[[0.0, 0.0, 10000.0]]], dtype),
        jnp.eye(3, dtype=dtype)[None],
        jnp.ones((1, 1), dtype),
        jnp.ones((1, 1), dtype),
    )
    frame = Frame(
        jnp.array([[10001.0], [9999.0]], dtype),
        jnp.ones((1, 3), dtype),
        jnp.zeros((1, 3, 3), dtype),
        jnp.array([True]),
    )
    w = jnp.ones((2, 1), dtype)
    result, ok = jax.jit(fit_frame)(prior, frame, w)
    assert ok
    np.testing.assert_array_equal(result.mean, prior.mean)
    np.testing.assert_allclose(result.rate, [[2.0]])
