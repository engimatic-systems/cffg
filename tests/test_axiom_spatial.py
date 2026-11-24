"""AXIOM spatial coordinates checked by fitting the quadratic log joint on a grid."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, softmax

from cffg.experiments.axiom_spatial import Positions, infer_slots
from test_frame_regression import example


def reference(model, pixels, prior, initial, alpha, active, sweeps):
    m, P, a, b = [np.asarray(x, dtype=float) for x in model]
    prior0, cov0 = [np.asarray(x, dtype=float) for x in prior]
    position, C = [np.array(x, dtype=float, copy=True) for x in initial]
    pixels = np.asarray(pixels, dtype=float)
    k, o, d = m.shape
    logmix = digamma(np.asarray(alpha, float)) - digamma(np.sum(alpha))

    def evaluate(i, location, covariance):
        u = np.r_[location, 1.0]
        S = np.outer(u, u)
        S[:-1, :-1] += covariance
        sq = pixels**2 - 2 * pixels * (m[i] @ u) + np.diag(m[i] @ S @ m[i].T)
        return 0.5 * np.sum(
            digamma(a[i])
            - np.log(2 * np.pi * b[i])
            - sq * a[i] / b[i]
            - np.trace(np.linalg.solve(P[i], S)),
            axis=1,
        )

    for _ in range(sweeps):
        ell = np.column_stack(
            [
                evaluate(i, position[i], C[i]) if active[i] else np.full(len(pixels), -np.inf)
                for i in range(k)
            ]
        )
        w = softmax(ell + logmix, axis=1)
        for i in np.flatnonzero(active):
            mu = prior0[i]

            def F(off):
                lik = w[:, i] @ evaluate(i, mu + off, np.zeros_like(C[i]))
                return lik - 0.5 * off @ np.linalg.solve(cov0[i], off)

            zero = np.zeros(d - 1)
            eye = np.eye(d - 1)
            g = np.array([(F(e) - F(-e)) / 2 for e in eye])
            H = np.array(
                [[(F(s + t) - F(s - t) - F(-s + t) + F(-s - t)) / 4 for t in eye] for s in eye]
            )
            C[i] = np.linalg.inv(-H)
            position[i] = mu + C[i] @ g
    return position, C, w, ell.max(axis=1)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_posterior_positions_and_assignments_match_quadratic_log_joint(dtype):
    model, frame, _ = example(dtype)
    k = 3
    prior = Positions(
        frame.design[:, :-1], jnp.tile(jnp.array([[1.0, 0.2], [0.2, 0.75]], dtype), (k, 1, 1))
    )
    initial = Positions(prior.mean + 0.3, prior.cov * 0.7)
    alpha = jnp.array([1.0, 4.0, 1.0], dtype)
    for sweeps in (1, 3):
        out = jax.jit(infer_slots, static_argnames="sweeps")(
            model, frame.pixels, prior, initial, alpha, frame.active, sweeps=sweeps
        )
        expected = reference(
            model, frame.pixels, prior, initial, alpha, np.asarray(frame.active), sweeps
        )
        assert out.valid
        for val, ref in zip(
            (*out.positions, out.assignment, out.best_log_likelihood), expected, strict=True
        ):
            np.testing.assert_allclose(
                val,
                ref,
                rtol=3e-5 if dtype == jnp.float32 else 1e-11,
                atol=3e-6 if dtype == jnp.float32 else 1e-11,
            )
        assert int(out.occupancy.sum()) == len(frame.pixels)
        np.testing.assert_array_equal(out.assignment[:, 2], 0)
        np.testing.assert_array_equal(out.positions.mean[2], initial.mean[2])
    index = jnp.array([2, 0, 1])
    permuted = infer_slots(
        jax.tree.map(lambda x: x[index], model),
        frame.pixels,
        jax.tree.map(lambda x: x[index], prior),
        jax.tree.map(lambda x: x[index], initial),
        alpha[index],
        frame.active[index],
        sweeps=3,
    )
    assert permuted.valid
    np.testing.assert_allclose(permuted.assignment, out.assignment[:, index], rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(permuted.occupancy, out.occupancy[index])


def test_empty_support_and_undefined_pixel_fail_explicitly():
    model, frame, _ = example(jnp.float32)
    pos = Positions(frame.design[:, :2], jnp.tile(jnp.eye(2), (3, 1, 1)).astype(jnp.float32))
    alpha = jnp.ones(3, jnp.float32)
    for pixels, mask in [
        (frame.pixels, jnp.zeros(3, bool)),
        (frame.pixels.at[0, 0].set(jnp.nan), frame.active),
    ]:
        out = infer_slots(model, pixels, pos, pos, alpha, mask)
        assert not out.valid
        np.testing.assert_array_equal(out.assignment, 0)
        np.testing.assert_array_equal(out.positions.mean, pos.mean)
