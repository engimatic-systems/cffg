"""Independent tiny D12 coordinate references; no CFFG imports.

The scalar JAX reference is also executable directly. The NumPy reference
below integrates generic local blocks and reuses projected messages.

This checks one local Gaussian/JJ integral, not a chain solver. The source
Gaussian has variance 1e-4, the recurrence logit is x-center, dynamics is
N(y; 0, 1), and the right cavity is vacuous. Each source regime has the same
conditional Gaussian. The fixed JJ auxiliary is zero.
"""

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def anchored_tilt(center, probabilities):
    """Integrate a 2D pair block in coordinates (x-center, y)."""
    dtype = center.dtype
    anchor = center
    logit_offset = -center + anchor
    variance = jnp.asarray(1e-4, dtype)
    signs = jnp.array([1.0, -1.0], dtype)
    recurrence_h = 0.5 * signs - 0.25 * logit_offset
    recurrence_a = (
        -jnp.log(jnp.asarray(2.0, dtype))
        + 0.5 * signs * logit_offset
        - 0.125 * logit_offset**2
    )
    source_a = jnp.log(probabilities) - 0.5 * jnp.log(2 * jnp.pi * variance)
    dynamics_a = -0.5 * jnp.log(jnp.asarray(2 * jnp.pi, dtype))
    constant = source_a[:, None] + recurrence_a[None, :] + dynamics_a
    precision = jnp.diag(jnp.stack([1 / variance + 0.25, jnp.asarray(1.0, dtype)]))
    linear = jnp.stack([recurrence_h, jnp.zeros_like(recurrence_h)], axis=-1)
    chol = jnp.linalg.cholesky(precision)
    solved = jax.scipy.linalg.solve_triangular(chol, linear.T, lower=True).T
    log_mass = (
        constant
        + jnp.log(jnp.asarray(2 * jnp.pi, dtype))
        - jnp.log(jnp.diag(chol)).sum()
        + 0.5 * jnp.square(solved).sum(axis=-1)[None, :]
    )
    log_normalizer = jax.scipy.special.logsumexp(log_mass)
    return jnp.exp(log_mass - log_normalizer), log_normalizer


def main():
    # Distinct intended label probabilities become identical stored arrays.
    rounded = []
    for probabilities in ([0.4, 0.6], [0.5, 0.5]):
        constant = np.asarray(
            np.log(probabilities)
            - 0.5 * (np.log(2 * np.pi * 1e-4) + 100.0**2 / 1e-4),
            dtype=np.float32,
        )
        rounded.append(constant)
        print("Expanded float32 source constants:", probabilities, constant.tolist())
    np.testing.assert_array_equal(*rounded)

    for probabilities in ([0.5, 0.5], [0.4, 0.6]):
        masses, log_normalizers = jax.vmap(
            lambda center: anchored_tilt(center, jnp.asarray(probabilities, jnp.float32))
        )(jnp.array([0.0, 100.0, 10000.0], jnp.float32))
        expected = np.asarray(probabilities)[:, None] * np.ones((1, 2)) / 2
        np.testing.assert_allclose(
            np.asarray(masses), np.broadcast_to(expected, masses.shape), rtol=1e-6
        )
        np.testing.assert_array_equal(masses[0], masses[1])
        np.testing.assert_array_equal(masses[0], masses[2])
        # Scalar analytic integral of exp(+/-r/2-r^2/8) under N(r;0,v).
        variance = 1e-4
        expected_log_normalizer = (
            -0.5 * np.log1p(0.25 * variance)
            + 0.125 * variance / (1 + 0.25 * variance)
        )
        np.testing.assert_allclose(log_normalizers, expected_log_normalizer, atol=5e-7)
        print("Anchored pair masses:", probabilities, np.asarray(masses).tolist())
        print("Anchored log normalizers:", np.asarray(log_normalizers).tolist())
    print("Independent local assertions passed.")


def gaussian(log_mass, mean, covariance):
    """A normalized Gaussian site in its retained mean origin."""
    return (log_mass - (mean.shape[-1] * np.log(2 * np.pi)
                        + np.linalg.slogdet(covariance)[1]) / 2,
            np.zeros_like(mean), np.linalg.inv(covariance))


def factor(parameters, left_center, right_center):
    """Expand each independent stick and affine residual in local coordinates."""
    weights, offsets, xi, matrix, bias, noise = map(lambda x: np.asarray(x, float), parameters)
    card, _, dim = weights.shape
    constant = np.zeros((card, card))
    linear = np.zeros((card, card, 2 * dim))
    precision = np.zeros((card, card, 2 * dim, 2 * dim))
    for i in range(card):
        for j in range(card):
            for stick in range(min(j + 1, card - 1)):
                sign = 1 if j == stick else -1
                c = np.sqrt(xi[i, stick])
                omega = 0.25 if c == 0 else np.tanh(c / 2) / (2 * c)
                offset = offsets[i, stick] + weights[i, stick] @ left_center[i]
                constant[i, j] += (-np.logaddexp(c / 2, -c / 2) + omega * c * c / 2
                                   + sign * offset / 2 - omega * offset * offset / 2)
                linear[i, j, :dim] += (sign / 2 - omega * offset) * weights[i, stick]
                precision[i, j, :dim, :dim] += omega * np.outer(weights[i, stick], weights[i, stick])
            inverse = np.linalg.inv(noise[j])
            transform = np.concatenate((-matrix[j], np.eye(dim)), axis=1)
            offset = matrix[j] @ left_center[i] + bias[j] - right_center[j]
            constant[i, j] -= (dim * np.log(2 * np.pi) + np.linalg.slogdet(noise[j])[1]
                               + offset @ inverse @ offset) / 2
            linear[i, j] += transform.T @ inverse @ offset
            precision[i, j] += transform.T @ inverse @ transform
    return constant, linear, precision


def lift(left, right):
    card, dim = left[1].shape
    linear = np.zeros((card, card, 2 * dim))
    precision = np.zeros((card, card, 2 * dim, 2 * dim))
    linear[..., :dim], linear[..., dim:] = left[1][:, None], right[1][None]
    precision[..., :dim, :dim], precision[..., dim:, dim:] = left[2][:, None], right[2][None]
    return left[0][:, None] + right[0][None], linear, precision


def add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def normalize(natural):
    a, h, matrix = natural
    active = np.isfinite(a)
    covariance = np.linalg.inv(matrix)
    mean = np.einsum("...ab,...b->...a", covariance, h)
    scores = a + (h.shape[-1] * np.log(2 * np.pi) - np.linalg.slogdet(matrix)[1]
                  + np.einsum("...d,...d->...", h, mean)) / 2
    log_normalizer = np.logaddexp.reduce(scores.ravel())
    probability = np.exp(scores - log_normalizer)
    mean = np.where(active[..., None], mean, 0)
    covariance = np.where(active[..., None, None], covariance, 0)
    return probability, mean, covariance, log_normalizer


def project(region, side):
    p, m, v, _ = region
    dim = m.shape[-1] // 2
    selection = slice(side * dim, (side + 1) * dim)
    axis = 1 - side
    probability = p.sum(axis)
    conditional = p / np.expand_dims(np.where(probability > 0, probability, 1), axis)
    mean = (conditional[..., None] * m[..., selection]).sum(axis)
    delta = m[..., selection] - np.expand_dims(mean, axis)
    covariance = (conditional[..., None, None] *
                  (v[..., selection, selection] + delta[..., :, None] * delta[..., None, :])).sum(axis)
    return probability, mean, covariance


def quotient(projected, cavity):
    p, mean, covariance = projected
    active = p > 0
    safe_covariance = np.where(active[..., None, None], covariance, np.eye(mean.shape[-1]))
    precision = np.linalg.inv(safe_covariance)
    linear = np.einsum("...ab,...b->...a", precision, mean)
    constant = np.log(np.where(active, p, 1)) - (
        mean.shape[-1] * np.log(2 * np.pi) + np.linalg.slogdet(safe_covariance)[1]
        + np.einsum("...d,...d->...", mean, linear)) / 2
    values = [constant - np.where(np.isfinite(cavity[0]), cavity[0], 0),
              linear - cavity[1], precision - cavity[2]]
    values[0] -= values[0][np.flatnonzero(active)[0]]
    return tuple(np.where(active.reshape(active.shape + (1,) * rank), value,
                          -np.inf if rank == 0 else 0) for rank, value in enumerate(values))


def entropy(belief):
    p, mean, covariance, *_ = belief
    active = p > 0
    covariance = np.where(active[..., None, None], covariance, np.eye(mean.shape[-1]))
    return np.sum(np.where(active, p * ((mean.shape[-1] * (1 + np.log(2 * np.pi))
                                        + np.linalg.slogdet(covariance)[1]) / 2
                                       - np.log(np.where(active, p, 1))), 0))


def expectation(belief, natural):
    p, mean, covariance, *_ = belief
    a, h, matrix = natural
    values = np.where(p > 0, a, 0) + np.einsum("...d,...d->...", h, mean)
    values -= (np.einsum("...ab,...ab->...", matrix, covariance)
               + np.einsum("...a,...ab,...b->...", mean, matrix, mean)) / 2
    return np.sum(np.where(p > 0, p * values, 0))


def statistics(region, parameters, center):
    weights, offsets, xi, *_ = map(lambda x: np.asarray(x, float), parameters)
    p, mean, covariance, _ = region
    card, _, dim = weights.shape
    target, exposure = np.zeros((card, card - 1)), np.zeros((card, card - 1))
    linear = np.zeros((card, card - 1, dim + 1))
    precision = np.zeros((card, card - 1, dim + 1, dim + 1))
    for i in range(card):
        for stick in range(card - 1):
            c = np.sqrt(xi[i, stick])
            omega = 0.25 if c == 0 else np.tanh(c / 2) / (2 * c)
            for j in range(stick, card):
                m, v = mean[i, j, :dim], covariance[i, j, :dim, :dim]
                feature = np.r_[m, 1]
                moment = np.outer(feature, feature)
                moment[:dim, :dim] += v
                exposure[i, stick] += p[i, j]
                logit = weights[i, stick] @ m + (offsets[i, stick] + weights[i, stick] @ center[i])
                target[i, stick] += p[i, j] * (weights[i, stick] @ v @ weights[i, stick] + logit ** 2)
                linear[i, stick] += p[i, j] * (0.5 if j == stick else -0.5) * feature
                precision[i, stick] += omega * p[i, j] * moment
    target /= np.where(exposure > 0, exposure, 1)
    return target, xi - target, exposure, linear, precision


if __name__ == "__main__":
    main()
