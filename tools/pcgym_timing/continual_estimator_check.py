"""Synthetic checks for one-step RK4, likelihood, scalar update and grid enumeration."""

import json
from math import exp, log, pi, sqrt

import jax
import numpy as np

from pcgym_timing import continual_estimator as estimator


def scalar_step(obs, command, ua):
    """Independent physical-unit RK4 for the retained reactor on the operating range."""
    ca, temp = (obs + 1) * np.array([.15, 25.]) + [0.7, 300.]
    coolant = 298.5 + 3.5 * command
    h = 26 / 60 / 4

    def rhs(c, t):
        reaction = 7.2e10 * exp(-8750 / t) * c
        return (1 - c - reaction,
                350 - t + 50_000. / 239 * reaction + ua / 23_900. * (coolant - t))

    for _ in range(4):
        c1, t1 = rhs(ca, temp)
        c2, t2 = rhs(ca + h*c1/2, temp + h*t1/2)
        c3, t3 = rhs(ca + h*c2/2, temp + h*t2/2)
        c4, t4 = rhs(ca + h*c3, temp + h*t3)
        ca += h * (c1 + 2*c2 + 2*c3 + c4) / 6
        temp += h * (t1 + 2*t2 + 2*t3 + t4) / 6
    return np.array([(ca - .85) / .15, (temp - 325.) / 25.])


def scalar_likelihood(obs, command, ua):
    """Finite-difference sensitivities and variances computed in physical sensor units."""
    h = np.cbrt(np.finfo(float).eps)
    prediction = scalar_step(obs, command, ua)
    state_error = []
    for axis, (lo, scale) in enumerate(((.7, .15), (300., 25.))):
        offset = np.eye(2)[axis] * h
        dz = (scalar_step(obs + offset, command, ua)[1] - scalar_step(obs - offset, command, ua)[1]) / (2*h)
        sensor_sd = .001 * ((obs[axis] + 1)*scale + lo) / scale
        state_error.append(dz * sensor_sd)
    sensor_sd = .001 * (prediction[1] * 25 + 325) / 25
    variance = sensor_sd**2 + np.dot(state_error, state_error)
    slope = (scalar_step(obs, command, ua * exp(h))[1] - scalar_step(obs, command, ua * exp(-h))[1]) / (2*h)
    return prediction, variance, slope


def check():
    if not jax.config.x64_enabled:
        raise RuntimeError("Run the independent finite-difference comparison with JAX_ENABLE_X64=1")
    ekf, bank = estimator.Learner(), estimator.GridReference()
    assert ekf.ua == bank.ua == 50_000.
    json.dumps(estimator.method_config(), allow_nan=False)
    mesh = np.arange(40_000., 60_001., 250.)
    posterior = np.exp(-.5 * ((mesh - 50_000.) / 5_000.)**2)
    posterior /= posterior.sum()
    transition = []
    for previous in mesh:
        p = np.exp(-.5 * ((mesh - previous) / (.012 * previous))**2)
        transition.append(.998 * p / sum(p) + .002 / len(mesh))
    assert np.all(np.array(transition) > 0)
    np.testing.assert_allclose(np.sum(transition, axis=1), 1, atol=3e-15)

    z, var = 0., .10**2
    examples = [
        ([-.27, .18], .4, 47_350., .003),
        ([.13, -.08], -.5, 51_430., -.002),
    ]
    for n, (obs, u, value, innovation) in enumerate(examples, 1):
        obs = np.array(obs)
        following = scalar_step(obs, u, value)
        np.testing.assert_allclose(estimator.predict(obs, [u], value), following, rtol=0, atol=1e-13)
        following[1] += innovation

        # Check the EKF against a direct scalar Gaussian conditioning calculation.
        mean, noise, slope = scalar_likelihood(obs, u, 50_000.*exp(z))
        prior = var + .012**2
        total = noise + prior * slope**2
        residual = following[1] - mean[1]
        z = np.clip(z + residual * prior * slope / total, log(.8), log(1.2))
        var = 1 / (1/prior + slope**2/noise)
        score = log(1 / sqrt(2*pi*total)) - residual**2 / (2*total)
        update = ekf.update(obs, u, following)
        np.testing.assert_allclose(
            [update["temperature_prediction"], update["conditional_temperature_variance"],
             update["temperature_log_ua_derivative"], update["log_ua_variance"],
             update["log_predictive_density"], update["ua_after"]],
            [mean[1], noise, slope, var, score, 50_000.*exp(z)], rtol=2e-8, atol=1e-12)

        # Sum every previous/current candidate pair directly in probability space.
        prior_grid = [sum(posterior[i]*row[j] for i, row in enumerate(transition)) for j in range(len(mesh))]
        joint = []
        for candidate, probability in zip(mesh, prior_grid, strict=True):
            expected, variance, _ = scalar_likelihood(obs, u, candidate)
            density = exp(-.5 * (following[1] - expected[1])**2 / variance) / sqrt(2*pi*variance)
            joint.append(probability * density)
        evidence = sum(joint)
        posterior = np.array(joint) / evidence
        update_grid = bank.update(obs, [u], following)
        json.dumps([update, update_grid], allow_nan=False)
        np.testing.assert_allclose(np.exp(bank._log_weights), posterior, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(update_grid["log_predictive_density"], log(evidence), rtol=1e-8)
        np.testing.assert_allclose(bank.ua, posterior @ mesh, rtol=1e-10)
        assert update["n_updates"] == update_grid["n_updates"] == n

    # A new concentration cannot contribute evidence until it becomes the next pair's input.
    for method in (estimator.Learner, estimator.GridReference):
        first, other = method(), method()
        assert first.update(obs, u, following) == other.update(obs, u, following + [.5, 0])
        before = (first.ua, first.n_updates)
        for invalid in ([np.nan, .1], [0., .1, .3]):
            try:
                first.update(obs, u, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid observation accepted")
        assert before == (first.ua, first.n_updates)

    # Contradictory measurements stress the bounded location and the grid's log support.
    ekf = estimator.Learner()
    extreme = ekf.update(obs, u, [0., 10.])
    np.testing.assert_allclose(extreme["ua_after"], 40_000.)
    bank.update(obs, u, [0., 10.])
    assert np.isfinite(bank._log_weights).all() and np.min(bank._log_weights) < log(np.finfo(float).tiny)
    least = np.argmin(bank._log_weights)
    bank.update(obs, u, scalar_step(obs, u, mesh[least]))
    assert np.isfinite(bank._log_weights).all() and np.exp(bank._log_weights[least]) > 0
    np.testing.assert_allclose(np.exp(bank._log_weights).sum(), 1.)
    print("Passed independent one-step, noise propagation, scalar EKF and 81-candidate enumeration checks.")


if __name__ == "__main__":
    check()
