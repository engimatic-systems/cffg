"""Adaptive Gaussian approximations with explicit frozen-factor node energy.

For an input cavity N(m,P), Taylor uses A=Dg(m), b=g(m)-Am, Omega=V.
The scaled unscented rule fixes alpha=1, beta=2, kappa=0: points are m
and m±sqrt(d)*chol(P)[:,i], mean weights 0 and 1/(2d), covariance weights
2 and 1/(2d). Regression uses A=Cov(g(X),X) P^-1. Omega adds V to a sum
of residual outer products, avoiding cancellation and any covariance ridge.

The tilted belief is proportional to N(y;Ax+b,Omega) times both cavities.
Its marginal natural parameters minus each cavity define replacement sites.
Sites are unnormalized and may be indefinite; required beliefs must be proper.

Reported energy is E_q[log q-log N(y;Ax+b,Omega)], including normalization
constants. It is a Gaussian Bethe node term conditional on frozen A,b,Omega.
Adaptive fixed points additionally require these coefficients to agree with
the current cavity. They need not be stationary points of the composed energy.
Differentiation of an adaptive inference root must include this dependence.
"""

from enum import IntEnum

import jax
import jax.numpy as jnp

from cffg import gauss


class NonlinearFailure(IntEnum):
    NONE = 0
    VACUOUS_INPUT = 1
    IMPROPER_CAVITY = 2
    INVALID_NOISE = 3
    NONFINITE_FUNCTION = 4
    IMPROPER_TILT = 5
    UNREPRESENTABLE_SIGMA_POINTS = 6


def _symmetric(matrix):
    return (matrix + matrix.T) / 2


def _inverse(matrix):
    return _symmetric(jnp.linalg.solve(matrix, jnp.eye(matrix.shape[0], dtype=matrix.dtype)))


def _sigma_geometry(mean, covariance):
    dimension = mean.shape[0]
    offsets = jnp.sqrt(jnp.asarray(dimension, dtype=mean.dtype)) * jnp.linalg.cholesky(covariance).T
    centered = jnp.concatenate((jnp.zeros_like(mean)[None], offsets, -offsets))
    points = mean + centered
    # Rounding must not erase the spread used to represent the input covariance.
    tolerance = 64 * dimension * jnp.finfo(mean.dtype).eps * jnp.max(jnp.abs(centered))
    represented = jnp.max(jnp.abs((points - mean) - centered)) <= tolerance
    return points, centered, represented


def approximation(evaluate, mean, covariance, noise, parameters, method):
    """Return the affine conditional matching the declared cavity approximation."""
    value = evaluate(mean, parameters)
    if method == "linearization":
        matrix = jax.jacfwd(evaluate, argnums=0)(mean, parameters)
        return matrix, value - matrix @ mean, noise
    if method != "sigma_point":
        raise ValueError(f"unknown nonlinear method {method!r}")
    dimension = mean.shape[0]
    points, centered, represented = _sigma_geometry(mean, covariance)
    values = jax.vmap(evaluate, in_axes=(0, None))(points, parameters)
    weights = jnp.concatenate((jnp.zeros(1, mean.dtype),
                               jnp.full((2 * dimension,), 1 / (2 * dimension), mean.dtype)))
    transformed = weights @ values
    cross = jnp.einsum("n,ni,nj->ij", weights, centered, values - transformed)
    matrix = jnp.linalg.solve(covariance, cross).T
    residual = values - transformed - centered @ matrix.T
    # Nonnegative outer-product weights preserve the regression covariance.
    correction = jnp.einsum("n,ni,nj->ij", weights, residual, residual)
    correction += 2 * jnp.outer(residual[0], residual[0])
    return tuple(jnp.where(represented, v, jnp.nan) for v in (
        matrix, transformed - matrix @ mean, _symmetric(noise + correction),
    ))


def _empty(di, do, dtype, reason, *, vacuous):
    zero = lambda shape: jnp.zeros(shape, dtype)
    bad = lambda shape: jnp.full(shape, jnp.nan, dtype)
    message = zero if vacuous else bad
    return {
        "eta_in": message((di,)), "precision_in": message((di, di)),
        "eta_out": message((do,)), "precision_out": message((do, do)),
        "mean_in": bad((di,)), "covariance_in": bad((di, di)),
        "mean_out": bad((do,)), "covariance_out": bad((do, do)),
        "energy": jnp.asarray(jnp.inf, dtype), "proper": jnp.asarray(False),
        "status": jnp.asarray(0 if vacuous else 2, jnp.int32),
        "failure_code": jnp.asarray(reason, jnp.int32),
    }


def _tilted(ei, li, eo, lo, matrix, offset, noise, observation, observed):
    di, do = ei.shape[0], eo.shape[0]
    dtype = ei.dtype
    w = _inverse(noise)
    if observed:
        precision = li + matrix.T @ w @ matrix
        eta = ei + matrix.T @ w @ (observation - offset)
    else:
        precision = jnp.block([[li + matrix.T @ w @ matrix, -matrix.T @ w],
                               [-w @ matrix, lo + w]])
        eta = jnp.concatenate((ei - matrix.T @ w @ offset, eo + w @ offset))
    proper = gauss.is_proper_precision(precision) & jnp.all(jnp.isfinite(eta))
    safe = jnp.where(proper, precision, jnp.eye(precision.shape[0], dtype=dtype))
    covariance = _inverse(safe)
    mean = jnp.linalg.solve(safe, eta)
    mx, px = mean[:di], covariance[:di, :di]
    if observed:
        my, py = observation, jnp.zeros((do, do), dtype)
        residual = observation - matrix @ mx - offset
        residual_covariance = matrix @ px @ matrix.T
    else:
        my, py = mean[di:], covariance[di:, di:]
        transform = jnp.concatenate((-matrix, jnp.eye(do, dtype=dtype)), axis=1)
        residual = transform @ mean - offset
        residual_covariance = transform @ covariance @ transform.T
    log2pi = jnp.log(jnp.asarray(2 * jnp.pi, dtype))
    energy = 0.5 * (
        jnp.linalg.slogdet(safe)[1] - mean.shape[0] * (1 + log2pi)
        + do * log2pi + jnp.linalg.slogdet(noise)[1]
        + residual @ w @ residual + jnp.trace(w @ residual_covariance)
    )
    cavity_covariance = _inverse(li)
    cavity_mean = jnp.linalg.solve(li, ei)
    predicted_covariance = noise + matrix @ cavity_covariance @ matrix.T
    out_precision = _inverse(predicted_covariance)
    out_eta = out_precision @ (matrix @ cavity_mean + offset)
    if observed:
        in_precision = matrix.T @ w @ matrix
        in_eta = matrix.T @ w @ (observation - offset)
    else:
        # Schur elimination equals cavity subtraction without cancelling a strong prior.
        effective = _symmetric(w @ jnp.linalg.solve(w + lo, lo))
        in_precision = matrix.T @ effective @ matrix
        in_eta = matrix.T @ w @ jnp.linalg.solve(w + lo, eo - lo @ offset)
    proper &= gauss.is_proper_precision(px)
    if not observed:
        proper &= gauss.is_proper_precision(py)
    proper &= gauss.is_proper_precision(predicted_covariance)
    for value in (energy, mx, my, in_eta, in_precision, out_eta, out_precision):
        proper &= jnp.all(jnp.isfinite(value))
    result = {
        "eta_in": in_eta, "precision_in": _symmetric(in_precision),
        "eta_out": out_eta, "precision_out": out_precision,
        "mean_in": mx, "covariance_in": px,
        "mean_out": my, "covariance_out": py,
        "energy": energy, "proper": proper,
        "status": jnp.asarray(0, jnp.int32),
        "failure_code": jnp.asarray(NonlinearFailure.NONE, jnp.int32),
    }
    failure = _empty(di, do, dtype, NonlinearFailure.IMPROPER_TILT, vacuous=False)
    return jax.tree.map(lambda good, bad: jnp.where(proper, good, bad), result, failure)


def local_result(evaluate, ei, li, eo, lo, parameters, noise, method,
                 observation=None, mask=None):
    """One local projected update; numerical facts remain separate from derivatives.

    No approximation point is selected for a vacuous input. Invalid required
    matrices expose non-finite energy and a machine-readable failure code.
    Output cavities may be indefinite when the full tilted belief is proper.
    """
    di, do, dtype = ei.shape[0], eo.shape[0], ei.dtype
    vacuous = jnp.all(ei == 0) & jnp.all(li == 0)
    proper_input = gauss.is_proper_precision(li) & jnp.all(jnp.isfinite(ei))
    proper_noise = gauss.is_proper_precision(noise)
    output_finite = jnp.all(jnp.isfinite(eo)) & jnp.all(jnp.isfinite(lo))
    parameters_finite = jnp.all(jnp.isfinite(parameters))
    # Symmetry is checked on the original cavity, before canonical construction.
    scale = jnp.max(jnp.abs(lo))
    symmetric_output = jnp.max(jnp.abs(lo - lo.T)) <= 16 * do * jnp.finfo(dtype).eps * scale
    valid = proper_input & proper_noise & output_finite & symmetric_output & parameters_finite

    def active(_):
        mean, covariance = jnp.linalg.solve(li, ei), _inverse(li)
        matrix, offset, omega = approximation(evaluate, mean, covariance, noise, parameters, method)
        reason = jnp.asarray(NonlinearFailure.NONFINITE_FUNCTION)
        if method == "sigma_point":
            represented = _sigma_geometry(mean, covariance)[2]
            reason = jnp.where(represented, reason, NonlinearFailure.UNREPRESENTABLE_SIGMA_POINTS)
        finite = jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(offset))
        finite &= gauss.is_proper_precision(omega)

        def evaluate_tilt(_):
            if observation is None:
                return _tilted(ei, li, eo, lo, matrix, offset, omega, None, False)
            return jax.lax.cond(
                mask > 0,
                lambda _: _tilted(ei, li, eo, lo, matrix, offset, omega, observation, True),
                lambda _: _tilted(ei, li, eo, lo, matrix, offset, omega, None, False),
                operand=None,
            )

        return jax.lax.cond(finite, evaluate_tilt,
                            lambda _: _empty(di, do, dtype, reason,
                                             vacuous=False), operand=None)

    reason = jnp.where(proper_noise, NonlinearFailure.IMPROPER_CAVITY, NonlinearFailure.INVALID_NOISE)
    reason = jnp.where(parameters_finite, reason, NonlinearFailure.NONFINITE_FUNCTION)
    invalid = lambda _: _empty(di, do, dtype, reason, vacuous=False)
    return jax.lax.cond(
        vacuous & proper_noise & output_finite & symmetric_output & parameters_finite,
        lambda _: _empty(di, do, dtype, NonlinearFailure.VACUOUS_INPUT,
                                               vacuous=True),
        lambda _: jax.lax.cond(valid, active, invalid, operand=None), operand=None,
    )
