"""Substitute observed equality coordinates in ordinary Gaussian factors.

A bound coordinate is absent from local entropy and integration. The value
still enters every adjacent normalized prior/conditional density. Unbound
members of the same batch use their ordinary messages, energies and statistics.
"""

import jax.numpy as jnp
import numpy as np

from cffg import efrules, gauss


def validate_point(value, dimension):
    from cffg.gaussian_kernels import validate_observation

    if np.iscomplexobj(value):
        raise ValueError("Gaussian tap requires a real observation")
    return validate_observation(value, dimension)


def _mv(matrix, vector):
    return (matrix @ vector[..., None])[..., 0]


def _choose(bound, yes, no):
    return jnp.where(bound.reshape((-1,) + (1,) * (no.ndim - 1)), yes, no)


def messages(group, outputs, p):
    """Eliminated roles carry vacuous messages; adjacent factors read the fixed value directly."""
    if group.type in ("gaussian", "equality"):
        return tuple(tuple(_choose(p["bound0"], 0., a) for a in pair) for pair in outputs)
    (forward_e, forward_l), (backward_e, backward_l) = outputs
    if "point1" in p:
        mean = _mv(p["A"], p["point1"]) + p["b"]
        forward_e = _choose(p["bound1"], _mv(p["W"], mean), forward_e)
        forward_l = _choose(p["bound1"], p["W"], forward_l)
    if "point0" in p:
        gain = p["A"].swapaxes(-2, -1) @ p["W"]
        backward_e = _choose(p["bound0"], _mv(gain, p["point0"] - p["b"]), backward_e)
        backward_l = _choose(p["bound0"], gain @ p["A"], backward_l)
        forward_e = _choose(p["bound0"], 0., forward_e)
        forward_l = _choose(p["bound0"], 0., forward_l)
    if "point1" in p:
        backward_e = _choose(p["bound1"], 0., backward_e)
        backward_l = _choose(p["bound1"], 0., backward_l)
    return (forward_e, forward_l), (backward_e, backward_l)


def _negative_log_density(value, mean, precision):
    residual = value - mean
    d = residual.shape[-1]
    nll = (d * efrules.LOG2PI - efrules._pd_logdet(precision)
           + (residual * _mv(precision, residual)).sum(axis=-1)) / 2
    return jnp.where(gauss.is_proper_precision(precision) & jnp.isfinite(nll), nll, jnp.inf)


def energy(group, ordinary, eta, precision, p, observation, mask):
    if group.type == "equality":
        return jnp.where(p["bound0"], 0., ordinary)
    if group.type == "gaussian":
        return jnp.where(p["bound0"], _negative_log_density(p["point0"], p["m"], p["W"]), ordinary)

    do, di = group.dims
    if "point0" in p:
        observation, mask = p["point0"], p["bound0"]
        fixed_out = gauss.linear_observed_energy(
            eta[:, 1, :di], precision[:, 1, :di, :di], observation, jnp.ones_like(mask),
            p["A"], p["b"], p["W"],
        )
        ordinary = jnp.where(mask, fixed_out, ordinary)
    if "point1" in p:
        shift = _mv(p["A"], p["point1"]) + p["b"]
        fixed_in = gauss.prior_energy(eta[:, 0, :do], precision[:, 0, :do, :do], shift, p["W"])
        if observation is not None:
            fixed_in = jnp.where(mask != 0, _negative_log_density(observation, shift, p["W"]), fixed_in)
        ordinary = jnp.where(p["bound1"], fixed_in, ordinary)
    return ordinary


def statistics(group, values, eta, precision, p, observation, mask):
    """Moments of a joint with a known coordinate have zero covariance in that coordinate."""
    from cffg.gaussian_kernels import linear_statistics

    if group.type == "gaussian":
        bound = np.asarray(p["bound0"]) & np.asarray(gauss.is_proper_precision(p["W"]))
        mean = np.asarray(p["point0"])[bound]
        values["mean"][bound] = mean
        values["second"][bound] = mean[..., None] * mean[:, None, :]
        values["proper"][bound] = True
        return values

    if "point0" in p:
        bound = np.asarray(p["bound0"])
        fixed = linear_statistics(eta, precision, p, group.dims, 0, p["point0"], np.ones(len(bound)))
        for name in values:
            values[name][bound] = fixed[name][bound]
        observation, mask = p["point0"], bound
    if "point1" in p:
        out_dim = group.dims[0]
        for i in np.flatnonzero(np.asarray(p["bound1"])):
            x = np.asarray(p["point1"][i])
            a, b, w = (np.asarray(p[k][i]) for k in ("A", "b", "W"))
            proper = bool(gauss.is_proper_precision(w))
            if observation is not None and mask[i]:
                y, cov = np.asarray(observation[i]), np.zeros_like(w)
            else:
                l = w + precision[i, 0, :out_dim, :out_dim]
                h = w @ (a @ x + b) + eta[i, 0, :out_dim]
                proper &= bool(gauss.is_proper_precision(l))
                y, cov = map(np.asarray, gauss.moments(h, l))
            xt = np.append(x, 1)
            values["Syy"][i] = cov + np.outer(y, y) if proper else np.nan
            values["Syx"][i] = np.outer(y, xt) if proper else np.nan
            values["Sxx"][i] = np.outer(xt, xt) if proper else np.nan
            values["proper"][i] = proper
    return values
