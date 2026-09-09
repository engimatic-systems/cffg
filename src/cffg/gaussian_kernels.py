"""Registered adapters for Gaussian priors and linear conditionals.

The message and energy formulas stay in gauss.py. Host sufficient-statistic
operations preserve the original per-member NumPy/JAX evaluation order.
"""

from __future__ import annotations

import numpy as np

from cffg.core.ir import FACTOR_TYPES
from cffg.kernel_bindings import GaussianBinding, GaussianConditioning, ParameterSpec


def _pack(parameters):
    def pack(template, members, resolved):
        out = {spec.key: [] for spec in parameters.values()}
        for name in members:
            factor = template.factors[name]
            for role, spec in parameters.items():
                value = resolved[factor.param_blobs[role]]
                if spec.transform == "covariance_to_precision":
                    value = np.linalg.inv(value)
                out[spec.key].append(value)
        return {key: np.stack(rows) for key, rows in out.items()}

    return pack


def prior_message(eta, precision, params, dims, observed_role, observation, mask):
    import jax.numpy as jnp

    from cffg import gauss

    w, m = params["W"], params["m"]
    return (gauss.prior_message(jnp.einsum("mij,mj->mi", w, m), w),)


def prior_energy(eta, precision, params, dims, observed_role, observation, mask):
    from cffg import gauss

    (d,) = dims
    return gauss.prior_energy(
        eta[:, 0, :d], precision[:, 0, :d, :d], params["m"], params["W"]
    )


def linear_message(eta, precision, params, dims, observed_role, observation, mask):
    from cffg import gauss

    do, di = dims
    a, b, w = params["A"], params["b"], params["W"]
    forward = gauss.linear_forward(eta[:, 1, :di], precision[:, 1, :di, :di], a, b, w)
    if observed_role is not None:
        backward = gauss.linear_observed_backward(observation, mask, a, b, w)
    else:
        backward = gauss.linear_backward(
            eta[:, 0, :do], precision[:, 0, :do, :do], a, b, w
        )
    return forward, backward


def linear_energy(eta, precision, params, dims, observed_role, observation, mask):
    from cffg import gauss

    do, di = dims
    a, b, w = params["A"], params["b"], params["W"]
    ei, li = eta[:, 1, :di], precision[:, 1, :di, :di]
    if observed_role is not None:
        return gauss.linear_observed_energy(ei, li, observation, mask, a, b, w)
    return gauss.linear_energy(
        eta[:, 0, :do], precision[:, 0, :do, :do], ei, li, a, b, w
    )


def validate_observation(value, dimension):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != dimension:
        raise ValueError(f"observation has dim {arr.shape[0]}, edge has {dimension}")
    return arr


def _stack_stats(rows):
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def prior_statistics(eta, precision, params, dims, observed_role, observation, mask):
    from cffg import gauss

    eta, precision = np.asarray(eta), np.asarray(precision)
    (d,) = dims
    rows = []
    for i in range(eta.shape[0]):
        w, m_blob = np.asarray(params["W"][i]), np.asarray(params["m"][i])
        lam_q = w + precision[i, 0, :d, :d]
        proper = bool(gauss.is_proper_precision(w)) and bool(
            gauss.is_proper_precision(lam_q)
        )
        m_q, sig_q = gauss.moments(w @ m_blob + eta[i, 0, :d], lam_q)
        m_q, sig_q = np.asarray(m_q), np.asarray(sig_q)
        if not proper:
            m_q, sig_q = np.full_like(m_q, np.nan), np.full_like(sig_q, np.nan)
        rows.append(
            {
                "mean": m_q,
                "second": sig_q + np.outer(m_q, m_q),
                "proper": np.asarray(proper),
            }
        )
    return _stack_stats(rows)


def linear_statistics(eta, precision, params, dims, observed_role, observation, mask):
    from cffg import gauss

    eta, precision = np.asarray(eta), np.asarray(precision)
    do, di = dims
    rows = []
    for i in range(eta.shape[0]):
        w, a, b = (np.asarray(params[key][i]) for key in ("W", "A", "b"))
        ei, li = eta[i, 1, :di], precision[i, 1, :di, :di]
        if observed_role is None:
            eo, lo = eta[i, 0, :do], precision[i, 0, :do, :do]
            lam_j = np.block([[w + lo, -w @ a], [-a.T @ w, a.T @ w @ a + li]])
            eta_j = np.concatenate([w @ b + eo, -a.T @ w @ b + ei])
            proper = bool(gauss.is_proper_precision(w)) and bool(
                gauss.is_proper_precision(lam_j)
            )
            m_j, sig_j = gauss.moments(eta_j, lam_j)
            m_j, sig_j = np.asarray(m_j), np.asarray(sig_j)
            if not proper:
                m_j, sig_j = np.full_like(m_j, np.nan), np.full_like(sig_j, np.nan)
            my, mx = m_j[:do], m_j[do:]
            syy = sig_j[:do, :do] + np.outer(my, my)
            syx = sig_j[:do, do:] + np.outer(my, mx)
            sxx = sig_j[do:, do:] + np.outer(mx, mx)
            syx_aug = np.concatenate([syx, my[:, None]], axis=1)
            sxx_aug = np.block([[sxx, mx[:, None]], [mx[None, :], np.ones((1, 1))]])
            if not proper:
                syy, syx_aug, sxx_aug = (
                    np.full_like(x, np.nan) for x in (syy, syx_aug, sxx_aug)
                )
            rows.append(
                {
                    "Syy": syy,
                    "Syx": syx_aug,
                    "Sxx": sxx_aug,
                    "proper": np.asarray(proper),
                }
            )
        else:
            y, mk = np.asarray(observation[i]), np.asarray(mask[i])
            lam_q = mk * (a.T @ w @ a) + li
            proper = bool(gauss.is_proper_precision(w)) and bool(
                gauss.is_proper_precision(lam_q)
            )
            mx, sig_q = gauss.moments(mk * (a.T @ (w @ (y - b))) + ei, lam_q)
            mx, sig_q = np.asarray(mx), np.asarray(sig_q)
            if not proper:
                mx, sig_q = np.full_like(mx, np.nan), np.full_like(sig_q, np.nan)
            sxx = sig_q + np.outer(mx, mx)
            syx_aug = np.concatenate([np.outer(y, mx), y[:, None]], axis=1)
            sxx_aug = np.block([[sxx, mx[:, None]], [mx[None, :], np.ones((1, 1))]])
            if not proper:
                syx_aug, sxx_aug = (
                    np.full_like(syx_aug, np.nan),
                    np.full_like(sxx_aug, np.nan),
                )
            rows.append(
                {
                    "Syy": np.outer(y, y) if proper else np.full((do, do), np.nan),
                    "Syx": syx_aug,
                    "Sxx": sxx_aug,
                    "mask": np.asarray(mk),
                    "proper": np.asarray(proper),
                }
            )
    return _stack_stats(rows)


def install(register, kernel, hostile):
    prior = {
        "mean": ParameterSpec(lambda s: (s["out"],), "m"),
        "cov": ParameterSpec(
            lambda s: (s["out"], s["out"]), "W", "covariance_to_precision"
        ),
    }
    linear = {
        "matrix": ParameterSpec(lambda s: (s["out"], s["in"]), "A"),
        "offset": ParameterSpec(lambda s: (s["out"],), "b"),
        "cov": ParameterSpec(
            lambda s: (s["out"], s["out"]), "W", "covariance_to_precision"
        ),
    }
    for name, specs, message, energy, stats, conditioning in (
        ("gaussian", prior, prior_message, prior_energy, prior_statistics, None),
        (
            "gaussian_linear",
            linear,
            linear_message,
            linear_energy,
            linear_statistics,
            GaussianConditioning("out", validate_observation, "gaussian_linear_obs"),
        ),
    ):
        register(
            kernel(
                FACTOR_TYPES[name],
                f"cffg.gaussian.{name}",
                "1.0.0",
                hostile=hostile,
                binding=GaussianBinding(
                    FACTOR_TYPES[name].edge_roles,
                    specs,
                    tuple(spec.key for spec in specs.values()),
                    _pack(specs),
                    message,
                    energy,
                    conditioning,
                    stats,
                ),
            )
        )
