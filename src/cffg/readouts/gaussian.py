"""gaussian readouts over explicit structure and current numerical inputs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from cffg import ef, gauss, mixg
from cffg.execution.transport import _gauss_rows, _obs_rows
from cffg.kernel_runtime import gaussian_readout as _gauss_readout
from cffg.kernel_runtime import parameters as _kernel_params
from cffg.readouts.types import (
    PairBeliefs,
)
from cffg.readouts.views import (
    GaussianStatisticsView,
    GaussianView,
    MixtureView,
)
from cffg.runtime_types import State


def gaussian_marginals(
    view: GaussianView, state: State
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Edge name -> (mean, cov). Requires a proper combined precision —
    run to coherence (and bind data) first. Improper beliefs return NaN
    mean/cov arrays, an explicit readout that cannot resemble a valid
    negative or huge covariance."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    eta, lam = np.asarray(state.g_eta), np.asarray(state.g_lam)
    for e in view.edges.values():
        if not e.dim:
            continue
        d = e.dim
        eta_q, lam_q = ef.GAUSSIAN.product(
            (
                eta[view.g_eta_indices(e.slot0, d)],
                lam[view.g_lam_indices(e.slot0, d)],
            ),
            (
                eta[view.g_eta_indices(e.slot1, d)],
                lam[view.g_lam_indices(e.slot1, d)],
            ),
        )
        mean, cov = gauss.moments(eta_q, lam_q)
        out[e.name] = (np.asarray(mean), np.asarray(cov))
    return out


def gaussian_node_stats(
    view: GaussianStatisticsView, state: State, *, pair_beliefs: PairBeliefs
) -> dict[str, dict[str, np.ndarray]]:
    """Factor name -> expected sufficient statistics of the node belief
    (cff-7c5o, D4): the continuous analog of node_beliefs, feeding the
    NIW/MNIW conjugate updates in cffg.conj.

    gaussian:            {"mean" E[x], "second" E[xxᵀ]}
    gaussian_linear:     {"Syy" E[yyᵀ], "Syx" E[yx̃ᵀ], "Sxx" E[x̃x̃ᵀ]}
                         with x̃ = [x; 1] (offset column absorbed)
    gaussian_linear_obs: same keys with y fixed at the bound observation,
                         plus "mask" (0.0 = missing: skip the occurrence).
    gaussian_mixture_linear: soft ``responsibility`` plus responsibility-
                         weighted ``Syy``, ``Syx``, and ``Sxx``.
    gaussian_mixture: soft ``responsibility``, ``weighted_mean``, and
                         ``weighted_second``.
    recurrence_link: normalized source/destination pair, its marginals and
                         literal support, plus shared D9 masses and direct
                         continuous drive moments.
    Every record includes ``proper``. Improper beliefs or mutated non-PD
    node precisions return ``proper=False`` with NaN sufficient statistics;
    equality nodes are skipped."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for g in view.gauss_groups:
        if g.kernel is None or g.kernel.binding.statistics is None:
            continue
        inc_e, inc_l = _gauss_rows(view, state.g_eta, state.g_lam, g.gather, g.dims)
        y = mk = None
        if g.obs_rows is not None:
            y = _obs_rows(view, state.g_obs, g.obs_rows, g.observed_dim)
            mk = state.g_mask[jnp.asarray(g.obs_rows)]
        values = _gauss_readout(
            g,
            inc_e,
            inc_l,
            _kernel_params(g, state.params),
            y,
            mk,
            statistics=True,
        )
        for i, name in enumerate(g.members):
            out[name] = {key: np.asarray(value)[i] for key, value in values.items()}

    for g in view.mix_groups:
        mu_e = state.messages[view.message_indices(g.s_gather, g.k_sel)]
        inc_e, inc_l = _gauss_rows(
            view,
            state.g_eta,
            state.g_lam,
            g.g_gather,
            g.dims,
        )
        w = state.params[g.name + ":W"]
        if g.type == "gaussian_mixture_linear":
            do, di = g.dims
            stats = mixg.mixlin_stats(
                inc_e[:, 0, :do],
                inc_l[:, 0, :do, :do],
                inc_e[:, 1, :di],
                inc_l[:, 1, :di, :di],
                mu_e,
                state.params[g.name + ":A"],
                state.params[g.name + ":b"],
                w,
            )
            for i, name in enumerate(g.members):
                out[name] = {
                    "responsibility": np.asarray(stats.responsibility[i]),
                    "Syy": np.asarray(stats.syy[i]),
                    "Syx": np.asarray(stats.syx[i]),
                    "Sxx": np.asarray(stats.sxx[i]),
                    "proper": np.asarray(stats.proper[i]),
                }
        else:
            (do,) = g.dims
            stats = mixg.mix_stats(
                inc_e[:, 0, :do],
                inc_l[:, 0, :do, :do],
                mu_e,
                state.params[g.name + ":m"],
                w,
            )
            for i, name in enumerate(g.members):
                out[name] = {
                    "responsibility": np.asarray(stats.responsibility[i]),
                    "weighted_mean": np.asarray(stats.weighted_mean[i]),
                    "weighted_second": np.asarray(stats.weighted_second[i]),
                    "proper": np.asarray(stats.proper[i]),
                }

    for name, pair in pair_beliefs(state).items():
        out[name] = {
            **pair,
            "out_probability": pair["destination_probability"],
            "in_probability": pair["source_probability"],
            "out_support": pair["destination_support"],
            "in_support": pair["source_support"],
        }
    return out


def gaussian_mixture_predictives(
    view: MixtureView, state: State
) -> dict[str, dict[str, np.ndarray]]:
    """Factor-to-output cavity predictives for Gaussian-mixture nodes.

    Each record exposes selector weights, component moments, collapsed
    moments, and ``proper``. The readout deliberately excludes the
    output-facing message and therefore remains suitable for one-step
    prediction after an observation has updated that edge.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for g in view.mix_groups:
        mu_e = state.messages[view.message_indices(g.s_gather, g.k_sel)]
        w = state.params[g.name + ":W"]
        if g.type == "gaussian_mixture_linear":
            _, di = g.dims
            inc_e, inc_l = _gauss_rows(
                view,
                state.g_eta,
                state.g_lam,
                g.g_gather,
                g.dims,
            )
            predictive = mixg.mixlin_predictive(
                inc_e[:, 1, :di],
                inc_l[:, 1, :di, :di],
                mu_e,
                state.params[g.name + ":A"],
                state.params[g.name + ":b"],
                w,
            )
        else:
            predictive = mixg.mix_predictive(
                mu_e,
                state.params[g.name + ":m"],
                w,
            )
        for i, name in enumerate(g.members):
            out[name] = {
                "weight": np.asarray(predictive.weight[i]),
                "component_mean": np.asarray(predictive.component_mean[i]),
                "component_covariance": np.asarray(predictive.component_covariance[i]),
                "mean": np.asarray(predictive.mean[i]),
                "covariance": np.asarray(predictive.covariance[i]),
                "proper": np.asarray(predictive.proper[i]),
            }
    return out
