"""gaussian readouts over explicit structure and current numerical inputs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from cffg import mixg
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


def _resolved_correlation(matrix):
    """Check positive definiteness at the stored precision, independently of coordinate units."""
    scale = np.sqrt(matrix.diagonal().astype(np.float64))
    correlation = (matrix / scale[:, None]) / scale[None, :]
    if not np.isfinite(correlation).all():
        raise FloatingPointError
    dim = len(scale)
    eps = np.finfo(matrix.dtype).eps
    asymmetry = np.max(np.abs(correlation - correlation.T))
    if asymmetry > 16 * dim * eps * np.max(np.abs(correlation)):
        raise np.linalg.LinAlgError
    correlation = (correlation + correlation.T) / 2
    # A Cholesky pivot subtracts up to dim products from a unit diagonal.
    # Reserve that source-dtype roundoff for admission; solve the unshifted system.
    guarded = correlation.copy()
    np.fill_diagonal(guarded, 1 - dim * eps)
    np.linalg.cholesky(guarded)
    return correlation, scale


def _edge_moments(view, edge, eta, precision):
    dim = edge.dim
    slots = (edge.slot0, edge.slot1)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            h = eta[view.g_eta_indices(slots, dim)].sum(axis=0)
            p = precision[view.g_lam_indices(slots, dim)].sum(axis=0)
            if not np.isfinite(h).all():
                raise FloatingPointError
            correlation, scale = _resolved_correlation(p)
            chol = np.linalg.cholesky(correlation)
            rhs = np.column_stack((h / scale, np.eye(dim)))
            solved = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs)) / scale[:, None]
            mean = solved[:, 0].astype(eta.dtype)
            covariance = (solved[:, 1:] / scale).astype(precision.dtype)
            _resolved_correlation(covariance)
            if np.isfinite(mean).all():
                return mean, covariance
    except (np.linalg.LinAlgError, FloatingPointError):
        pass
    return np.full(dim, np.nan, dtype=eta.dtype), np.full((dim, dim), np.nan, dtype=precision.dtype)


def gaussian_marginal(view: GaussianView, state: State, edge: str):
    """Host readout for a controller requesting one proper Gaussian edge."""
    info = view.edges[edge]
    if not info.dim:
        raise ValueError(f"edge {edge!r} is not Gaussian")
    return _edge_moments(view, info, np.asarray(state.g_eta), np.asarray(state.g_lam))


def gaussian_marginals(
    view: GaussianView, state: State
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return moments by Gaussian edge name. Improper or unresolved beliefs have NaN moments."""
    eta, precision = np.asarray(state.g_eta), np.asarray(state.g_lam)
    return {
        edge.name: _edge_moments(view, edge, eta, precision)
        for edge in view.edges.values() if edge.dim
    }


def gaussian_node_stats(
    view: GaussianStatisticsView, state: State, *, pair_beliefs: PairBeliefs,
    centered_mixtures: bool = False,
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
                         ``weighted_second``, or the conditional mean and
                         covariance with ``centered_mixtures=True``.
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
            read_stats = mixg.mix_centered_stats if centered_mixtures else mixg.mix_stats
            stats = read_stats(
                inc_e[:, 0, :do],
                inc_l[:, 0, :do, :do],
                mu_e,
                state.params[g.name + ":m"],
                w,
            )
            for i, name in enumerate(g.members):
                out[name] = {key: np.asarray(value[i]) for key, value in stats._asdict().items()}

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
