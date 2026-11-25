"""gaussian readouts over explicit structure and current numerical inputs."""

from __future__ import annotations

from operator import attrgetter

import jax.numpy as jnp
import numpy as np

from cffg import mixg
from cffg.execution.transport import _gauss_rows, _obs_rows
from cffg.kernel_runtime import gaussian_readout as _gauss_readout
from cffg.kernel_runtime import parameters as _kernel_params
from cffg.readouts.selection import EdgeSelection, grouped_edges, selected_edges
from cffg.readouts.types import (
    PairBeliefs,
)
from cffg.readouts.views import (
    GaussianStatisticsView,
    GaussianView,
    MixtureView,
)
from cffg.runtime_types import State


def _cholesky(matrices):
    """Isolate failed LAPACK factorizations without losing the other rows."""
    try:
        return np.linalg.cholesky(matrices)
    except np.linalg.LinAlgError:
        if matrices.ndim == 2 or len(matrices) == 1:
            return np.full_like(matrices, np.nan)
        middle = len(matrices) // 2
        return np.concatenate((_cholesky(matrices[:middle]), _cholesky(matrices[middle:])))


def _resolved_correlation(matrix):
    """Test each matrix at its stored precision independently of coordinate units."""
    dim = matrix.shape[-1]
    scale = np.sqrt(np.diagonal(matrix, axis1=-2, axis2=-1).astype(np.float64))
    correlation = (matrix / scale[..., :, None]) / scale[..., None, :]
    transpose = correlation.swapaxes(-2, -1)
    eps = np.finfo(matrix.dtype).eps
    valid = np.isfinite(correlation).all(axis=(-2, -1))
    valid &= (np.max(np.abs(correlation - transpose), axis=(-2, -1))
              <= 16 * dim * eps * np.max(np.abs(correlation), axis=(-2, -1)))
    correlation = (correlation + transpose) / 2
    guarded = correlation.copy()
    if not valid.all():
        guarded[~valid] = 0
    # Reserve source-dtype roundoff in each Cholesky pivot for admission only.
    diagonal = np.arange(dim)
    guarded[..., diagonal, diagonal] = 1 - dim * eps
    valid &= np.isfinite(_cholesky(guarded)[..., -1, -1])
    return correlation, scale, valid


def _moments(h, precision):
    correlation, scale, admitted = _resolved_correlation(precision)
    admitted &= np.isfinite(h).all(axis=-1)
    dim = h.shape[-1]
    if not admitted.any():
        return np.full_like(h, np.nan), np.full_like(precision, np.nan)
    if not admitted.all():
        # Keep one batched solve; rejected rows have no influence on their neighbors.
        correlation[~admitted] = np.eye(dim)
        scale[~admitted] = 1
    chol = _cholesky(correlation)
    rhs = np.empty((*h.shape, dim + 1), dtype=chol.dtype)
    rhs[..., 0] = h / scale
    rhs[..., 1:] = np.eye(dim)
    solved = np.linalg.solve(chol.swapaxes(-2, -1), np.linalg.solve(chol, rhs)) / scale[..., None]
    mean = solved[..., 0].astype(h.dtype)
    covariance = (solved[..., 1:] / scale[..., None, :]).astype(precision.dtype)
    _, _, valid = _resolved_correlation(covariance)
    valid &= admitted & np.isfinite(mean).all(axis=-1)
    if not valid.all():
        mean[~valid] = np.nan
        covariance[~valid] = np.nan
    return mean, covariance


def _group_moments(view, slots, dim, eta, precision):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = eta[view.g_eta_indices(slots, dim)].sum(axis=-2)
        p = precision[view.g_lam_indices(slots, dim)].sum(axis=-3)
        return _moments(h, p)


def gaussian_marginal(view: GaussianView, state: State, edge: str):
    """Host readout for a controller requesting one proper Gaussian edge."""
    info = view.edges[edge]
    if not info.dim:
        raise ValueError(f"edge {edge!r} is not Gaussian")
    return _group_moments(view, (info.slot0, info.slot1), info.dim,
                          np.asarray(state.g_eta), np.asarray(state.g_lam))


def gaussian_marginals(
    view: GaussianView, state: State, edges: EdgeSelection = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return moments by Gaussian edge name. Improper or unresolved beliefs have NaN moments."""
    dimension = attrgetter("dim")
    selected = selected_edges(view, edges, "Gaussian", dimension)
    if not selected:
        return {}
    if len(selected) == 1:
        name = selected[0].name
        return {name: gaussian_marginal(view, state, name)}
    eta, precision = np.asarray(state.g_eta), np.asarray(state.g_lam)
    out = {}
    for dim, group in grouped_edges(selected, dimension):
        slots = [(edge.slot0, edge.slot1) for edge in group]
        means, covariances = _group_moments(view, slots, dim, eta, precision)
        out.update((edge.name, (mean, cov))
                   for edge, mean, cov in zip(group, means, covariances, strict=True))
    return {edge.name: out[edge.name] for edge in selected}


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
