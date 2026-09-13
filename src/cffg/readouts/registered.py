"""registered readouts over explicit structure and current numerical inputs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from cffg import kernels
from cffg.execution.transport import _disc_both, _disc_rows, _gauss_rows, _obs_rows
from cffg.kernel_runtime import discrete_readout as _disc_readout
from cffg.kernel_runtime import gaussian_readout as _gauss_readout
from cffg.kernel_runtime import parameters as _kernel_params
from cffg.readouts.views import (
    KernelView,
)
from cffg.runtime_types import State


def kernel_readouts(view: KernelView, state: State) -> dict[str, dict[str, np.ndarray]]:
    """Factor name -> the optional readout its FactorKernel declares
    (cff-ri8m): posterior summaries beyond the generic edge marginals,
    computed from the same (inc, params, obs, mask) contract as the
    message/energy callables. Kernels without a readout are skipped."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for g in view.groups:
        if g.kernel is None:
            continue
        mode = g.kernel.binding.vmp if g.mode == "vmp" else g.kernel.binding.bp
        if mode.readout is None:
            continue
        inc = (
            _disc_both(view, state.messages, g.gather_both, g.cards)
            if g.mode == "vmp"
            else _disc_rows(view, state.messages, g.gather, g.cards)
        )
        vals = _disc_readout(g, inc, _kernel_params(g, state.params))
        for i, member in enumerate(g.members):
            out[member] = {key: np.asarray(v)[i] for key, v in vals.items()}
    for g in view.gauss_groups:
        if g.kernel is None or g.kernel.binding.readout is None:
            continue
        inc_e, inc_l = _gauss_rows(view, state.g_eta, state.g_lam, g.gather, g.dims)
        y = mk = None
        if g.obs_rows is not None:
            y = _obs_rows(view, state.g_obs, g.obs_rows, g.observed_dim)
            mk = state.g_mask[jnp.asarray(g.obs_rows)]
        vals = _gauss_readout(g, inc_e, inc_l, _kernel_params(g, state.params), y, mk)
        for i, member in enumerate(g.members):
            out[member] = {key: np.asarray(v)[i] for key, v in vals.items()}
    for g in view.ef_groups:
        kern = kernels.kernel_for(g.type)
        if kern is None or kern.readout is None:
            continue
        gp = {key: state.params[f"{g.name}:{key}"] for key in kern.param_keys}
        obs = state.ef_obs[jnp.asarray(g.obs_rows)] if g.obs_rows is not None else None
        mask = (
            state.ef_mask[jnp.asarray(g.obs_rows)] if g.obs_rows is not None else None
        )
        vals = kern.readout(state.ef_eta[np.asarray(g.gather)], gp, obs, mask)
        for i, member in enumerate(g.members):
            out[member] = {key: np.asarray(v)[i] for key, v in vals.items()}
    return out
