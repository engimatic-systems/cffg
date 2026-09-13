"""hybrid readouts over explicit structure and current numerical inputs."""

from __future__ import annotations

import numpy as np

from cffg import hybrid_gfe
from cffg.execution.transport import _gauss_rows
from cffg.readouts.views import (
    HybridGfeView,
)
from cffg.runtime_types import RunStatus, State


def hybrid_gfe_diagnostics(
    view: HybridGfeView, state: State
) -> dict[str, dict[str, np.ndarray | float | str]]:
    """D5 objective channels and exact-mixture entropy bounds by factor."""
    output: dict[str, dict[str, np.ndarray | float | str]] = {}
    for group in view.hybrid_gfe_groups:
        incoming = state.messages[view.message_indices(group.s_gather, group.k_sel)]
        inc_eta, inc_lam = _gauss_rows(
            view,
            state.g_eta,
            state.g_lam,
            group.g_gather,
            group.dims,
        )
        common = (
            state.params[group.name + ":W"],
            state.params[group.name + ":goal_m"],
            state.params[group.name + ":goal_W"],
        )
        if group.type == "gaussian_mixture_linear":
            _, di = group.dims
            diagnostics = hybrid_gfe.linear_diagnostics(
                incoming,
                inc_eta[:, 1, :di],
                inc_lam[:, 1, :di, :di],
                state.params[group.name + ":A"],
                state.params[group.name + ":b"],
                *common,
            )
        else:
            diagnostics = hybrid_gfe.prior_diagnostics(
                incoming,
                state.params[group.name + ":m"],
                *common,
            )
        for row, member in enumerate(group.members):
            code = int(np.asarray(diagnostics.status[row]))
            output[member] = {
                "goal_factor": group.goal_members[row],
                "rho": np.asarray(diagnostics.rho[row]),
                "gfe_lower_bound": float(diagnostics.total[row]),
                "mi_regime_surrogate": float(diagnostics.mi_regime_surrogate[row]),
                "mi_regime_exact_lower": float(diagnostics.mi_regime_lower[row]),
                "mi_regime_exact_upper": float(diagnostics.mi_regime_upper[row]),
                "mi_state": float(diagnostics.mi_state[row]),
                "h_regime": float(diagnostics.h_regime[row]),
                "h_state": float(diagnostics.h_state[row]),
                "goal_risk": float(diagnostics.goal_risk[row]),
                "local_residual": float(diagnostics.residual[row]),
                "status": (
                    RunStatus.CONVERGED.value
                    if code == 0
                    else RunStatus.BUDGET_EXHAUSTED.value
                    if code == 1
                    else RunStatus.DIVERGED.value
                ),
                "bound_direction": "reported_gfe_lower_bounds_exact",
            }
    return output
