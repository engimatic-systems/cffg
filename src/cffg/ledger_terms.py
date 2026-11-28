"""Existing local objective terms, with explicit shared-parameter ownership.

The add_* functions carry the accumulator to preserve floating-point addition
order across extracted groups rather than regrouping their sums.
"""

import jax
import jax.numpy as jnp
import numpy as np

from cffg import csp, efrules, gauss, hybrid_gfe, kernels, mixg, recurrence
from cffg.execution.transport import gaussian_tap_parameters as _tap_parameters
from cffg.compiled import Arena
from cffg.execution import parameters as _parameters
from cffg.execution.support import log_support as _jax_log_support
from cffg.execution.transport import _disc_both, _disc_rows, _gauss_rows, _obs_rows
from cffg.goal_parameters import shared_goal_energy
from cffg.kernel_runtime import gaussian_energy as _gauss_energy
from cffg.kernel_runtime import parameters as _kernel_params
from cffg.readouts.energy import sum_edges, sum_nodes
from cffg.rules import (
    dirichlet_expected_log,
    dirichlet_vmp_energy,
    gfe_energy_stats,
    gfe_goal_structural_stats,
    gfe_parameter_stats,
    gfe_tm_energy,
)
from cffg.schedule_contracts import Parameters


def gaussian_energy(arena, g_eta, g_lam, g_obs, g_mask, params, report=None):
    """Gaussian node energies + interior-edge differential entropies. Exact
    closed forms (cffg.gauss docstring); with conditioned observations the
    tree identity BFE = −log p(y) holds to fixed-point/ridge precision."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for g in arena.gauss_groups:
        inc_e, inc_l = _gauss_rows(arena, g_eta, g_lam, g.gather, g.dims)
        y = mk = None
        if g.obs_rows is not None:
            y = _obs_rows(arena, g_obs, g.obs_rows, g.observed_dim)
            mk = g_mask[jnp.asarray(g.obs_rows)]
        parameters = _tap_parameters(g, _kernel_params(g, params), g_obs, g_mask)
        values = _gauss_energy(g, inc_e, inc_l, parameters, y, mk)
        total = total + sum_nodes(g, values, report)
    for d, pairs in arena.gauss_interior:
        l0 = g_lam[arena.g_lam_indices(pairs[:, 0], d)]
        l1 = g_lam[arena.g_lam_indices(pairs[:, 1], d)]
        if arena.gaussian_taps:
            slots = {arena.edges[name].slot0: row for name, row in arena.gaussian_taps.items()}
            rows = np.array([slots.get(int(pair[0]), -1) for pair in pairs])
            fixed = (rows >= 0) & (g_mask[np.maximum(rows, 0)] != 0)
            # Supply a proper unused operand to the entropy primitive on eliminated coordinates.
            l0 = jnp.where(fixed[:, None, None], jnp.eye(d, dtype=l0.dtype), l0)
            l1 = jnp.where(fixed[:, None, None], 0., l1)
            values = jnp.where(fixed, 0., gauss.edge_entropy(l0, l1))
        else:
            values = gauss.edge_entropy(l0, l1)
        total = total + sum_edges("gaussian", pairs, values, report)
    return total


def mixture_energy(arena, messages, g_eta, g_lam, params, report=None):
    """Hybrid Gaussian-mixture node energies: exact for the held
    labelled-mixture beliefs (mixg energies). The Gaussian and selector edge
    entropies need no code here — the out/in edges land in gauss_interior and
    the selector edge in interior_slot_pairs, both factor-agnostic tables."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for g in arena.mix_groups:
        mu_e = messages[arena.message_indices(g.s_gather, g.k_sel)]
        inc_e, inc_l = _gauss_rows(arena, g_eta, g_lam, g.g_gather, g.dims)
        w = params[g.name + ":W"]
        if g.type == "gaussian_mixture_linear":
            do, di = g.dims
            a, b = params[g.name + ":A"], params[g.name + ":b"]
            e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
            e_i, l_i = inc_e[:, 1, :di], inc_l[:, 1, :di, :di]
            values = mixg.mixlin_energy(e_o, l_o, e_i, l_i, mu_e, a, b, w)
        else:
            (do,) = g.dims
            m = params[g.name + ":m"]
            e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
            values = mixg.mix_energy(e_o, l_o, mu_e, m, w)
        total = total + sum_nodes(g, values, report)
    return total


def recurrence_energy(arena, messages, g_eta, g_lam, params, report=None):
    """D9 pair-region energies, upper-bounding exact transition terms."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for group in arena.recurrence_groups:
        card = group.cards[0]
        out_cavity = messages[arena.message_indices(group.d_gather_both[:, 0, 1], card)]
        in_cavity = messages[arena.message_indices(group.d_gather_both[:, 1, 1], card)]
        out_cavity = jnp.where(_jax_log_support(out_cavity), out_cavity, -jnp.inf)
        in_cavity = jnp.where(_jax_log_support(in_cavity), in_cavity, -jnp.inf)
        drive_eta = g_eta[arena.g_eta_indices(group.drive_gather_both, group.dim)].sum(
            axis=1
        )
        drive_precision = g_lam[
            arena.g_lam_indices(group.drive_gather_both, group.dim)
        ].sum(axis=1)
        values = recurrence.structured_recurrence_energy(
            out_cavity, in_cavity, drive_eta, drive_precision,
            params[group.name + ":weights"], params[group.name + ":offsets"],
        )
        total = total + sum_nodes(group, values, report)
    return total


def hybrid_gfe_energy(arena, messages, g_eta, g_lam, params, report=None):
    """Direct D5 G-hat terms; absorbed goals and output entropies stay out."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for group in arena.hybrid_gfe_groups:
        incoming = messages[arena.message_indices(group.s_gather, group.k_sel)]
        inc_eta, inc_lam = _gauss_rows(arena, g_eta, g_lam, group.g_gather, group.dims)
        common = (
            params[group.name + ":W"],
            params[group.name + ":goal_m"],
            params[group.name + ":goal_W"],
        )
        if group.type == "gaussian_mixture_linear":
            _, di = group.dims
            values = hybrid_gfe.linear_energy(
                incoming,
                inc_eta[:, 1, :di],
                inc_lam[:, 1, :di, :di],
                params[group.name + ":A"],
                params[group.name + ":b"],
                *common,
            )
        else:
            values = hybrid_gfe.prior_energy(
                incoming,
                params[group.name + ":m"],
                *common,
            )
        total = total + sum_nodes(group, values, report, kind="hybrid_gfe",
                                  direction="reported_gfe_lower_bounds_exact")
    return total


def precision_energy(arena, g_eta, g_lam, ef_eta, params, report=None):
    """gaussian_prec node energies: the mean-field VFE contribution
    −H[q_x] − H[q_τ] − E_q[log f]. gauss.prec_energy supplies −E_q[log f] (with
    the digamma correction that makes the mean-field VFE exact); the two edge
    neg-entropies are added here, exactly as the discrete VMP energies add one
    _neg_entropy per edge (cffg.rules.vmp_energy_matrix_cat). They net against
    the neighbours' −H and the interior +H (gauss_interior / ef_interior) to
    the single mean-field −H[q_i] per variable. Improper marginals ⇒ +inf."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for g in arena.prec_groups:
        tau_q = ef_eta[jnp.asarray(g.tau_gather_both)].sum(axis=1)  # [M, 2]
        x_eta_q = g_eta[arena.g_eta_indices(g.x_gather_both, 1)].sum(axis=1)
        x_lam_q = g_lam[arena.g_lam_indices(g.x_gather_both, 1)].sum(axis=1)
        m = params[g.name + ":m"]
        e = gauss.prec_energy(
            tau_q, x_eta_q, x_lam_q, m
        )  # −E_q[log f], +inf if improper
        neg_h_x = gauss._neg_entropy(x_lam_q)  # −H[q_x]
        neg_h_tau = -efrules.GAMMA.entropy(tau_q)  # −H[q_τ]
        ok = (x_lam_q[..., 0, 0] > 0) & efrules.GAMMA.is_proper(tau_q)
        total = total + sum_nodes(g, jnp.where(ok, neg_h_x + neg_h_tau + e, jnp.inf), report)
    return total


def ef_energy(arena, ef_eta, ef_obs, ef_mask, params, report=None):
    """EF node energies + interior-edge entropies, all closed form
    (cffg.efrules). Improper beliefs anywhere make the total +inf — loud.
    With conditioned observations the tree identity BFE = −log p(data)
    holds exactly (no ridge: nothing here is iterative or solved)."""
    total = jnp.asarray(0.0, dtype=ef_eta.dtype)
    for g in arena.ef_groups:
        inc = ef_eta[jnp.asarray(g.gather)]  # [M, R, 2]
        if g.type == "equality":
            values = efrules.equality_energy(efrules.JFAMILIES[g.family], inc)
        else:  # The registered kernel owns node energy.
            kern = kernels.KERNELS[g.type]
            gp = {key: params[f"{g.name}:{key}"] for key in kern.param_keys}
            obs = ef_obs[jnp.asarray(g.obs_rows)] if g.obs_rows is not None else None
            mask = ef_mask[jnp.asarray(g.obs_rows)] if g.obs_rows is not None else None
            values = kern.energy(inc, gp, obs, mask)
        total = total + sum_nodes(g, values, report)
    for fam_name, pairs in arena.ef_interior:
        fam = efrules.JFAMILIES[fam_name]
        values = efrules.edge_entropy(
            fam, ef_eta[jnp.asarray(pairs[:, 0])], ef_eta[jnp.asarray(pairs[:, 1])]
        )
        total = total + sum_edges("ef", pairs, values, report)
    return total


def add_dirichlet_terms(
    total: jax.Array, arena: Arena, messages: jax.Array, params: Parameters, report=None
) -> jax.Array:
    for g in arena.dirichlet_groups:
        inc = _disc_both(arena, messages, g.gather_both, g.cards)
        packed_alpha = (
            messages[arena.message_indices(g.prior_scatter, g.param_size)]
            + messages[arena.message_indices(g.count_scatter, g.param_size)]
        )
        alpha = packed_alpha.reshape((len(g.members), *g.param_shape))
        values = dirichlet_vmp_energy(g.type, inc, alpha, params[f"{g.name}:alpha0"], g.cards)
        total = total + sum_nodes(g, values, report, kind="dirichlet_region")
    return total


def add_constraint_terms(
    total: jax.Array, arena: Arena, messages: jax.Array, params: Parameters, report=None
) -> jax.Array:
    for c in arena.constraint_groups:
        inc = _disc_rows(arena, messages, c.gather, c.cards)
        fn = csp.CONSTRAINT_ENERGIES[c.type]
        parameters = _parameters.constraint_parameters(c, params)
        if c.type in ("exactly_k", "at_most_k"):
            value = fn(inc, c.cards, k=parameters.threshold)
        elif c.type == "table":
            value = fn(inc, c.cards, table=parameters.table)
        else:
            value = fn(inc, c.cards)
        if report is not None:
            report.add("constraint", c.members, value)
        total = total + value
    return total


def add_gfe_terms(
    total: jax.Array, arena: Arena, messages: jax.Array, params: Parameters, report=None
) -> jax.Array:
    goal_predictions = {}
    for g in arena.gfe_groups:
        _, ki = g.cards
        ko, _ = g.cards
        log_d = messages[arena.message_indices(g.gather_z, ki)]
        a_key = g.name + (":counts" if g.learned else ":A")
        epistemic_label = "parameter_aware_epistemic" if g.learned or g.mu3 else "epistemic"
        if g.goal_learned:
            alpha0 = params[g.name + ":c_alpha0"]
            current_counts = messages[arena.message_indices(g.goal_count_scatter, ko)]
            alpha = efrules.DIRICHLET.posterior(alpha0, current_counts)
            expected_log_c = dirichlet_expected_log(alpha)
            abar, hbar = gfe_parameter_stats(params[a_key], learned=g.learned)
            result = gfe_goal_structural_stats(
                log_d, abar, hbar, expected_log_c, components=report is not None
            )
            if report is None:
                structural, predicted = result
            else:
                structural, predicted, info, entropy = result
                report.add("observation_gfe", g.members, structural, components={
                    epistemic_label: info, "state_negative_entropy": entropy,
                })
            total = total + structural.sum()
            goal_predictions[g.name] = predicted
        else:
            if g.mu3:
                abar, hbar = params[g.name + ":Abar"], params[g.name + ":hbar"]
            else:
                abar, hbar = gfe_parameter_stats(params[a_key], learned=g.learned)
            values = gfe_energy_stats(log_d, abar, hbar, params[g.name + ":c"],
                                      components=report is not None)
            if report is not None:
                values, extrinsic, info, entropy = values
                report.add("observation_gfe", g.members, values, components={
                    "extrinsic": extrinsic, epistemic_label: info, "state_negative_entropy": entropy,
                })
            total = total + values.sum()
    total = total + shared_goal_energy(arena, messages, params, goal_predictions, report=report)
    # The structural terms above contain -E_q[Σ phi_t]. At the coordinate
    # optimum q*(A) ∝ q0(A) exp(Σ phi_t), the shared parameter ledger is
    # KL(q*||q0) = E_q[Σ phi_t] - log Z. Add it exactly once per tied
    # particle group, leaving the complete parameter contribution -log Z.
    for plan in arena.mu3_groups:
        value = params[plan.name + ":kl"]
        if report is not None:
            report.add("importance_sampled_parameter_kl", (plan.param_blob,), value)
        total = total + value
    return total


def add_transition_gfe_terms(
    total: jax.Array, arena: Arena, messages: jax.Array, params: Parameters, report=None
) -> jax.Array:
    for g in arena.gfe_tm_groups:
        ko, ki, ku = g.cards
        md = messages[arena.message_indices(g.gather[:, 0], ko)]
        mi = messages[arena.message_indices(g.gather[:, 1], ki)]
        me = messages[arena.message_indices(g.gather[:, 2], ku)]
        b_key = g.name + (":counts" if g.learned else ":B")
        values = gfe_tm_energy(mi, me, md, params[b_key], learned=g.learned,
                               components=report is not None)
        if report is not None:
            values, info, entropy = values
            report.add("transition_gfe", g.members, values, components={
                "optimistic_transition_information_bound": info,
                "state_negative_entropy": entropy,
            })
        total = total + values.sum()
    return total
