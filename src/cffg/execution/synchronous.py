"""Synchronous transport through shared local factor-execution owners.

Every local update reads the pre-sweep arrays. Collected writes are combined
and damped afterward, including shared goals and cross-domain factor outputs.
This owner has no public convergence, lifecycle or free-energy responsibility.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from cffg.compiled import NEG, Arena
from cffg.execution import (
    composites as _composites,
)
from cffg.execution import (
    cross_domain as _cross_domain,
)
from cffg.execution import (
    parameters as _parameters,
)
from cffg.execution.constraints import execute_constraint as _execute_constraint
from cffg.execution.failures import FailureIndex, pooled_goal_failure, simultaneous_failure
from cffg.execution.registered import (
    ef_parameters as _ef_parameters,
)
from cffg.execution.registered import (
    execute_discrete as _execute_discrete,
)
from cffg.execution.registered import (
    execute_ef as _execute_ef,
)
from cffg.execution.registered import (
    execute_gaussian as _execute_gaussian,
)
from cffg.execution.transport import (
    _disc_both,
    _disc_rows,
    _gauss_rows,
    _obs_rows,
    _set_disc_block,
    _set_disc_rows,
    _set_gauss_block,
)
from cffg.goal_parameters import write_goal_counts
from cffg.failure_facts import empty_failure
from cffg.kernel_runtime import parameters as _kernel_params
from cffg.schedule_contracts import Sweep


def _mix_sweep(arena, messages, g_eta, g_lam, params, failures):
    """Gather mixture cavities and collect writes for both message arenas."""
    sel_writes = []
    gauss_writes = []
    failure = empty_failure()
    for g in arena.mix_groups:
        selector = messages[arena.message_indices(g.s_gather, g.k_sel)]
        eta, precision = _gauss_rows(arena, g_eta, g_lam, g.g_gather, g.dims)
        delta = _cross_domain.execute_mixture(
            g, selector, eta, precision, _parameters.mixture_parameters(g, params)
        )
        failure = simultaneous_failure(failure, failures.delta(
            g, delta, "hybrid", discrete_dtype=messages.dtype,
            gaussian_dtypes=(g_eta.dtype, g_lam.dtype)))
        for role, outgoing in enumerate(delta.outgoing.gaussian):
            gauss_writes.append(
                (g.g_scatter[:, role], g.dims[role], outgoing.eta, outgoing.precision)
            )
        sel_writes.append((g.s_scatter, g.k_sel, delta.outgoing.selector))
    return sel_writes, gauss_writes, failure


def _recurrence_sweep(arena, messages, g_eta, g_lam, params, failures):
    """Gather declared recurrence inputs and collect categorical/drive writes."""
    discrete_writes = []
    gaussian_writes = []
    status = jnp.asarray(0)
    failure = empty_failure()
    for group in arena.recurrence_groups:
        card = group.cards[0]
        destination = messages[
            arena.message_indices(group.d_gather_both[:, 0, 1], card)
        ]
        source = messages[arena.message_indices(group.d_gather_both[:, 1, 1], card)]
        eta_both = g_eta[arena.g_eta_indices(group.drive_gather_both, group.dim)]
        precision_both = g_lam[arena.g_lam_indices(group.drive_gather_both, group.dim)]
        delta = _cross_domain.execute_recurrence(
            group,
            destination,
            source,
            eta_both,
            precision_both,
            _parameters.recurrence_parameters(group, params),
        )
        discrete_writes.extend(
            (
                (group.d_scatter[:, 0], card, delta.outgoing.destination),
                (group.d_scatter[:, 1], card, delta.outgoing.source),
            )
        )
        gaussian_writes.append(
            (
                group.drive_scatter,
                group.dim,
                delta.outgoing.drive.eta,
                delta.outgoing.drive.precision,
            )
        )
        status = jnp.maximum(status, jnp.max(delta.status))
        failure = simultaneous_failure(failure, failures.delta(
            group, delta, "hybrid", discrete_dtype=messages.dtype,
            gaussian_dtypes=(g_eta.dtype, g_lam.dtype)))
    return discrete_writes, gaussian_writes, status, failure


def _hybrid_gfe_sweep(arena, messages, g_eta, g_lam, params, failures):
    """Gather hybrid-composite inputs and retain their local solver facts."""
    selector_writes = []
    gaussian_writes = []
    residual = jnp.asarray(0.0, dtype=messages.dtype)
    status = jnp.asarray(0)
    failure = empty_failure()
    for group in arena.hybrid_gfe_groups:
        selector = messages[arena.message_indices(group.s_gather, group.k_sel)]
        eta, precision = _gauss_rows(arena, g_eta, g_lam, group.g_gather, group.dims)
        delta = _composites.execute_hybrid_gfe(
            group,
            selector,
            eta,
            precision,
            _parameters.hybrid_parameters(group, params),
        )
        for role, outgoing in enumerate(delta.outgoing.gaussian):
            gaussian_writes.append(
                (
                    group.g_scatter[:, role],
                    group.dims[role],
                    outgoing.eta,
                    outgoing.precision,
                )
            )
        selector_writes.append((group.s_scatter, group.k_sel, delta.outgoing.selector))
        residual = jnp.maximum(residual, jnp.max(delta.residual))
        status = jnp.maximum(status, jnp.max(delta.status))
        failure = simultaneous_failure(failure, failures.delta(
            group, delta, "hybrid", discrete_dtype=messages.dtype,
            gaussian_dtypes=(g_eta.dtype, g_lam.dtype)))
    return selector_writes, gaussian_writes, residual, status, failure


def _prec_sweep(arena, g_eta, g_lam, ef_eta, params, failures):
    """Gather both edge sides for the precision owner's VMP belief inputs."""
    gauss_writes = []
    ef_writes = []
    failure = empty_failure()
    for g in arena.prec_groups:
        tau_both = ef_eta[jnp.asarray(g.tau_gather_both)]
        eta_both = g_eta[arena.g_eta_indices(g.x_gather_both, 1)]
        precision_both = g_lam[arena.g_lam_indices(g.x_gather_both, 1)]
        delta = _cross_domain.execute_precision(
            g, tau_both, eta_both, precision_both, _parameters.precision_mean(g, params)
        )
        gaussian = delta.outgoing.gaussian
        gauss_writes.append((g.x_scatter, 1, gaussian.eta, gaussian.precision))
        ef_writes.append((g.tau_scatter, delta.outgoing.precision_parameter))
        failure = simultaneous_failure(failure, failures.delta(
            g, delta, "hybrid", ef_dtype=ef_eta.dtype,
            gaussian_dtypes=(g_eta.dtype, g_lam.dtype)))
    return gauss_writes, ef_writes, failure


def _gauss_sweep(arena, g_eta, g_lam, g_obs, g_mask, params, failures, *, only_iterative=False):
    """One synchronous flooding sweep over the Gaussian groups; returns the
    fully rewritten flat (eta, Lambda) arrays. Dims are static per group, so
    every rule call sees a shape-local block."""
    new_eta, new_lam = g_eta, g_lam
    residual, status = jnp.asarray(0.0, g_eta.dtype), jnp.asarray(0)
    failure = empty_failure()
    for g in arena.gauss_groups:
        if only_iterative and not (g.kernel is not None and g.kernel.binding.iterative):
            continue
        inc_e, inc_l = _gauss_rows(arena, g_eta, g_lam, g.gather, g.dims)
        y = mk = None
        if g.obs_rows is not None:
            y, mk = (
                _obs_rows(arena, g_obs, g.obs_rows, g.observed_dim),
                g_mask[jnp.asarray(g.obs_rows)],
            )
        delta = _execute_gaussian(
            g, inc_e, inc_l, _kernel_params(g, params), y, mk
        )
        if delta.residual is not None:
            residual = jnp.maximum(residual, jnp.max(delta.residual))
        if delta.status is not None:
            status = jnp.maximum(status, jnp.max(delta.status))
        failure = simultaneous_failure(failure, failures.delta(
            g, delta, "gauss", gaussian_dtypes=(g_eta.dtype, g_lam.dtype)))
        for r, (oe, ol) in enumerate(delta.outgoing):
            new_eta, new_lam = _set_gauss_block(
                arena, new_eta, new_lam, g.scatter[:, r], g.dims[r], oe, ol
            )
    return new_eta, new_lam, residual, status, failure


def _ef_group_out(g, inc, params, rows=None, obs=None, mask=None):
    """Outgoing EF messages for gathered local inputs and selected parameters."""
    return _execute_ef(g, inc, _ef_parameters(g, params, rows), obs, mask).outgoing


def _ef_sweep(arena, ef_eta, ef_obs, ef_mask, params, failures):
    """One synchronous flooding sweep over the EF groups — conjugate BP in
    its entirety (DERIVATIONS.md D2). Each non-equality group's update comes
    from its registered FactorKernel (cffg.kernels); equality is
    structural sum-minus-self. Groups are static, so kernel dispatch happens
    at trace time and the jitted step is branch-free."""
    new = ef_eta
    failure = empty_failure()
    for g in arena.ef_groups:
        obs = ef_obs[jnp.asarray(g.obs_rows)] if g.obs_rows is not None else None
        mask = ef_mask[jnp.asarray(g.obs_rows)] if g.obs_rows is not None else None
        delta = _execute_ef(
            g, ef_eta[jnp.asarray(g.gather)], _ef_parameters(g, params), obs, mask
        )
        out = delta.outgoing.astype(new.dtype)
        failure = simultaneous_failure(failure, failures.delta(g, delta, "ef", ef_dtype=ef_eta.dtype))
        new = new.at[jnp.asarray(g.scatter).reshape(-1)].set(
            out.reshape(-1, out.shape[-1])
        )
    return new, failure


def _constraint_writes(arena: Arena, messages: jax.Array, params, op, failures):
    """Combinatorial messages for the hard-constraint factors. Reads
    the pre-sweep messages (synchronous flooding); returns exact segmented
    writes for the discrete arena, folded into damping by the caller."""
    writes = []
    failure = empty_failure()
    for c in arena.constraint_groups:
        inc = _disc_rows(arena, messages, c.gather, c.cards)
        delta = _execute_constraint(
            c, inc, _parameters.constraint_parameters(c, params), op
        )
        writes.append((c.scatter, c.cards, delta.outgoing))
        failure = simultaneous_failure(failure, failures.delta(c, delta, "disc", discrete_dtype=messages.dtype))
    return writes, failure


def make_sweep(arena: Arena, skip_scheduled: bool = False) -> Sweep:
    """The synchronous sweep. With ``skip_scheduled=True`` (the mixed-mode
    phase step) the plain BP discrete groups, Gaussian groups, and
    EF groups are omitted — the directed island passes own them — while
    every iterative structure (vmp beads, GFE composites, constraints,
    hybrid mixtures, gaussian_prec) still fires and damps exactly as in
    flooding."""
    groups = (
        [g for g in arena.groups if g.mode != "bp"] if skip_scheduled else arena.groups
    )
    op = jnp.max if arena.semiring == "max" else logsumexp
    failures = FailureIndex(arena)

    def step(
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        ef_eta: jax.Array,
        ef_obs: jax.Array,
        ef_mask: jax.Array,
        params: dict[str, jax.Array],
        damp: float = 0.0,
        harden: bool = True,
    ):
        new = messages
        failure = empty_failure()
        inner_residual = jnp.asarray(0.0, dtype=messages.dtype)
        inner_status = jnp.asarray(0)
        for g in groups:
            if g.mode == "vmp":
                inc = _disc_both(arena, messages, g.gather_both, g.cards)
            else:
                inc = _disc_rows(arena, messages, g.gather, g.cards)
            delta = _execute_discrete(g, inc, _kernel_params(g, params), op)
            failure = simultaneous_failure(failure, failures.delta(g, delta, "disc", discrete_dtype=messages.dtype))
            out = delta.outgoing
            new = _set_disc_rows(arena, new, g.scatter, g.cards, out)
        for g in arena.dirichlet_groups:
            inc2 = _disc_both(arena, messages, g.gather_both, g.cards)
            delta = _composites.execute_dirichlet(
                g, inc2, _parameters.dirichlet_prior(g, params)
            )
            failure = simultaneous_failure(failure, failures.delta(g, delta, "disc", discrete_dtype=messages.dtype))
            new = _set_disc_rows(
                arena, new, g.scatter, g.cards, delta.outgoing.structural
            )
            new = _set_disc_block(
                arena,
                new,
                g.prior_scatter,
                g.param_size,
                delta.outgoing.prior.reshape((len(g.members), g.param_size)),
            )
            new = _set_disc_block(
                arena,
                new,
                g.count_scatter,
                g.param_size,
                delta.outgoing.counts.reshape((len(g.members), g.param_size)),
            )
        constraint_writes, constraint_failure = _constraint_writes(arena, messages, params, op, failures)
        failure = simultaneous_failure(failure, constraint_failure)
        for scatter, cards, out in constraint_writes:
            new = _set_disc_rows(arena, new, scatter, cards, out)
        goal_predictions = {}
        for g in arena.gfe_groups:
            ko, ki = g.cards
            incoming = messages[arena.message_indices(g.gather_z, ki)]
            goal_counts = (
                messages[arena.message_indices(g.goal_count_scatter, ko)]
                if g.goal_learned
                else None
            )
            delta = _composites.execute_gfe(
                g, incoming, _parameters.gfe_parameters(g, params), goal_counts
            )
            failure = simultaneous_failure(failure, failures.delta(g, delta, "disc", discrete_dtype=messages.dtype))
            new = _set_disc_block(arena, new, g.scatter_z, ki, delta.outgoing.latent)
            new = _set_disc_block(
                arena, new, g.scatter_x, ko, delta.outgoing.log_prediction
            )
            if g.goal_learned:
                goal_predictions[g.name] = delta.outgoing.goal_prediction
                new = _set_disc_block(
                    arena, new, g.goal_prior_scatter, ko, delta.outgoing.goal_prior
                )
            inner_residual = jnp.maximum(inner_residual, jnp.max(delta.residual))
            inner_status = jnp.maximum(inner_status, jnp.max(delta.status))
        new = write_goal_counts(arena, new, goal_predictions)
        failure = simultaneous_failure(failure, pooled_goal_failure(arena, new))
        for g in arena.gfe_tm_groups:
            ko, ki, ku = g.cards
            destination = messages[arena.message_indices(g.gather[:, 0], ko)]
            source = messages[arena.message_indices(g.gather[:, 1], ki)]
            selector = messages[arena.message_indices(g.gather[:, 2], ku)]
            delta = _composites.execute_transition_gfe(
                g,
                destination,
                source,
                selector,
                _parameters.transition_parameters(g, params),
            )
            failure = simultaneous_failure(failure, failures.delta(g, delta, "disc", discrete_dtype=messages.dtype))
            new = _set_disc_block(
                arena, new, g.scatter[:, 0], ko, delta.outgoing.destination
            )
            new = _set_disc_block(
                arena, new, g.scatter[:, 1], ki, delta.outgoing.source
            )
            new = _set_disc_block(
                arena, new, g.scatter[:, 2], ku, delta.outgoing.selector
            )
            inner_residual = jnp.maximum(inner_residual, jnp.max(delta.residual))
            inner_status = jnp.maximum(inner_status, jnp.max(delta.status))
        hybrid_sel, hybrid_gauss, hybrid_residual, hybrid_status, hybrid_failure = (
            _hybrid_gfe_sweep(arena, messages, g_eta, g_lam, params, failures)
            if arena.hybrid_gfe_groups
            else ([], [], jnp.asarray(0.0, messages.dtype), jnp.asarray(0), empty_failure())
        )
        inner_residual = jnp.maximum(inner_residual, hybrid_residual)
        inner_status = jnp.maximum(inner_status, hybrid_status)
        recurrence_disc, recurrence_gauss, recurrence_status, recurrence_failure = (
            _recurrence_sweep(arena, messages, g_eta, g_lam, params, failures)
            if arena.recurrence_groups
            else ([], [], jnp.asarray(0), empty_failure())
        )
        inner_status = jnp.maximum(inner_status, recurrence_status)
        # Hybrid Gaussian-mixture sweep: straddles both arenas.
        # Computed here from the pre-sweep arrays so its writes fold into each
        # arena's damping phase — Gaussian writes below, selector writes with
        # the discrete blend (synchronous flooding: gather from OLD arrays).
        mix_sel, mix_gauss, mix_failure = (
            _mix_sweep(arena, messages, g_eta, g_lam, params, failures)
            if arena.mix_groups
            else ([], [], empty_failure())
        )
        # gaussian_prec sweep: also straddles two arenas — its out
        # edge writes the Gaussian arena, its prec edge the EF arena. Computed
        # here from the pre-sweep arrays so both writes fold into the matching
        # arena's damping phase (synchronous flooding: gather from OLD arrays).
        prec_gauss, prec_ef, prec_failure = (
            _prec_sweep(arena, g_eta, g_lam, ef_eta, params, failures)
            if arena.prec_groups
            else ([], [], empty_failure())
        )
        for local in (hybrid_failure, recurrence_failure, mix_failure, prec_failure):
            failure = simultaneous_failure(failure, local)
        # Gaussian sweep: information-form damping below is the log-space
        # damping of the Gaussian family (log of a Gaussian is its (eta,
        # Lambda) quadratic), so one damp knob covers both arenas.
        new_eta, new_lam = g_eta, g_lam
        if arena.gauss_groups:
            new_eta, new_lam, gaussian_residual, gaussian_status, gaussian_failure = _gauss_sweep(
                arena, g_eta, g_lam, g_obs, g_mask, params, failures,
                only_iterative=skip_scheduled,
            )
            inner_residual = jnp.maximum(inner_residual, gaussian_residual)
            inner_status = jnp.maximum(inner_status, gaussian_status)
            failure = simultaneous_failure(failure, gaussian_failure)
        for scatter, dim, oe, ol in [
            *recurrence_gauss,
            *hybrid_gauss,
            *mix_gauss,
            *prec_gauss,
        ]:
            new_eta, new_lam = _set_gauss_block(
                arena, new_eta, new_lam, scatter, dim, oe, ol
            )
        if (
            arena.gauss_groups
            or arena.hybrid_gfe_groups
            or arena.mix_groups
            or arena.recurrence_groups
            or arena.prec_groups
        ):
            new_eta = damp * g_eta + (1.0 - damp) * new_eta
            new_lam = damp * g_lam + (1.0 - damp) * new_lam
        # EF sweep: natural parameters are the log-space of their family, so
        # the same damp knob is log-space damping here too.
        new_ef = ef_eta
        if arena.ef_groups or arena.prec_groups:
            if arena.ef_groups and not skip_scheduled:
                new_ef, ef_failure = _ef_sweep(arena, ef_eta, ef_obs, ef_mask, params, failures)
                failure = simultaneous_failure(failure, ef_failure)
            for scatter, out in prec_ef:
                new_ef = new_ef.at[jnp.asarray(scatter)].set(out)
            new_ef = damp * ef_eta + (1.0 - damp) * new_ef
        # Hybrid selector writes join the discrete arena before its damping.
        for scatter, card, rows in [
            *recurrence_disc,
            *hybrid_sel,
            *mix_sel,
        ]:
            new = _set_disc_block(arena, new, scatter, card, rows)
        # Log-space damping: stabilizes the outer coordination between GFE
        # composites (their local solves are exact, but synchronous flooding
        # between them can limit-cycle — the outer face of Part II §4.5).
        new = damp * messages + (1.0 - damp) * new
        # Floor at NEG: exact log-zeros compound additively through chains
        # (NEG + NEG + ...), and unchecked they overflow to -inf and NaN once
        # δ-hardened controls make states truly impossible. exp(NEG) == 0
        # either way, so clamping changes nothing semantically.
        new = jnp.maximum(new, NEG)
        if harden and len(arena.delta_pairs):
            # δ form constraint: project each constrained edge onto a pointmass
            # at the argmax of its current marginal (both slots one-hot).
            for card, pairs in arena.delta_slot_groups:
                p0, p1 = pairs[:, 0], pairs[:, 1]
                i0 = arena.message_indices(p0, card)
                i1 = arena.message_indices(p1, card)
                lq = new[i0] + new[i1]
                hot = jnp.where(
                    jnp.arange(card)[None, :] == jnp.argmax(lq, axis=-1)[:, None],
                    0.0,
                    NEG,
                ).astype(new.dtype)
                new = new.at[i0].set(hot).at[i1].set(hot)
        return new, new_eta, new_lam, new_ef, inner_residual, inner_status, failure

    return step
