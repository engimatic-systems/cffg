"""Static schedule construction over typed local execution owners.

This module owns stage order, register transport and certification composition.
The caller supplies synchronous sweeps and the free-energy ledger; neither
factor mathematics nor public termination policy is defined here.
"""

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cffg.compiled import NEG, Arena
from cffg.execution.failures import FailureIndex
from cffg.execution.owners import SCHEDULED_EXECUTORS, ScheduledExecutors
from cffg.execution.startup import StartupProgram, make_startup_program
from cffg.execution.scans import _flat_scan_fn as _flat_scan_fn
from cffg.execution.scans import _scan_stage_fn
from cffg.execution.support import log_support as _jax_log_support
from cffg.execution.transport import (
    _disc_rows,
    _gauss_rows,
    _obs_rows,
    _set_disc_rows,
    _set_gauss_block,
)
from cffg.schedule import BatchStage, Schedule, _fingerprint, build_schedule
from cffg.failure_facts import FailurePhase, empty_failure, first_failure, in_phase
from cffg.schedule_contracts import ExactUnit, FreeEnergy, Propagation, Sweep


@dataclass(frozen=True)
class ScheduleProgram:
    """Static executable stages; numeric instance data is supplied on each call."""

    schedule: Schedule
    step: Sweep
    iter_step: Sweep
    propagate: Propagation | None = None
    prime: Propagation | None = None
    reconstruct: Propagation | None = None
    exact_unit: ExactUnit | None = None
    scan_registerized: tuple[bool, ...] = ()
    startup: StartupProgram | None = None


def build_schedule_program(
    arena: Arena,
    selection: str | Schedule,
    executors: ScheduledExecutors,
    *,
    make_sweep: Callable[[Arena, bool], Sweep],
    free_energy: FreeEnergy,
) -> ScheduleProgram:
    """Compose existing sweeps and local executors without owning their math."""
    raw_step = make_sweep(arena, False)
    step = eqx.filter_jit(raw_step)
    if isinstance(selection, Schedule):
        if selection.kind != "flood":
            expected = build_schedule(arena, "auto")
            if _fingerprint(arena, selection) != expected.fingerprint:
                raise ValueError(
                    "prebuilt schedule does not match this arena "
                    "(stale, doctored, or built for another model); "
                    "rebuild it with cffg.schedule.build_schedule"
                )
        schedule = selection
    else:
        schedule = build_schedule(arena, selection)
    if schedule.kind == "two_pass":
        raw = make_propagate(arena, schedule.stages, executors)
        return ScheduleProgram(
            schedule,
            step,
            step,
            propagate=eqx.filter_jit(raw),
            exact_unit=eqx.filter_jit(
                make_exact_unit(arena, raw, raw_step, free_energy, executors)
            ),
            scan_registerized=raw.scan_registerized,
        )
    if schedule.kind not in ("colored", "mixed"):
        return ScheduleProgram(schedule, step, step,
                               startup=make_startup_program(arena, make_sweep))

    island = make_propagate(arena, schedule.stages, executors)
    prime = reconstruct = None
    if schedule.prime_stages:
        prime_raw = (
            island
            if schedule.prime_stages == schedule.stages
            else make_propagate(arena, schedule.prime_stages, executors)
        )
        prime = eqx.filter_jit(prime_raw)
    if schedule.kind == "mixed":
        phase_step = make_sweep(arena, True)
        if schedule.forest_stages:
            reconstruct = eqx.filter_jit(
                make_propagate(arena, schedule.forest_stages, executors)
            )

        def iter_step(
            messages,
            g_eta,
            g_lam,
            g_obs,
            g_mask,
            ef_eta,
            ef_obs,
            ef_mask,
            params,
            damp=0.0,
            harden=True,
        ):
            messages, g_eta, g_lam, ef_eta, ir, ic, failure = phase_step(
                messages,
                g_eta,
                g_lam,
                g_obs,
                g_mask,
                ef_eta,
                ef_obs,
                ef_mask,
                params,
                damp,
                harden,
            )
            messages, g_eta, g_lam, ef_eta, island_failure = island(
                messages,
                g_eta,
                g_lam,
                ef_eta,
                g_obs,
                g_mask,
                ef_obs,
                ef_mask,
                params,
            )
            return messages, g_eta, g_lam, ef_eta, ir, ic, first_failure(failure, island_failure)
    else:

        def iter_step(
            messages,
            g_eta,
            g_lam,
            g_obs,
            g_mask,
            ef_eta,
            ef_obs,
            ef_mask,
            params,
            damp=0.0,
            harden=True,
        ):
            """One colored sweep, with damping after all stage writes."""
            m2, e2, l2, f2, failure = island(
                messages,
                g_eta,
                g_lam,
                ef_eta,
                g_obs,
                g_mask,
                ef_obs,
                ef_mask,
                params,
            )
            m2 = damp * messages + (1.0 - damp) * m2
            e2 = damp * g_eta + (1.0 - damp) * e2
            l2 = damp * g_lam + (1.0 - damp) * l2
            f2 = damp * ef_eta + (1.0 - damp) * f2
            zero = jnp.asarray(0.0, dtype=m2.dtype)
            return m2, e2, l2, f2, zero, jnp.asarray(0), failure

    return ScheduleProgram(
        schedule,
        step,
        eqx.filter_jit(iter_step),
        prime=prime,
        reconstruct=reconstruct,
        scan_registerized=island.scan_registerized,
        startup=make_startup_program(arena, make_sweep,
                                     skip_scheduled=schedule.kind == "mixed",
                                     propagate=island),
    )


def _batch_stage_fn(arena: Arena, stage, op, executors=SCHEDULED_EXECUTORS):
    """Executor for one BatchStage: fire a member subset of one group,
    reading the *current* arrays (sequential semantics — the whole point of
    a directed schedule) and scattering all of its outgoing roles."""
    failures = FailureIndex(arena)
    if stage.kind == "disc":
        g = arena.groups[stage.group]
        rows = stage.rows
        gather, scatter = np.asarray(g.gather[rows]), np.asarray(g.scatter[rows])

        def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
            inc = _disc_rows(arena, messages, gather, g.cards)
            delta = executors.discrete(
                g, inc, executors.parameters(g, params, rows), op
            )
            failure = failures.delta(g, delta, "disc", rows=rows, discrete_dtype=messages.dtype)
            out = jnp.maximum(delta.outgoing, NEG)
            messages = _set_disc_rows(arena, messages, scatter, g.cards, out)
            return messages, g_eta, g_lam, ef_eta, failure

        return fn

    if stage.kind == "ef":
        g = arena.ef_groups[stage.group]
        rows = stage.rows
        gather, scatter = np.asarray(g.gather[rows]), np.asarray(g.scatter[rows])
        obs_rows = np.asarray(g.obs_rows[rows]) if g.obs_rows is not None else None

        def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
            obs = ef_obs[obs_rows] if obs_rows is not None else None
            mask = ef_mask[obs_rows] if obs_rows is not None else None
            delta = executors.exponential_family(
                g, ef_eta[gather], executors.ef_parameters(g, params, rows), obs, mask
            )
            failure = failures.delta(g, delta, "ef", rows=rows, ef_dtype=ef_eta.dtype)
            out = delta.outgoing.astype(ef_eta.dtype)
            ef_eta = ef_eta.at[scatter.reshape(-1)].set(out.reshape(-1, out.shape[-1]))
            return messages, g_eta, g_lam, ef_eta, failure

        return fn

    g = arena.gauss_groups[stage.group]
    rows = stage.rows
    gather, scatter = np.asarray(g.gather[rows]), np.asarray(g.scatter[rows])
    obs_rows = np.asarray(g.obs_rows[rows]) if g.obs_rows is not None else None

    def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
        inc_e, inc_l = _gauss_rows(arena, g_eta, g_lam, gather, g.dims)
        y = mk = None
        if g.obs_rows is not None:
            y, mk = _obs_rows(arena, g_obs, obs_rows, g.observed_dim), g_mask[obs_rows]
        delta = executors.gaussian(
            g, inc_e, inc_l, executors.parameters(g, params, rows), y, mk
        )
        failure = failures.delta(g, delta, "gauss", rows=rows, gaussian_dtypes=(g_eta.dtype, g_lam.dtype))
        for r, (oe, ol) in enumerate(delta.outgoing):
            g_eta, g_lam = _set_gauss_block(
                arena, g_eta, g_lam, scatter[:, r], g.dims[r], oe, ol
            )
        return messages, g_eta, g_lam, ef_eta, failure

    return fn


def make_support_check(
    arena: Arena, executors: ScheduledExecutors = SCHEDULED_EXECUTORS
):
    """Jitted mirror of ``Engine._has_zero_support`` for constraint-free
    models (every two-pass model): True iff an edge or factor belief has
    empty exact support. Semantics are identical to the NumPy check: NEG is
    the one reserved zero sentinel and every other finite log value has
    support, irrespective of magnitude."""
    if arena.constraint_groups:  # combinatorial feasibility is host-side
        raise ValueError("jitted support check requires a constraint-free model")
    by_card: dict[int, list[list[int]]] = {}
    for e in arena.edges.values():
        if not e.dim and not e.ef and not e.param_shape:
            by_card.setdefault(e.card, []).append([e.slot0, e.slot1])
    edge_buckets = [
        (card, np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
        for card, pairs in sorted(by_card.items())
    ]

    def check(messages, params):
        bad = jnp.asarray(False)
        for card, pairs in edge_buckets:
            support = _jax_log_support(
                messages[arena.message_indices(pairs[:, 0], card)]
            ) & _jax_log_support(messages[arena.message_indices(pairs[:, 1], card)])
            bad |= ~jnp.all(jnp.any(support, axis=-1))
        for group in arena.groups:
            supports = [
                _jax_log_support(
                    messages[arena.message_indices(group.gather[:, r], card)]
                )
                for r, card in enumerate(group.cards)
            ]
            bad |= executors.discrete_zero_support(
                group, tuple(supports), executors.parameters(group, params, None)
            )
        return bad

    return check


def make_exact_unit(
    arena: Arena,
    propagate: Propagation,
    raw_step: Sweep,
    raw_bfe: FreeEnergy,
    executors: ScheduledExecutors = SCHEDULED_EXECUTORS,
) -> ExactUnit:
    """Fuse propagation, certification, support and the supplied ledger.

    The unjitted stage pipeline and callbacks form one XLA computation.
    """
    support_bad = make_support_check(arena, executors)
    is_max = arena.semiring == "max"

    def exact_unit(
        messages, g_eta, g_lam, g_obs, g_mask, ef_eta, ef_obs, ef_mask, params
    ):
        pre_bad = support_bad(messages, params)
        m, e, l, f, failure = propagate(
            messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params
        )
        vm, ve, vl, vf, _, _, certificate_failure = raw_step(
            m, e, l, g_obs, g_mask, f, ef_obs, ef_mask, params, 0.0, False
        )
        failure = first_failure(
            in_phase(failure, FailurePhase.PROPAGATION),
            in_phase(certificate_failure, FailurePhase.CERTIFICATION),
        )
        delta = jnp.asarray(0.0, dtype=m.dtype)
        rel = jnp.asarray(0.0, dtype=m.dtype)
        for a, b in ((vm, m), (ve, e), (vl, l), (vf, f)):
            if a.size:
                diff = jnp.abs(a - b)
                delta = jnp.maximum(delta, jnp.max(diff))
                # Per-entry scaling prevents a huge Gaussian precision from
                # widening the certificate for an unrelated discrete chain.
                rel = jnp.maximum(rel, jnp.max(diff / (1.0 + jnp.abs(b))))
        fe = (
            jnp.asarray(jnp.nan, dtype=m.dtype)
            if is_max
            else raw_bfe(m, e, l, g_obs, g_mask, f, ef_obs, ef_mask, params)
        )
        return m, e, l, f, delta, rel, fe, pre_bad, support_bad(m, params), failure

    return exact_unit


def make_propagate(
    arena: Arena, stages: tuple, executors: ScheduledExecutors = SCHEDULED_EXECUTORS
):
    """Compile a stage list into one jittable propagation function.

    Stages execute in order against the running arrays (asynchronous /
    directed semantics), unlike the synchronous sweep, which gathers from
    the pre-sweep arrays (synchronous flooding). Iterative structures (vmp,
    GFE, constraints, hybrid, prec) never appear as stages — they run in the
    restricted synchronous step of the mixed loop."""
    op = jnp.max if arena.semiring == "max" else logsumexp
    stage_fns = [
        _batch_stage_fn(arena, s, op, executors)
        if isinstance(s, BatchStage)
        else _scan_stage_fn(arena, s, op, executors)
        for s in stages
    ]

    def propagate(
        messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params
    ):
        failure = empty_failure()
        for fn in stage_fns:
            messages, g_eta, g_lam, ef_eta, stage_failure = fn(
                messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params
            )
            failure = first_failure(failure, stage_failure)
        return messages, g_eta, g_lam, ef_eta, failure

    propagate.scan_registerized = tuple(
        fn.registerized for fn in stage_fns if hasattr(fn, "registerized")
    )
    return propagate
