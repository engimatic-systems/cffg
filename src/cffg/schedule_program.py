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
from cffg.execution_availability import require_execution_available
from cffg.execution.owners import SCHEDULED_EXECUTORS, ScheduledExecutors
from cffg.execution.support import log_support as _jax_log_support
from cffg.execution.transport import (
    _disc_rows,
    _gauss_rows,
    _obs_rows,
    _set_disc_rows,
    _set_gauss_block,
)
from cffg.schedule import BatchStage, Schedule, _fingerprint, build_schedule
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


def build_schedule_program(
    arena: Arena,
    selection: str | Schedule,
    executors: ScheduledExecutors,
    *,
    make_sweep: Callable[[Arena, bool], Sweep],
    free_energy: FreeEnergy,
) -> ScheduleProgram:
    """Compose existing sweeps and local executors without owning their math."""
    require_execution_available(arena, operation="build_schedule_program")
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
        return ScheduleProgram(schedule, step, step)

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
            messages, g_eta, g_lam, ef_eta, ir, ic = phase_step(
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
            messages, g_eta, g_lam, ef_eta = island(
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
            return messages, g_eta, g_lam, ef_eta, ir, ic
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
            m2, e2, l2, f2 = island(
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
            return m2, e2, l2, f2, zero, jnp.asarray(0)

    return ScheduleProgram(
        schedule,
        step,
        eqx.filter_jit(iter_step),
        prime=prime,
        reconstruct=reconstruct,
        scan_registerized=island.scan_registerized,
    )


def _batch_stage_fn(arena: Arena, stage, op, executors=SCHEDULED_EXECUTORS):
    """Executor for one BatchStage: fire a member subset of one group,
    reading the *current* arrays (sequential semantics — the whole point of
    a directed schedule) and scattering all of its outgoing roles."""
    if stage.kind == "disc":
        g = arena.groups[stage.group]
        rows = stage.rows
        gather, scatter = np.asarray(g.gather[rows]), np.asarray(g.scatter[rows])

        def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
            inc = _disc_rows(arena, messages, gather, g.cards)
            out = executors.discrete(
                g, inc, executors.parameters(g, params, rows), op
            ).outgoing
            out = jnp.maximum(out, NEG)
            messages = _set_disc_rows(arena, messages, scatter, g.cards, out)
            return messages, g_eta, g_lam, ef_eta

        return fn

    if stage.kind == "ef":
        g = arena.ef_groups[stage.group]
        rows = stage.rows
        gather, scatter = np.asarray(g.gather[rows]), np.asarray(g.scatter[rows])
        obs_rows = np.asarray(g.obs_rows[rows]) if g.obs_rows is not None else None

        def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
            obs = ef_obs[obs_rows] if obs_rows is not None else None
            mask = ef_mask[obs_rows] if obs_rows is not None else None
            out = executors.exponential_family(
                g, ef_eta[gather], executors.ef_parameters(g, params, rows), obs, mask
            ).outgoing
            out = out.astype(ef_eta.dtype)
            ef_eta = ef_eta.at[scatter.reshape(-1)].set(out.reshape(-1, out.shape[-1]))
            return messages, g_eta, g_lam, ef_eta

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
        outs = executors.gaussian(
            g, inc_e, inc_l, executors.parameters(g, params, rows), y, mk
        ).outgoing
        for r, (oe, ol) in enumerate(outs):
            g_eta, g_lam = _set_gauss_block(
                arena, g_eta, g_lam, scatter[:, r], g.dims[r], oe, ol
            )
        return messages, g_eta, g_lam, ef_eta

    return fn


def _scan_group_of(arena: Arena, elem):
    return {
        "disc": arena.groups,
        "gauss": arena.gauss_groups,
        "ef": arena.ef_groups,
    }[elem.kind][elem.group]


def _classify_scan(arena: Arena, stage):
    """Registerization analysis for a ScanStage.

    Per (elem, role, member-column) gather, find where its slots come from
    *within* the stage, uniformly across all steps: written by elem ``we``
    role ``wr`` column ``wc`` in the previous step (d=1, a carried
    register), the same step (d=0: a register when the writer runs earlier,
    otherwise a pre-stage read), or the next step (d=-1: always a pre-stage
    read — the write comes after every read of that slot, and each directed
    slot is written at most once per pass). Columns whose slots the stage
    never writes are pre-stage reads too ("static").

    Returns per-elem per-role plans compressed into three shapes — whole
    role static ("table"), whole role one aligned register ("direct"), or
    per-column mixed ("mix") — or None when any column's dependency is
    non-uniform, in which case the flat-carry executor stays correct."""
    elems = stage.elems
    # Slot numberings are independent per arena (disc/gauss/ef), so reader
    # and writer tables only compare within the same kind.
    writers = []
    written_by_kind: dict[str, set[int]] = {}
    for ei, elem in enumerate(elems):
        sc = _scan_group_of(arena, elem).scatter[elem.rows]  # [S, m, R]
        for r in range(sc.shape[2]):
            writers.append((ei, r, sc[:, :, r]))
            written_by_kind.setdefault(elem.kind, set()).update(
                int(s) for s in sc[:, :, r].reshape(-1)
            )

    plans = []
    for ei, elem in enumerate(elems):
        ga = _scan_group_of(arena, elem).gather[elem.rows]  # [S, m, R]
        all_written = written_by_kind.get(elem.kind, set())
        role_plans = []
        for r in range(ga.shape[2]):
            cols = []
            for c in range(ga.shape[1]):
                tab = ga[:, c, r]  # [S]
                match = None
                for we, wr, wt in writers:
                    if elems[we].kind != elem.kind:
                        continue
                    for wc in range(wt.shape[1]):
                        col = wt[:, wc]
                        if stage.length > 1 and np.array_equal(col[:-1], tab[1:]):
                            match = ("r", we, wr, wc, 1)
                        elif np.array_equal(col, tab):
                            match = ("r", we, wr, wc, 0) if we < ei else ("s",)
                        elif (
                            stage.length > 1
                            and np.array_equal(col[1:], tab[:-1])
                            and int(tab[-1]) not in all_written
                        ):
                            # Written only at the *next* step — every read
                            # precedes the write, so the pre-stage value is
                            # correct, provided the final step's slot (outside
                            # the matched pattern) is not written at all.
                            match = ("s",)
                        if match:
                            break
                    if match:
                        break
                if match is None:
                    if {int(s) for s in tab.reshape(-1)} & all_written:
                        return None  # non-uniform in-stage dependency
                    match = ("s",)
                cols.append(match)
            if all(p == ("s",) for p in cols):
                role_plans.append(("table",))
                continue
            direct = (
                all(p[0] == "r" for p in cols)
                and len({(p[1], p[2], p[4]) for p in cols}) == 1
                and [p[3] for p in cols] == list(range(len(cols)))
            )
            if direct:
                direct = elems[cols[0][1]].rows.shape[1] == len(cols)
            if direct:
                role_plans.append(("direct", cols[0][1], cols[0][2], cols[0][4]))
            else:
                role_plans.append(("mix", tuple(cols)))
        plans.append(role_plans)
    return plans


def _registerized_scan_fn(
    arena: Arena, stage, plans, op, executors=SCHEDULED_EXECUTORS
):
    """Registerized executor: the scan carries only the small message blocks
    that cross steps; pre-stage inputs are prefetched [S, ...] tensors,
    outputs are stacked by the scan and scattered into the flat arrays once
    at the end. This is the executor that makes a chain cost what a
    hand-written ``lax.scan`` smoother costs — no dynamic gather/scatter on
    the flat arena inside the loop."""
    elems = stage.elems
    metas = []
    for ei, elem in enumerate(elems):
        g = _scan_group_of(arena, elem)
        ga, sc = g.gather[elem.rows], g.scatter[elem.rows]  # [S, m, R]
        metas.append((elem, g, ga, sc))

    def _col_plans(plan, m):
        if plan[0] == "mix":
            return plan[1]
        if plan[0] == "table":
            return (("s",),) * m
        _, we, wr, d = plan
        return tuple(("r", we, wr, wc, d) for wc in range(m))

    # Carried registers: (we, wr) consumed with d=1 anywhere. Their step-0
    # value is the pre-stage message at each reader's first gather slot;
    # unread entries default to the writer's own step-0 slots (any finite
    # value — they are overwritten before every read).
    reg_init_slots: dict[tuple[int, int], np.ndarray] = {}
    for ei, (elem, g, ga, _) in enumerate(metas):
        for r, plan in enumerate(plans[ei]):
            for c, cp in enumerate(_col_plans(plan, ga.shape[1])):
                if cp[0] == "r" and cp[4] == 1:
                    key = (cp[1], cp[2])
                    if key not in reg_init_slots:
                        _, _, _, we_sc = metas[cp[1]]
                        reg_init_slots[key] = we_sc[0, :, cp[2]].copy()
                    reg_init_slots[key][cp[3]] = ga[0, c, r]

    def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
        xs = []
        for ei, (elem, g, ga, _) in enumerate(metas):
            x = {}
            for r, plan in enumerate(plans[ei]):
                if plan[0] == "direct":
                    continue
                cols = _col_plans(plan, ga.shape[1])
                scols = [c for c, cp in enumerate(cols) if cp[0] == "s"]
                if not scols:
                    continue
                slots = ga[:, scols, r]  # [S, ns]
                if elem.kind == "disc":
                    idx = arena.message_indices(slots, g.cards[r])
                    x[f"s{r}"] = messages[jnp.asarray(idx)]
                elif elem.kind == "ef":
                    x[f"s{r}"] = ef_eta[jnp.asarray(slots)]
                else:
                    d = g.dims[r]
                    x[f"se{r}"] = g_eta[jnp.asarray(arena.g_eta_indices(slots, d))]
                    x[f"sl{r}"] = g_lam[jnp.asarray(arena.g_lam_indices(slots, d))]
            if elem.kind == "disc":
                for key, value in executors.parameters(g, params, elem.rows).items():
                    x[f"p{key}"] = value
            elif elem.kind == "ef":
                for key, value in executors.ef_parameters(g, params, elem.rows).items():
                    x[f"p{key}"] = value
                if g.obs_rows is not None:
                    orows = g.obs_rows[elem.rows]
                    x["ov"] = ef_obs[jnp.asarray(orows)]
                    x["om"] = ef_mask[jnp.asarray(orows)]
            else:
                for k, v in executors.parameters(g, params, None).items():
                    x[f"p{k}"] = v[elem.rows]
                if g.obs_rows is not None:
                    orows = g.obs_rows[elem.rows]
                    x["ov"] = g_obs[
                        jnp.asarray(arena.g_obs_indices(orows, g.observed_dim))
                    ]
                    x["om"] = g_mask[jnp.asarray(orows)]
            xs.append(x)
        xs = tuple(xs)

        init = {}
        for (we, wr), slots in reg_init_slots.items():
            wg = metas[we][1]
            if elems[we].kind == "disc":
                init[f"{we}:{wr}"] = messages[
                    jnp.asarray(arena.message_indices(slots, wg.cards[wr]))
                ]
            elif elems[we].kind == "ef":
                init[f"{we}:{wr}"] = ef_eta[jnp.asarray(slots)]
            else:
                d = wg.dims[wr]
                init[f"{we}:{wr}:e"] = g_eta[jnp.asarray(arena.g_eta_indices(slots, d))]
                init[f"{we}:{wr}:l"] = g_lam[jnp.asarray(arena.g_lam_indices(slots, d))]

        def body(prev, x_t):
            cur: dict = {}
            ys = {}

            def reg(cp, pair):
                _, we, wr, wc, d = cp
                src = prev if d == 1 else cur
                if pair:
                    return (
                        src[f"{we}:{wr}:e"][wc],
                        src[f"{we}:{wr}:l"][wc],
                    )
                return src[f"{we}:{wr}"][wc]

            def read(ei, r, pair):
                plan = plans[ei][r]
                if plan[0] == "table":
                    if pair:
                        return x_t[ei][f"se{r}"], x_t[ei][f"sl{r}"]
                    return x_t[ei][f"s{r}"]
                if plan[0] == "direct":
                    _, we, wr, d = plan
                    src = prev if d == 1 else cur
                    if pair:
                        return src[f"{we}:{wr}:e"], src[f"{we}:{wr}:l"]
                    return src[f"{we}:{wr}"]
                cols = plan[1]
                si = 0
                pieces = []
                for cp in cols:
                    if cp[0] == "s":
                        if pair:
                            pieces.append(
                                (x_t[ei][f"se{r}"][si], x_t[ei][f"sl{r}"][si])
                            )
                        else:
                            pieces.append(x_t[ei][f"s{r}"][si])
                        si += 1
                    else:
                        pieces.append(reg(cp, pair))
                if pair:
                    return (
                        jnp.stack([p[0] for p in pieces]),
                        jnp.stack([p[1] for p in pieces]),
                    )
                return jnp.stack(pieces)

            for ei, (elem, g, _, _) in enumerate(metas):
                if elem.kind == "disc":
                    k_local = max(g.cards)
                    blocks = [
                        jnp.pad(
                            read(ei, r, False),
                            ((0, 0), (0, k_local - card)),
                            constant_values=NEG,
                        )
                        for r, card in enumerate(g.cards)
                    ]
                    inc = jnp.stack(blocks, axis=-2)
                    gp = {
                        key[1:]: value
                        for key, value in x_t[ei].items()
                        if key.startswith("p")
                    }
                    out = executors.discrete(g, inc, gp, op).outgoing
                    out = jnp.maximum(out, NEG)
                    for r, card in enumerate(g.cards):
                        block = out[:, r, :card]
                        cur[f"{ei}:{r}"] = block
                        ys[f"{ei}:{r}"] = block
                elif elem.kind == "ef":
                    inc = jnp.stack(
                        [read(ei, r, False) for r in range(len(g.roles))], axis=-2
                    )
                    gp = {k[1:]: v for k, v in x_t[ei].items() if k.startswith("p")}
                    out = executors.exponential_family(
                        g, inc, gp, x_t[ei].get("ov"), x_t[ei].get("om")
                    ).outgoing
                    out = out.astype(inc.dtype)
                    for r in range(len(g.roles)):
                        block = out[:, r]
                        cur[f"{ei}:{r}"] = block
                        ys[f"{ei}:{r}"] = block
                else:
                    width = max(g.dims)
                    eb, lb = [], []
                    for r, d in enumerate(g.dims):
                        e, l = read(ei, r, True)
                        eb.append(jnp.pad(e, ((0, 0), (0, width - d))))
                        lb.append(jnp.pad(l, ((0, 0), (0, width - d), (0, width - d))))
                    inc_e, inc_l = jnp.stack(eb, axis=-2), jnp.stack(lb, axis=-3)
                    par = {k[1:]: v for k, v in x_t[ei].items() if k.startswith("p")}
                    y = mk = None
                    if g.obs_rows is not None:
                        y, mk = x_t[ei]["ov"], x_t[ei]["om"]
                    outs = executors.gaussian(g, inc_e, inc_l, par, y, mk).outgoing
                    for r, (oe, ol) in enumerate(outs):
                        d = g.dims[r]
                        oe, ol = oe[..., :d], ol[..., :d, :d]
                        cur[f"{ei}:{r}:e"], cur[f"{ei}:{r}:l"] = oe, ol
                        ys[f"{ei}:{r}:e"], ys[f"{ei}:{r}:l"] = oe, ol
            new_carry = {k: cur[k] for k in prev}
            return new_carry, ys

        _, ys = jax.lax.scan(body, init, xs)

        for ei, (elem, g, _, sc) in enumerate(metas):
            if elem.kind == "disc":
                for r, card in enumerate(g.cards):
                    idx = jnp.asarray(arena.message_indices(sc[:, :, r], card))
                    messages = messages.at[idx].set(ys[f"{ei}:{r}"])
            elif elem.kind == "ef":
                for r in range(len(g.roles)):
                    ef_eta = ef_eta.at[jnp.asarray(sc[:, :, r])].set(ys[f"{ei}:{r}"])
            else:
                for r, d in enumerate(g.dims):
                    ei_idx = jnp.asarray(arena.g_eta_indices(sc[:, :, r], d))
                    li_idx = jnp.asarray(arena.g_lam_indices(sc[:, :, r], d))
                    g_eta = g_eta.at[ei_idx].set(ys[f"{ei}:{r}:e"])
                    g_lam = g_lam.at[li_idx].set(ys[f"{ei}:{r}:l"])
        return messages, g_eta, g_lam, ef_eta

    return fn


def _scan_stage_fn(arena: Arena, stage, op, executors=SCHEDULED_EXECUTORS):
    """Pick the ScanStage executor: registerized when the dependency
    analysis succeeds (chains always do), flat-carry otherwise."""
    plans = _classify_scan(arena, stage)
    if plans is not None:
        fn = _registerized_scan_fn(arena, stage, plans, op, executors)
        fn.registerized = True
        return fn
    fn = _flat_scan_fn(arena, stage, op, executors)
    fn.registerized = False
    return fn


def _flat_scan_fn(arena: Arena, stage, op, executors=SCHEDULED_EXECUTORS):
    """Fallback ScanStage executor: S structurally identical rounds as one
    ``lax.scan`` over the flat arrays. All flat value indices are
    precomputed [S, ...] tables fed through the scan inputs, so
    trace/compile cost is O(1) in S while the executed work stays the same
    as the unrolled rounds."""
    metas, xs = [], []
    for elem in stage.elems:
        if elem.kind == "disc":
            g = arena.groups[elem.group]
            slots_g, slots_s = g.gather[elem.rows], g.scatter[elem.rows]  # [S, m, R]
            x = {
                "rows": jnp.asarray(elem.rows),
                "gv": tuple(
                    jnp.asarray(arena.message_indices(slots_g[:, :, r], card))
                    for r, card in enumerate(g.cards)
                ),
                "sv": tuple(
                    jnp.asarray(arena.message_indices(slots_s[:, :, r], card))
                    for r, card in enumerate(g.cards)
                ),
            }
        elif elem.kind == "ef":
            g = arena.ef_groups[elem.group]
            x = {
                "rows": jnp.asarray(elem.rows),
                "gs": jnp.asarray(g.gather[elem.rows]),  # [S, m, R] ef slots
                "ss": jnp.asarray(g.scatter[elem.rows]),
            }
            if g.obs_rows is not None:
                x["or"] = jnp.asarray(g.obs_rows[elem.rows])
        else:
            g = arena.gauss_groups[elem.group]
            slots_g, slots_s = g.gather[elem.rows], g.scatter[elem.rows]
            x = {
                "rows": jnp.asarray(elem.rows),
                "ge": tuple(
                    jnp.asarray(arena.g_eta_indices(slots_g[:, :, r], d))
                    for r, d in enumerate(g.dims)
                ),
                "gl": tuple(
                    jnp.asarray(arena.g_lam_indices(slots_g[:, :, r], d))
                    for r, d in enumerate(g.dims)
                ),
                "se": tuple(
                    jnp.asarray(arena.g_eta_indices(slots_s[:, :, r], d))
                    for r, d in enumerate(g.dims)
                ),
                "sl": tuple(
                    jnp.asarray(arena.g_lam_indices(slots_s[:, :, r], d))
                    for r, d in enumerate(g.dims)
                ),
            }
            if g.obs_rows is not None:
                orows = g.obs_rows[elem.rows]  # [S, m]
                x["ov"] = jnp.asarray(arena.g_obs_indices(orows, g.observed_dim))
                x["mr"] = jnp.asarray(orows)
        metas.append((elem.kind, g))
        xs.append(x)
    xs = tuple(xs)

    def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
        def body(carry, x_t):
            messages, g_eta, g_lam, ef_eta = carry
            for (kind, g), x in zip(metas, x_t):
                if kind == "disc":
                    k_local = max(g.cards)
                    blocks = [
                        jnp.pad(
                            messages[x["gv"][r]],
                            ((0, 0), (0, k_local - card)),
                            constant_values=NEG,
                        )
                        for r, card in enumerate(g.cards)
                    ]
                    inc = jnp.stack(blocks, axis=-2)
                    out = executors.discrete(
                        g, inc, executors.parameters(g, params, x["rows"]), op
                    ).outgoing
                    out = jnp.maximum(out, NEG)
                    for r, card in enumerate(g.cards):
                        messages = messages.at[x["sv"][r]].set(out[:, r, :card])
                elif kind == "ef":
                    obs = ef_obs[x["or"]] if "or" in x else None
                    mask = ef_mask[x["or"]] if "or" in x else None
                    out = executors.exponential_family(
                        g,
                        ef_eta[x["gs"]],
                        executors.ef_parameters(g, params, x["rows"]),
                        obs,
                        mask,
                    ).outgoing.astype(ef_eta.dtype)
                    ef_eta = ef_eta.at[x["ss"]].set(out)
                else:
                    width = max(g.dims)
                    eb = [
                        jnp.pad(g_eta[x["ge"][r]], ((0, 0), (0, width - d)))
                        for r, d in enumerate(g.dims)
                    ]
                    lb = [
                        jnp.pad(
                            g_lam[x["gl"][r]],
                            ((0, 0), (0, width - d), (0, width - d)),
                        )
                        for r, d in enumerate(g.dims)
                    ]
                    inc_e, inc_l = jnp.stack(eb, axis=-2), jnp.stack(lb, axis=-3)
                    par = executors.parameters(g, params, x["rows"])
                    y = mk = None
                    if g.obs_rows is not None:
                        y, mk = g_obs[x["ov"]], g_mask[x["mr"]]
                    outs = executors.gaussian(g, inc_e, inc_l, par, y, mk).outgoing
                    for r, (oe, ol) in enumerate(outs):
                        d = g.dims[r]
                        g_eta = g_eta.at[x["se"][r]].set(oe[..., :d])
                        g_lam = g_lam.at[x["sl"][r]].set(ol[..., :d, :d])
            return (messages, g_eta, g_lam, ef_eta), None

        (messages, g_eta, g_lam, ef_eta), _ = jax.lax.scan(
            body, (messages, g_eta, g_lam, ef_eta), xs
        )
        return messages, g_eta, g_lam, ef_eta

    return fn


def make_support_check(
    arena: Arena, executors: ScheduledExecutors = SCHEDULED_EXECUTORS
):
    """Jitted mirror of ``Engine._has_zero_support`` for constraint-free
    models (every two-pass model): True iff an edge or factor belief has
    empty exact support. Semantics are identical to the NumPy check: NEG is
    the one reserved zero sentinel and every other finite log value has
    support, irrespective of magnitude."""
    require_execution_available(arena, operation="make_support_check")
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
    require_execution_available(arena, operation="make_exact_unit")
    support_bad = make_support_check(arena, executors)
    is_max = arena.semiring == "max"

    def exact_unit(
        messages, g_eta, g_lam, g_obs, g_mask, ef_eta, ef_obs, ef_mask, params
    ):
        pre_bad = support_bad(messages, params)
        m, e, l, f = propagate(
            messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params
        )
        vm, ve, vl, vf, _, _ = raw_step(
            m, e, l, g_obs, g_mask, f, ef_obs, ef_mask, params, 0.0, False
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
        return m, e, l, f, delta, rel, fe, pre_bad, support_bad(m, params)

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
    require_execution_available(arena, operation="make_propagate")
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
        for fn in stage_fns:
            messages, g_eta, g_lam, ef_eta = fn(
                messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params
            )
        return messages, g_eta, g_lam, ef_eta

    propagate.scan_registerized = tuple(
        fn.registerized for fn in stage_fns if hasattr(fn, "registerized")
    )
    return propagate
