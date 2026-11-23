"""Register-carried and flat scan executors for directed message schedules."""

import jax
import jax.numpy as jnp
import numpy as np

from cffg.compiled import NEG, Arena
from cffg.execution.failures import FailureIndex
from cffg.failure_facts import empty_failure, first_failure
from cffg.execution.owners import SCHEDULED_EXECUTORS


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
    failures = FailureIndex(arena)
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
            x = {"failure_ids": failures.ids(g, elem.rows)}
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

        def body(carry, x_t):
            prev, failure = carry
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
                    delta = executors.discrete(g, inc, gp, op)
                    out = jnp.maximum(delta.outgoing, NEG)
                    for r, card in enumerate(g.cards):
                        block = out[:, r, :card]
                        cur[f"{ei}:{r}"] = block
                        ys[f"{ei}:{r}"] = block
                elif elem.kind == "ef":
                    inc = jnp.stack(
                        [read(ei, r, False) for r in range(len(g.roles))], axis=-2
                    )
                    gp = {k[1:]: v for k, v in x_t[ei].items() if k.startswith("p")}
                    delta = executors.exponential_family(
                        g, inc, gp, x_t[ei].get("ov"), x_t[ei].get("om")
                    )
                    out = delta.outgoing.astype(inc.dtype)
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
                    delta = executors.gaussian(g, inc_e, inc_l, par, y, mk)
                    for r, (oe, ol) in enumerate(delta.outgoing):
                        d = g.dims[r]
                        oe, ol = oe[..., :d], ol[..., :d, :d]
                        cur[f"{ei}:{r}:e"], cur[f"{ei}:{r}:l"] = oe, ol
                        ys[f"{ei}:{r}:e"], ys[f"{ei}:{r}:l"] = oe, ol
                failure = first_failure(
                    failure, failures.delta(
                        g, delta, elem.kind, ids=x_t[ei]["failure_ids"],
                        discrete_dtype=messages.dtype, ef_dtype=ef_eta.dtype,
                        gaussian_dtypes=(g_eta.dtype, g_lam.dtype))
                )
            new_carry = {k: cur[k] for k in prev}
            return (new_carry, failure), ys

        (_, failure), ys = jax.lax.scan(body, (init, empty_failure()), xs)

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
        return messages, g_eta, g_lam, ef_eta, failure

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
    failures = FailureIndex(arena)
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
            messages, g_eta, g_lam, ef_eta, failure = carry
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
                    delta = executors.discrete(
                        g, inc, executors.parameters(g, params, x["rows"]), op
                    )
                    out = jnp.maximum(delta.outgoing, NEG)
                    for r, card in enumerate(g.cards):
                        messages = messages.at[x["sv"][r]].set(out[:, r, :card])
                elif kind == "ef":
                    obs = ef_obs[x["or"]] if "or" in x else None
                    mask = ef_mask[x["or"]] if "or" in x else None
                    delta = executors.exponential_family(
                        g,
                        ef_eta[x["gs"]],
                        executors.ef_parameters(g, params, x["rows"]),
                        obs,
                        mask,
                    )
                    out = delta.outgoing.astype(ef_eta.dtype)
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
                    delta = executors.gaussian(g, inc_e, inc_l, par, y, mk)
                    for r, (oe, ol) in enumerate(delta.outgoing):
                        d = g.dims[r]
                        g_eta = g_eta.at[x["se"][r]].set(oe[..., :d])
                        g_lam = g_lam.at[x["sl"][r]].set(ol[..., :d, :d])
                failure = first_failure(failure, failures.delta(
                    g, delta, kind, rows=x["rows"], discrete_dtype=messages.dtype,
                    ef_dtype=ef_eta.dtype, gaussian_dtypes=(g_eta.dtype, g_lam.dtype)))
            return (messages, g_eta, g_lam, ef_eta, failure), None

        result, _ = jax.lax.scan(
            body, (messages, g_eta, g_lam, ef_eta, empty_failure()), xs
        )
        return result

    return fn
