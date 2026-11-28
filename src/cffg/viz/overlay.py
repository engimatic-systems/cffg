"""Directed message writes in one unit of an Engine schedule."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from cffg.core import Template, validate
from cffg.core.ir import FACTOR_TYPES
from cffg.core.serialize import structure_hash

if TYPE_CHECKING:
    from cffg.engine import Engine


@dataclass(frozen=True)
class MessageEvent:
    wire_ref: str
    source_ref: str
    target_ref: str
    source_role: str
    arena: str
    slot: int
    phase: str
    step: int
    batch: int
    gfe: bool


@dataclass(frozen=True)
class ScheduleOverlay:
    kind: str
    reason: str
    events: tuple[MessageEvent, ...]
    structure: str

    def select_step(self, step: int) -> ScheduleOverlay:
        """Select one zero-based logical step, preserving its original identity."""
        events = tuple(e for e in self.events if e.step == step)
        if not events:
            raise ValueError(f"no message writes at schedule step {step}")
        return replace(self, events=events)


def _writes(members, roles, scatter, arena, *, rows=None, gfe=False):
    """Yield logical slots, before the executor expands them into scalar indices."""
    if rows is not None:
        # Slice before conversion: FrozenArray exports a copy, even for scans
        # that fire just one member of a large group at each step.
        scatter = scatter[rows]
        members = tuple(members[i] for i in rows)
    slots = np.asarray(scatter).reshape(len(members), len(roles))
    for member, row in zip(members, slots, strict=True):
        for role, slot in zip(roles, row, strict=True):
            yield arena, int(slot), member, role, gfe


def _synchronous(arena, groups, *, mixed):
    for kind, collection in groups.items():
        for g in collection:
            if mixed:
                if kind == "disc" and g.mode == "bp":
                    continue
                if kind == "ef" or (
                    kind == "gauss" and not (g.kernel and g.kernel.binding.iterative)
                ):
                    continue
            yield from _writes(g.members, g.roles, g.scatter, kind)
    for g in arena.constraint_groups:
        roles = tuple(f"x{i}" for i in range(len(g.cards)))
        yield from _writes(g.members, roles, g.scatter, "disc")
    for g in arena.dirichlet_groups:
        yield from _writes(g.members, g.roles, g.scatter, "disc")
        yield from _writes(g.priors, ("out",), g.prior_scatter, "disc")
        yield from _writes(
            g.members, FACTOR_TYPES[g.type].param_roles, g.count_scatter, "disc"
        )
    for g in arena.gfe_groups:
        yield from _writes(g.members, ("in",), g.scatter_z, "disc", gfe=True)
        yield from _writes(g.members, ("out",), g.scatter_x, "disc", gfe=True)
        if g.goal_learned:
            yield from _writes(
                g.goal_priors, ("out",), g.goal_prior_scatter, "disc", gfe=True
            )
    goal_owners = {
        int(slot): member
        for g in arena.gfe_groups
        for slot, member in zip(g.goal_count_scatter, g.goal_members, strict=True)
    }
    for plan in arena.goal_parameter_plans:
        # write_goal_counts scatters a pooled prediction once to each facing goal.
        for slot in plan.count_scatter:
            yield "disc", int(slot), goal_owners[int(slot)], "statistic", True
    for g in arena.gfe_tm_groups:
        yield from _writes(
            g.members, ("out", "in", "selector"), g.scatter, "disc", gfe=True
        )
    for collection, gfe in ((arena.mix_groups, False), (arena.hybrid_gfe_groups, True)):
        for g in collection:
            roles = tuple(
                role for role in FACTOR_TYPES[g.type].edge_roles if role != "selector"
            )
            yield from _writes(g.members, roles, g.g_scatter, "gauss", gfe=gfe)
            yield from _writes(g.members, ("selector",), g.s_scatter, "disc", gfe=gfe)
    for g in arena.recurrence_groups:
        yield from _writes(g.members, ("out", "in"), g.d_scatter, "disc")
        yield from _writes(g.members, ("drive",), g.drive_scatter, "gauss")
    for g in arena.prec_groups:
        yield from _writes(g.members, ("out",), g.x_scatter, "gauss")
        yield from _writes(g.members, ("prec",), g.tau_scatter, "ef")


def _stage_batches(stages, groups):
    from cffg.schedule import BatchStage

    def batch(stage, rows):
        g = groups[stage.kind][stage.group]
        return _writes(g.members, g.roles, g.scatter, stage.kind, rows=rows)

    for stage in stages:
        if isinstance(stage, BatchStage):
            yield batch(stage, stage.rows)
        else:
            for step in range(stage.length):
                for elem in stage.elems:
                    yield batch(elem, elem.rows[step])


def _batches(arena, schedule):
    groups = {"disc": arena.groups, "gauss": arena.gauss_groups, "ef": arena.ef_groups}
    for batch in _stage_batches(schedule.prime_stages, groups):
        yield "prime", batch
    if schedule.kind in ("mixed", "flood"):
        yield "synchronous", _synchronous(arena, groups, mixed=schedule.kind == "mixed")
    phase = {
        "mixed": "island_refresh",
        "two_pass": "two_pass",
        "colored": "colored",
        "flood": "synchronous",
    }
    for batch in _stage_batches(schedule.stages, groups):
        yield phase[schedule.kind], batch


def derive_schedule_overlay(template: Template, engine: Engine) -> ScheduleOverlay:
    """Describe directed writes without executing the schedule or reading a State."""
    from cffg.schedule import _fingerprint
    from cffg.topology import build_topology

    if template.plates:
        raise ValueError(
            "use expand_plates(template) to locate each compiled copy's message writes"
        )
    arena, schedule = engine.arena, engine.schedule
    if schedule.fingerprint != _fingerprint(arena, schedule):
        raise ValueError(
            "schedule fingerprint does not match the engine arena and stages"
        )
    if set(template.variables) != set(arena.edges):
        raise ValueError("template variables do not match the compiled edges")
    validate(template)
    topology = build_topology(template)
    owners = {}
    for name, e in arena.edges.items():
        declared = topology.edges[name]
        if (e.dim, e.ef, e.card, e.interior) != (
            declared.dim,
            declared.ef,
            declared.card,
            declared.interior,
        ):
            raise ValueError(
                f"template domain or incidence disagrees with compiled edge {name!r}"
            )
        kind = "gauss" if e.dim else "ef" if e.ef else "disc"
        ends = topology.endpoints[name]
        for side, (factor, role) in enumerate(ends):
            slot = (e.slot0, e.slot1)[side]
            target = (
                f"factor:{ends[1 - side][0]}"
                if len(ends) == 2
                else f"port:{topology.ports_by_variable[name][0].name}"
            )
            owners[kind, slot] = name, factor, role, target

    events = []
    for step, (phase, batch) in enumerate(_batches(arena, schedule)):
        for kind, slot, factor, role, gfe in batch:
            owner = owners.get((kind, slot))
            if owner is None or owner[2] != role or owner[1] != factor:
                raise ValueError(
                    f"{kind} scatter slot {slot} for {factor}.{role} disagrees with template"
                )
            wire, source, _, target = owner
            events.append(
                MessageEvent(
                    f"var:{wire}",
                    f"factor:{source}",
                    target,
                    role,
                    kind,
                    slot,
                    phase,
                    step,
                    step,
                    gfe,
                )
            )
    return ScheduleOverlay(
        schedule.kind, schedule.reason, tuple(events), structure_hash(template)
    )
