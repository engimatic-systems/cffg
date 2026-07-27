"""Message-passing schedules: topology-aware alternatives to flooding
(cff-d56d).

Synchronous flooding updates every directed message every sweep, so a tree
that admits exact two-pass BP costs O(E * diameter) instead of O(E) — O(T^2)
on chains. This module builds a *schedule* from a compiled Arena:

- ``two_pass``: an exact inward/outward directed schedule for acyclic BP
  subgraphs over the discrete and Gaussian arenas. Factors fire in leaf-peel
  rounds toward the interior, then in reverse; each factor fires exactly
  twice, so each directed message is written O(1) times and one execution
  reaches the BP fixed point (see the correctness note on ``build_rounds``).
- ``flood``: the synchronous whole-graph sweep (`Engine.step`), retained as
  an explicit schedule for everything the directed analysis does not cover:
  cyclic graphs, mean-field (vmp) beads, GFE composites, hard constraints,
  hybrid/precision/EF groups, and delta form constraints. ``Schedule.reason``
  records which boundary forced the fallback, so callers and docs can see
  the phase semantics they are getting (docs/ARCHITECTURE.md "Schedules").

Determinism (serialization contract): the schedule is a pure function of the
Arena, which is a pure function of the template — group buckets, member
stacking orders, and round ordering are all sorted. Rebuilding from a
template round-trip therefore reproduces the schedule exactly;
``Schedule.describe()`` is the stable textual form tests pin.

Uniform runs of rounds (chains produce one long run) are compiled to
``ScanStage``s so the engine can execute them as a single ``lax.scan``
instead of O(T) unrolled stages: compile time stays O(1) in T and the warm
kernel is dispatch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cffg.compiler import Arena

# Rounds shorter than this unroll as batch stages; longer uniform runs
# become one lax.scan (trace/compile cost O(1) in the run length).
SCAN_MIN = 8

Node = tuple[str, int, int]  # (kind "disc"|"gauss", group index, member index)


@dataclass(frozen=True)
class BatchStage:
    """Fire a subset of one factor group's members, vectorized in one call."""

    kind: str  # "disc" (arena.groups) | "gauss" (arena.gauss_groups)
    group: int
    rows: np.ndarray  # [m] member indices into the group's stacking order


@dataclass(frozen=True)
class ScanElem:
    """One pattern element of a ScanStage: the same group fires ``rows[s]``
    at scan step s. ``rows`` is [S, m] with constant member count m."""

    kind: str
    group: int
    rows: np.ndarray  # [S, m]


@dataclass(frozen=True)
class ScanStage:
    """S structurally identical rounds executed as one ``lax.scan``.

    Each scan step fires ``elems`` in order; step s of elem e fires members
    ``elems[e].rows[s]``. Semantically identical to the S unrolled rounds."""

    length: int
    elems: tuple[ScanElem, ...]


@dataclass(frozen=True)
class Schedule:
    """A compiled message-passing schedule for one Arena.

    kind "two_pass": ``stages`` is the inward pass followed by the outward
    pass; executing it once yields the exact BP fixed point on the (acyclic)
    graph. kind "flood": ``stages`` is empty and the engine iterates its
    synchronous sweep; ``reason`` names the boundary that forced it."""

    kind: str  # "two_pass" | "flood"
    reason: str = ""
    rounds: int = 0  # inward-pass depth (scales the exactness certificate)
    stages: tuple = field(default_factory=tuple)

    def describe(self) -> str:
        """Stable textual form (the deterministic-reconstruction contract)."""
        if self.kind == "flood":
            return f"flood: {self.reason}" if self.reason else "flood"
        lines = [f"two_pass rounds={self.rounds}"]
        for s in self.stages:
            if isinstance(s, BatchStage):
                lines.append(
                    f"  batch {s.kind}[{s.group}] rows={s.rows.tolist()}"
                )
            else:
                elems = " ".join(
                    f"{e.kind}[{e.group}]x{e.rows.shape[1]}" for e in s.elems
                )
                lines.append(f"  scan S={s.length} elems=({elems})")
        return "\n".join(lines)


def _flood_reasons(arena: Arena) -> list[str]:
    """Structures outside the directed-schedule contract (the documented
    schedule boundaries): each forces synchronous flooding for the whole
    model so no phase boundary can pass a stale message silently."""
    reasons = []
    if any(g.mode == "vmp" for g in arena.groups):
        reasons.append("mean-field (vmp) groups iterate to a fixed point")
    if arena.constraint_groups:
        reasons.append("hard-constraint factors")
    if arena.gfe_groups or arena.gfe_tm_groups:
        reasons.append("GFE composites need outer coordination sweeps")
    if arena.mix_groups:
        reasons.append("hybrid mixture groups")
    if arena.prec_groups:
        reasons.append("gaussian_prec VMP groups iterate to a fixed point")
    if arena.ef_groups:
        reasons.append("EF groups (directed conjugate-BP schedules: follow-up)")
    if len(arena.delta_pairs):
        reasons.append("delta form constraints harden per sweep")
    return reasons


def _bp_nodes_and_edges(
    arena: Arena,
) -> tuple[list[Node], list[tuple[Node, Node]] | None]:
    """Factor nodes of the schedulable subgraph and its interior links.

    Returns (nodes, links); links is None when an interior edge's endpoint
    is not owned by a schedulable group (an unknown structure — flood)."""
    nodes: list[Node] = []
    disc_owner: dict[int, Node] = {}
    gauss_owner: dict[int, Node] = {}
    for gi, g in enumerate(arena.groups):
        for mi in range(len(g.members)):
            node = ("disc", gi, mi)
            nodes.append(node)
            for r in range(len(g.roles)):
                disc_owner[int(g.scatter[mi, r])] = node
    for gi, g in enumerate(arena.gauss_groups):
        for mi in range(len(g.members)):
            node = ("gauss", gi, mi)
            nodes.append(node)
            for r in range(len(g.roles)):
                gauss_owner[int(g.scatter[mi, r])] = node

    links: list[tuple[Node, Node]] = []
    for e in arena.edges.values():
        if not e.interior:
            continue
        owner = gauss_owner if e.dim else disc_owner
        a, b = owner.get(e.slot0), owner.get(e.slot1)
        if a is None or b is None:
            return nodes, None
        links.append((a, b))
    return nodes, links


def build_rounds(
    nodes: list[Node], links: list[tuple[Node, Node]]
) -> list[list[Node]] | None:
    """Leaf-peel firing rounds for the two-pass schedule; None on a cycle.

    Correctness: a factor fires when at most one of its interior edges is
    unresolved (its parent). Firing writes *all* its outgoing messages; only
    the parent-directed one is final, and its inputs (the resolved child
    edges) are final by induction. The outward pass replays the rounds in
    reverse, when every input is final, so after inward + outward each
    directed message holds its fixed-point value. Non-final inward writes
    are never read before their outward rewrite."""
    adj: dict[Node, list[Node]] = {n: [] for n in nodes}
    deg: dict[Node, int] = {n: 0 for n in nodes}
    parent: dict[Node, Node] = {}

    def find(n: Node) -> Node:
        root = n
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(n, n) != n:
            parent[n], n = root, parent[n]
        return root

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra == rb:
            return None  # cycle: directed two-pass is not exact here
        parent[ra] = rb
        adj[a].append(b)
        adj[b].append(a)
        deg[a] += 1
        deg[b] += 1

    fired: set[Node] = set()
    current = sorted(n for n in nodes if deg[n] <= 1)
    rounds: list[list[Node]] = []
    while current:
        rounds.append(current)
        fired.update(current)
        ready: list[Node] = []
        for n in current:
            for nb in adj[n]:
                if nb in fired:
                    continue
                deg[nb] -= 1
                if deg[nb] <= 1 and nb not in ready:
                    ready.append(nb)
        current = sorted(set(ready) - fired)
    assert len(fired) == len(nodes)  # forests always peel completely
    return rounds


def _round_stages(rnd: list[Node]) -> list[BatchStage]:
    """One round's members, bucketed per (kind, group) into batch stages."""
    buckets: dict[tuple[str, int], list[int]] = {}
    for kind, gi, mi in rnd:
        buckets.setdefault((kind, gi), []).append(mi)
    return [
        BatchStage(kind, gi, np.asarray(sorted(rows), dtype=np.int64))
        for (kind, gi), rows in sorted(buckets.items())
    ]


def _signature(stages: list[BatchStage]) -> tuple:
    return tuple((s.kind, s.group, len(s.rows)) for s in stages)


def _compile_rounds(per_round: list[list[BatchStage]]) -> list:
    """Fuse maximal runs of structurally identical rounds into ScanStages;
    everything else stays unrolled batch stages."""
    out: list = []
    i = 0
    while i < len(per_round):
        j = i + 1
        sig = _signature(per_round[i])
        while j < len(per_round) and _signature(per_round[j]) == sig:
            j += 1
        if j - i >= SCAN_MIN:
            elems = tuple(
                ScanElem(
                    kind=st.kind,
                    group=st.group,
                    rows=np.stack([per_round[r][k].rows for r in range(i, j)]),
                )
                for k, st in enumerate(per_round[i])
            )
            out.append(ScanStage(length=j - i, elems=elems))
        else:
            for r in range(i, j):
                out.extend(per_round[r])
        i = j
    return out


def build_schedule(arena: Arena, mode: str = "auto") -> Schedule:
    """Build the message-passing schedule for ``arena``.

    mode "auto": two-pass when the model is an acyclic BP graph over the
    discrete/Gaussian arenas, flooding otherwise. mode "flood": always
    flooding. mode "two_pass": require the directed schedule and raise
    ``ValueError`` when the model is outside its contract."""
    if mode not in ("auto", "flood", "two_pass"):
        raise ValueError(f"schedule mode must be auto|flood|two_pass, got {mode!r}")
    if mode == "flood":
        return Schedule(kind="flood", reason="requested")

    reasons = _flood_reasons(arena)
    rounds = None
    if not reasons:
        nodes, links = _bp_nodes_and_edges(arena)
        if links is None:
            reasons.append("interior edge without a schedulable owner")
        else:
            rounds = build_rounds(nodes, links)
            if rounds is None:
                reasons.append("cycles in the BP graph (iterate flooding)")
    if reasons:
        if mode == "two_pass":
            raise ValueError(
                "two_pass schedule unavailable: " + "; ".join(reasons)
            )
        return Schedule(kind="flood", reason="; ".join(reasons))

    per_round = [_round_stages(r) for r in rounds]
    inward = _compile_rounds(per_round)
    outward = _compile_rounds(list(reversed(per_round)))
    return Schedule(
        kind="two_pass",
        rounds=len(per_round),
        stages=tuple(inward + outward),
    )
