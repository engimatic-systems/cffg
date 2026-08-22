"""Message-passing schedules: topology-aware alternatives to flooding
(cff-d56d).

Synchronous flooding updates every directed message every sweep, so a tree
that admits exact two-pass BP costs O(E * diameter) instead of O(E) — O(T^2)
on chains. This module builds a *schedule* from a compiled Arena. The factor
graph over the schedulable arenas (discrete BP, Gaussian, EF conjugate-BP)
is partitioned into islands, and each island gets the strongest schedule its
topology admits:

- ``two_pass`` (whole model): every island is a forest and no iterative
  structure exists. Factors fire in BFS-depth rounds rooted at a diameter
  endpoint, inward then outward; each factor fires exactly twice, so each
  directed message is written O(1) times and one execution reaches the BP
  fixed point (see ``build_rounds``). The engine certifies the result with
  one synchronous sweep.
- ``colored``: pure BP with cycles. Forest islands still run their exact
  two-pass once (``prime_stages``); cyclic islands iterate *colored* sweeps
  under the normal convergence loop — factors are greedily colored so no
  two adjacent factors share a color, and colors fire sequentially
  (Gauss–Seidel across colors, Jacobi within one), which propagates
  information further per sweep than flooding.
- ``mixed``: iterative structures present (mean-field beads, GFE
  composites, hard constraints, hybrid mixtures, recurrence links,
  gaussian_prec). Each outer
  iteration updates the iterative phase synchronously, then refreshes the
  BP islands before the loop evaluates the global residual and FE: forest
  islands are re-equilibrated *exactly* (a full two-pass), cyclic islands
  advance one colored sweep — they equilibrate across outer iterations,
  which the residual and FE stationarity criteria measure. A forest island
  can therefore never be the stale thing convergence is judged on.
- ``flood``: the synchronous whole-graph sweep, retained as an explicit
  option and as the fallback for δ form constraints (whose per-sweep
  hardening is defined against flooding) and models with no schedulable
  structure at all. ``Schedule.reason`` records what forced it.

Determinism (serialization contract): the schedule is a pure function of
the Arena, which is a pure function of the template — group buckets, member
stacking orders, round ordering, and coloring are all sorted. Rebuilding
from a template round-trip therefore reproduces the schedule exactly;
``Schedule.describe()`` is the stable textual form tests pin, and
``Schedule.fingerprint`` seals it against the arena structure so the engine
can reject a stale or doctored prebuilt schedule.

Uniform runs of forest rounds (chains produce one long run) are compiled to
``ScanStage``s so the engine can execute them as a single ``lax.scan``
instead of O(T) unrolled stages: compile time stays O(1) in T and the warm
kernel is dispatch-free.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

import numpy as np

from cffg.compiler import Arena

# Rounds shorter than this unroll as batch stages; longer uniform runs
# become one lax.scan (trace/compile cost O(1) in the run length).
SCAN_MIN = 8

Node = tuple[str, int, int]  # (kind "disc"|"gauss"|"ef", group index, member index)


@dataclass(frozen=True)
class BatchStage:
    """Fire a subset of one factor group's members, vectorized in one call."""

    kind: str  # "disc" (arena.groups) | "gauss" (arena.gauss_groups) | "ef"
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
    pass; executing it once yields the exact BP fixed point. kind "colored":
    ``prime_stages`` is the one-shot forest pass and ``stages`` is one
    colored sweep, iterated by the engine's convergence loop. kind "mixed":
    ``stages`` refreshes the BP islands (forest passes exactly, cyclic
    islands by one colored sweep) after the iterative phase inside every
    outer iteration (and once up front as priming); ``forest_stages`` is the
    conditional reconstruction used by a reduced outer chart. kind "flood":
    no stages; the engine iterates its synchronous sweep."""

    kind: str  # "two_pass" | "colored" | "mixed" | "flood"
    reason: str = ""
    rounds: int = 0  # forest inward-pass depth (diagnostic)
    colors: int = 0  # colors over cyclic islands (diagnostic)
    stages: tuple = field(default_factory=tuple)
    prime_stages: tuple = field(default_factory=tuple)
    forest_stages: tuple = field(default_factory=tuple)
    fingerprint: str = ""  # sha256 over the arena structure + describe();
    # the engine rejects a prebuilt Schedule whose fingerprint does not match
    # a fresh build for its arena, so a doctored or stale schedule (a missing
    # phase, a different model) cannot execute at all.

    def describe(self) -> str:
        """Stable textual form (the deterministic-reconstruction contract)."""
        head = f"{self.kind} rounds={self.rounds} colors={self.colors}"
        if self.reason:
            head += f" reason={self.reason}"
        lines = [head]

        def emit(label: str, stages: tuple) -> None:
            for s in stages:
                if isinstance(s, BatchStage):
                    lines.append(
                        f"  {label} batch {s.kind}[{s.group}] rows={s.rows.tolist()}"
                    )
                else:
                    elems = " ".join(
                        f"{e.kind}[{e.group}]x{e.rows.shape[1]}" for e in s.elems
                    )
                    lines.append(f"  {label} scan S={s.length} elems=({elems})")

        emit("prime", self.prime_stages)
        emit("forest", self.forest_stages)
        emit("iter", self.stages)
        return "\n".join(lines)


def _arena_signature(arena: Arena) -> str:
    """Deterministic structural summary of an arena for fingerprinting."""
    h = hashlib.sha256()
    h.update(
        f"{arena.template_name}|{arena.semiring}|{arena.n_slots}|"
        f"{arena.n_gslots}|{arena.n_efslots}".encode()
    )
    for kind, groups in (
        ("d", arena.groups),
        ("p", arena.dirichlet_groups),
        ("g", arena.gauss_groups),
        ("e", arena.ef_groups),
    ):
        for g in groups:
            h.update(f"{kind}|{g.name}|{len(g.members)}".encode())
            gather = getattr(g, "gather", None)
            if gather is None:
                gather = g.gather_both
            h.update(np.ascontiguousarray(gather).tobytes())
            h.update(np.ascontiguousarray(g.scatter).tobytes())
    for g in arena.gfe_groups:
        h.update(
            f"gfe|{g.name}|{len(g.members)}|{g.learned}|{g.mu3}|"
            f"{g.param_blobs}|{g.goal_learned}|{g.goal_members}|"
            f"{g.goal_param_edges}|"
            f"{g.goal_prior_blobs}|{g.goal_tie_groups}".encode()
        )
        for rows in (
            g.gather_z,
            g.scatter_z,
            g.scatter_x,
            g.goal_prior_scatter,
            g.goal_count_scatter,
        ):
            h.update(np.ascontiguousarray(rows).tobytes())
    for plan in arena.mu3_groups:
        h.update(
            f"mu3|{plan.name}|{plan.evaluator}|{plan.param_blob}|"
            f"{plan.count_key}|{plan.cards}".encode()
        )
        h.update(np.ascontiguousarray(plan.rows).tobytes())
    for group in arena.hybrid_gfe_groups:
        h.update(
            f"gfhybrid|{group.name}|{group.type}|{group.dims}|"
            f"{group.k_sel}|{group.members}|{group.goal_members}".encode()
        )
        for rows in (
            group.g_gather,
            group.g_scatter,
            group.s_gather,
            group.s_scatter,
        ):
            h.update(np.ascontiguousarray(rows).tobytes())
    for group in arena.recurrence_groups:
        h.update(
            f"recurrence|{group.name}|{group.cards}|{group.dim}|"
            f"{group.members}".encode()
        )
        for rows in (
            group.d_gather_both,
            group.d_scatter,
            group.drive_gather_both,
            group.drive_scatter,
            group.component_ids,
        ):
            h.update(np.ascontiguousarray(rows).tobytes())
    for component in arena.recurrence_components:
        h.update(
            (
                f"recurrence-component|{component.members}|"
                f"{component.categorical_factors}|{component.acyclic}"
            ).encode()
        )
    for plan in arena.structured_hybrid_plans:
        topology = plan.topology
        h.update(
            (
                f"structured-hybrid|{plan.name}|{topology.recurrence_members}|"
                f"{topology.dynamics_members}|{topology.emission_members}|"
                f"{topology.regime_edges}|{topology.state_edges}|"
                f"{topology.cardinality}|{topology.dimension}"
            ).encode()
        )
        h.update(np.ascontiguousarray(plan.observation_rows).tobytes())
    return h.hexdigest()


def _stage_digest(h, stages: tuple) -> None:
    """Feed every stage's full row tables (shape, dtype, bytes) into the
    digest — the textual describe() alone cannot distinguish permuted or
    reversed scan row tables (2026-07-27 review, second round)."""
    for s in stages:
        if isinstance(s, ScanStage):
            h.update(f"scan|{s.length}".encode())
            elems = s.elems
        else:
            h.update(b"batch")
            elems = (s,)
        for e in elems:
            rows = np.ascontiguousarray(e.rows)
            h.update(f"|{e.kind}|{e.group}|{rows.dtype.str}|{rows.shape}|".encode())
            h.update(rows.tobytes())


def _fingerprint(arena: Arena, sched: "Schedule") -> str:
    h = hashlib.sha256()
    h.update(_arena_signature(arena).encode())
    h.update(b"\n")
    h.update(sched.describe().encode())
    h.update(b"\nprime\n")
    _stage_digest(h, sched.prime_stages)
    h.update(b"\niter\n")
    _stage_digest(h, sched.stages)
    h.update(b"\nforest\n")
    _stage_digest(h, sched.forest_stages)
    return h.hexdigest()


def _iterative_structures(arena: Arena) -> list[str]:
    """Structures that iterate to a fixed point (the mixed-mode phase):
    they run in the restricted synchronous step each outer iteration."""
    reasons = []
    if any(g.mode == "vmp" for g in arena.groups):
        reasons.append("mean-field (vmp) groups")
    if arena.dirichlet_groups:
        reasons.append("Dirichlet parameter-edge VMP groups")
    if arena.constraint_groups:
        reasons.append("hard-constraint factors")
    if arena.gfe_groups or arena.gfe_tm_groups or arena.hybrid_gfe_groups:
        reasons.append("GFE composites")
    if arena.mix_groups:
        reasons.append("hybrid mixture groups")
    if arena.recurrence_groups:
        reasons.append("recurrence-link VMP groups")
    if arena.prec_groups:
        reasons.append("gaussian_prec VMP groups")
    return reasons


def _island_nodes_and_links(
    arena: Arena,
) -> tuple[list[Node], list[tuple[Node, Node]], int]:
    """Factor nodes of the schedulable subgraph, its internal links, and the
    count of boundary edges (one schedulable endpoint, one iterative)."""
    nodes: list[Node] = []
    disc_owner: dict[int, Node] = {}
    gauss_owner: dict[int, Node] = {}
    ef_owner: dict[int, Node] = {}
    for gi, g in enumerate(arena.groups):
        if g.mode != "bp":
            continue
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
    for gi, g in enumerate(arena.ef_groups):
        for mi in range(len(g.members)):
            node = ("ef", gi, mi)
            nodes.append(node)
            for r in range(len(g.roles)):
                ef_owner[int(g.scatter[mi, r])] = node

    links: list[tuple[Node, Node]] = []
    boundary = 0
    for e in arena.edges.values():
        if not e.interior:
            continue
        if e.dim:
            owner = gauss_owner
        elif e.ef:
            owner = ef_owner
        elif e.param_shape:
            owner = {}
        else:
            owner = disc_owner
        a, b = owner.get(e.slot0), owner.get(e.slot1)
        if a is not None and b is not None:
            links.append((a, b))
        elif a is not None or b is not None:
            boundary += 1
    return nodes, links, boundary


def build_rounds(
    nodes: list[Node], links: list[tuple[Node, Node]]
) -> list[list[Node]] | None:
    """Depth-level firing rounds for forest graphs; None on a cycle.

    Each component is rooted at a diameter endpoint (double BFS) and its
    nodes fire by decreasing BFS depth: when a node fires, all its children
    fired in earlier rounds, so its parent-directed message is final by
    induction; the outward pass replays the rounds in reverse, when every
    input is final. Firing writes *all* of a node's outgoing messages —
    non-final inward writes are never read before their outward rewrite,
    because tree edges connect consecutive depths only (no two same-round
    nodes are adjacent).

    Rooting at a diameter endpoint keeps a chain a *single* strand: rounds
    step through it monotonically, which is what lets the engine fuse the
    round pattern into a lax.scan with register-carried messages instead of
    per-step scatters into the flat arena."""
    adj, cyclic = _adjacency(nodes, links)
    if cyclic:
        return None
    return _depth_rounds(nodes, adj)


def _adjacency(
    nodes: list[Node], links: list[tuple[Node, Node]]
) -> tuple[dict[Node, list[Node]], set[Node]]:
    """Sorted adjacency plus the set of nodes living in cyclic components."""
    adj: dict[Node, list[Node]] = {n: [] for n in nodes}
    parent: dict[Node, Node] = {}
    cyclic_roots: set[Node] = set()

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
            cyclic_roots.add(ra)
        else:
            parent[ra] = rb
            if rb in cyclic_roots or ra in cyclic_roots:
                cyclic_roots.add(rb)
        adj[a].append(b)
        adj[b].append(a)
    for n in adj:
        adj[n].sort()
    cyclic = {n for n in nodes if find(n) in {find(r) for r in cyclic_roots}}
    return adj, cyclic


def _depth_rounds(
    nodes: list[Node], adj: dict[Node, list[Node]]
) -> list[list[Node]]:
    """BFS-depth rounds (deepest level first) for forest nodes."""

    def bfs_depths(start: Node, allowed: set[Node]) -> dict[Node, int]:
        depths = {start: 0}
        frontier = [start]
        while frontier:
            nxt = []
            for n in frontier:
                for nb in adj[n]:
                    if nb in allowed and nb not in depths:
                        depths[nb] = depths[n] + 1
                        nxt.append(nb)
            frontier = nxt
        return depths

    allowed = set(nodes)
    visited: set[Node] = set()
    by_round: dict[int, list[Node]] = {}
    for start in sorted(nodes):
        if start in visited:
            continue
        probe = bfs_depths(start, allowed - visited)
        root = min(n for n, d in probe.items() if d == max(probe.values()))
        depths = bfs_depths(root, allowed - visited)
        visited.update(depths)
        deepest = max(depths.values())
        for n, d in depths.items():  # deepest level fires first
            by_round.setdefault(deepest - d, []).append(n)
    return [sorted(by_round[i]) for i in range(len(by_round))]


def _color_nodes(
    nodes: list[Node], adj: dict[Node, list[Node]]
) -> list[list[Node]]:
    """Greedy graph coloring: no two adjacent factors share a color, so a
    sweep firing colors in order is Gauss–Seidel across colors. Sorted node
    order makes the coloring deterministic."""
    color: dict[Node, int] = {}
    for n in sorted(nodes):
        used = {color[nb] for nb in adj[n] if nb in color}
        c = 0
        while c in used:
            c += 1
        color[n] = c
    classes: dict[int, list[Node]] = {}
    for n, c in color.items():
        classes.setdefault(c, []).append(n)
    return [sorted(classes[c]) for c in range(len(classes))]


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
    """Fuse maximal periodic runs of rounds into ScanStages; everything else
    stays unrolled batch stages.

    A run of rounds whose structure signatures repeat with period p (chains
    give p = 2: a {transition, emission} level alternating with an
    {equality} level) becomes one ScanStage whose scan step executes p
    consecutive rounds; a step's elems keep the round order, so intra-step
    reads still happen after the writes they depend on."""
    out: list = []
    sigs = [_signature(r) for r in per_round]
    i = 0
    while i < len(per_round):
        best = None  # (steps, period)
        for period in (1, 2, 3, 4):
            n = 0
            while (
                i + n < len(per_round)
                and sigs[i + n] == sigs[i + (n % period)]
            ):
                n += 1
            steps = n // period
            if steps * period >= max(SCAN_MIN, 2 * period) and (
                best is None or steps * period > best[0] * best[1]
            ):
                best = (steps, period)
        if best is None:
            out.extend(per_round[i])
            i += 1
            continue
        steps, period = best
        elems = []
        for q in range(period):
            for k, st in enumerate(per_round[i + q]):
                elems.append(
                    ScanElem(
                        kind=st.kind,
                        group=st.group,
                        rows=np.stack(
                            [
                                per_round[i + q + s * period][k].rows
                                for s in range(steps)
                            ]
                        ),
                    )
                )
        out.append(ScanStage(length=steps, elems=tuple(elems)))
        i += steps * period
    return out


def build_schedule(arena: Arena, mode: str = "auto") -> Schedule:
    """Build the message-passing schedule for ``arena``.

    mode "auto" picks the strongest applicable schedule: two_pass when the
    whole model is an acyclic BP graph over the discrete/Gaussian/EF arenas,
    colored when pure BP has cycles, mixed when iterative structures are
    present, flooding otherwise (δ form constraints, or nothing schedulable).
    mode "flood" always floods. mode "two_pass" requires the exact directed
    schedule and raises ``ValueError`` when the model is outside its
    contract."""
    if mode not in ("auto", "flood", "two_pass"):
        raise ValueError(f"schedule mode must be auto|flood|two_pass, got {mode!r}")

    def sealed(sched: Schedule) -> Schedule:
        return replace(sched, fingerprint=_fingerprint(arena, sched))

    if mode == "flood":
        return sealed(Schedule(kind="flood", reason="requested"))

    if len(arena.delta_pairs):
        reason = "delta form constraints harden per synchronous sweep"
        if mode == "two_pass":
            raise ValueError(f"two_pass schedule unavailable: {reason}")
        return sealed(Schedule(kind="flood", reason=reason))

    iterative = _iterative_structures(arena)
    nodes, links, boundary = _island_nodes_and_links(arena)
    if boundary and not iterative:
        # An interior edge with exactly one schedulable owner can only face
        # an iterative structure; anything else is a structure this build
        # does not understand — refuse to schedule it.
        reason = "interior edge without a schedulable owner"
        if mode == "two_pass":
            raise ValueError(f"two_pass schedule unavailable: {reason}")
        return sealed(Schedule(kind="flood", reason=reason))
    if not nodes:
        reason = "no schedulable BP structure (" + "; ".join(iterative) + ")"
        if mode == "two_pass":
            raise ValueError(f"two_pass schedule unavailable: {reason}")
        return sealed(Schedule(kind="flood", reason=reason))

    adj, cyclic = _adjacency(nodes, links)
    forest_nodes = [n for n in nodes if n not in cyclic]
    per_round = [_round_stages(r) for r in _depth_rounds(forest_nodes, adj)]
    forest_passes = tuple(
        _compile_rounds(per_round) + _compile_rounds(list(reversed(per_round)))
    )
    colored_classes = _color_nodes(sorted(cyclic), adj) if cyclic else []
    colored_sweep = tuple(
        st for cls in colored_classes for st in _round_stages(cls)
    )
    n_colors = len(colored_classes)

    if not iterative and not cyclic:
        return sealed(
            Schedule(
                kind="two_pass",
                rounds=len(per_round),
                stages=forest_passes,
            )
        )
    if mode == "two_pass":
        raise ValueError(
            "two_pass schedule unavailable: "
            + "; ".join(
                (["cyclic BP components"] if cyclic else []) + iterative
            )
        )
    if not iterative:
        return sealed(
            Schedule(
                kind="colored",
                reason=f"cyclic BP components ({len(cyclic)} factors)",
                rounds=len(per_round),
                colors=n_colors,
                stages=colored_sweep,
                prime_stages=forest_passes,
            )
        )
    island_stages = forest_passes + colored_sweep
    return sealed(
        Schedule(
            kind="mixed",
            reason="; ".join(iterative),
            rounds=len(per_round),
            colors=n_colors,
            stages=island_stages,
            prime_stages=island_stages,
            forest_stages=forest_passes,
        )
    )


def eliminated_slot_sets(arena: Arena) -> dict[str, frozenset[int]]:
    """Message slots reconstructed by exact conditional forest passes.

    Cyclic BP-island messages deliberately remain outer coordinates. Only
    forest factors have a unique conditional fixed point that the mixed-GFE
    chart may eliminate.
    """
    nodes, links, _ = _island_nodes_and_links(arena)
    _, cyclic = _adjacency(nodes, links)
    slots: dict[str, set[int]] = {
        "messages": set(),
        "gaussian": set(),
        "ef": set(),
    }
    for edge in arena.edges.values():
        if edge.interior:
            continue
        key = "gaussian" if edge.dim else "ef" if edge.ef else "messages"
        slots[key].add(edge.slot1)
    plans = {
        "disc": (arena.groups, "messages"),
        "gauss": (arena.gauss_groups, "gaussian"),
        "ef": (arena.ef_groups, "ef"),
    }
    for kind, group, row in nodes:
        if (kind, group, row) in cyclic:
            continue
        groups, key = plans[kind]
        slots[key].update(
            int(slot)
            for slot in np.asarray(groups[group].scatter[row]).reshape(-1)
        )
    return {key: frozenset(values) for key, values in slots.items()}
