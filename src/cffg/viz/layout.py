"""Deterministic rectilinear layout: diagram + presentation -> Scene.

The papers lay CFFGs out on an invisible grid (Figs 22/26): a horizontal
*rail* — the state chain — read left to right in time, control priors above
it, observation/goal branches hanging below, parameter terminals tucked into
the nearest free side. This module reproduces that convention:

- **Rail.** The path between glue-port factors (or from the far end of the
  graph to the single glue port). Composite members never sit on the rail.
  Orientation: the first-declared glue port is the left end; a single glue
  port is the right end (the model continues rightward, Fig 22's ellipsis).
- **Branches.** BFS from the rail: a branch attaching at a `selector` role
  goes above (control priors), everything else hangs below, one row per BFS
  depth (obs at +1, its goal at +2 — the Fig 26 stack).
- **Sides.** Structural wires claim box sides by neighbor direction; dangling
  port wires continue away from the rail; parameter terminals take the first
  free side in (left, up, down, right) order — which reproduces where the
  papers put d, A, B, c, and e.

Indexed recurrence chains use fixed left-to-right time columns and a bounded
spring relaxation of the vertical lanes. Wire routes leave both attachment
sides outward and follow clear corridors around unrelated boxes.
"""

from __future__ import annotations

from .labels import CaptionSpace, place_domains, terminal_layout
from .routing import DIRECTIONS, WireRouter
from .temporal_layout import temporal_grid

import re
from collections import deque
from dataclasses import dataclass, replace
from math import isfinite

from cffg.viz.compress import Placement, Presentation
from cffg.viz.diagram import (
    BeadShape,
    Diagram,
    TerminalEnd,
    TerminalKind,
    WireView,
)
from cffg.viz.scene import (
    Bead,
    Box,
    CompositeBox,
    Dome,
    EllipsisMark,
    Label,
    Path,
    Scene,
    Terminal,
)

_TERMINAL_PREFERENCE = ("left", "up", "down", "right")


class OccupiedCell(ValueError):
    """Two positioned factors occupy the same grid cell."""


@dataclass(frozen=True)
class LayoutStyle:
    box: float = 58.0  # box side when beads live inside (levels 0-3)
    compact_box: float = 36.0  # fully compressed boxes (Fig 19 right)
    pitch_x: float = 128.0
    pitch_y: float = 104.0
    compact_pitch_x: float = 100.0
    compact_pitch_y: float = 84.0
    margin: float = 42.0
    bead_r: float = 6.0
    terminal_offset: float = 0.52  # of pitch, box center -> terminal center
    composite_pad: float = 13.0

    temporal_flow: bool = True
    label_size: float = 13.0
    caption_size: float = 9.5
    label_clearance: float = 5.0
    continuous_terminal_radius: float = 6.0

    def __post_init__(self):
        for name in ("label_size", "caption_size", "label_clearance"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def sizes(self, compact: bool) -> tuple[float, float, float]:
        if compact:
            return self.compact_box, self.compact_pitch_x, self.compact_pitch_y
        return self.box, self.pitch_x, self.pitch_y


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", value))


def layout(
    diagram: Diagram, presentation: Presentation, style: LayoutStyle = LayoutStyle(), *, offsets=None
) -> Scene:
    components = _component_diagrams(diagram)
    if not components:
        return _framed(diagram.name, [], style)
    scenes = [_layout_connected(component, presentation, style, offsets) for component in components]
    if len(scenes) == 1:
        return scenes[0]
    items = []
    height = 0.0
    for scene in scenes:
        items.extend(_shift(item, 0.0, height) for item in scene.items)
        height += scene.height
    return Scene(diagram.name, max(scene.width for scene in scenes), height, items,
                 wire_corner_limit=scenes[0].wire_corner_limit)


def _layout_connected(diagram, presentation, style, offsets=None) -> Scene:
    grid, rail = _place_factors(diagram, temporal_flow=style.temporal_flow)
    if offsets:
        grid = {name: (x + offsets.get(name, (0, 0))[0], y + offsets.get(name, (0, 0))[1])
                for name, (x, y) in grid.items()}
        if len(set(grid.values())) != len(grid):
            raise OccupiedCell("Another factor occupies that grid cell")
    sides = _assign_sides(diagram, grid, rail)
    return _emit(diagram, presentation, style, grid, sides)


def _component_diagrams(diagram: Diagram) -> list[Diagram]:
    """Partition by structural wires, ordered by each component's first name."""
    adjacency = _adjacency(diagram)
    owner: dict[str, int] = {}
    components: list[Diagram] = []
    for start in sorted(diagram.nodes, key=natural_key):
        if start in owner:
            continue
        index = len(components)
        components.append(Diagram(diagram.name))
        owner[start] = index
        queue = deque([start])
        while queue:
            for neighbor, _ in adjacency[queue.popleft()]:
                if neighbor not in owner:
                    owner[neighbor] = index
                    queue.append(neighbor)
    # Preserve declaration order within each component, including glue ports.
    for name, node in diagram.nodes.items():
        components[owner[name]].nodes[name] = node
    for name, wire in diagram.wires.items():
        components[owner[wire.node_ends()[0].factor]].wires[name] = wire
    for composite in diagram.composites:
        components[owner[composite.members[0]]].composites.append(composite)
    return components


# --------------------------------------------------------------------------
# Grid placement


def _adjacency(diagram: Diagram) -> dict[str, list[tuple[str, WireView]]]:
    adj: dict[str, list[tuple[str, WireView]]] = {n: [] for n in diagram.nodes}
    for wire in diagram.wires.values():
        ends = wire.node_ends()
        if len(ends) == 2 and ends[0].factor != ends[1].factor:
            adj[ends[0].factor].append((ends[1].factor, wire))
            adj[ends[1].factor].append((ends[0].factor, wire))
    for lst in adj.values():
        lst.sort(key=lambda t: natural_key(t[1].id))
    return adj


def _bfs_path(
    adj: dict[str, list[tuple[str, WireView]]],
    start: str,
    goal: str | None,
    allowed: set[str],
) -> list[str]:
    """Shortest path start->goal, or to the farthest reachable node."""
    prev: dict[str, str] = {}
    seen = {start}
    queue = deque([start])
    last = start
    while queue:
        cur = queue.popleft()
        last = cur
        if cur == goal:
            break
        for nxt, _ in adj[cur]:
            if nxt in allowed and nxt not in seen:
                seen.add(nxt)
                prev[nxt] = cur
                queue.append(nxt)
    end = goal if goal in seen else last
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def _rail(diagram: Diagram, adj) -> list[str]:
    members = {m for c in diagram.composites for m in c.members}
    allowed = set(diagram.nodes) - members or set(diagram.nodes)

    glue: list[str] = []  # factors at glue ports, in wire declaration order
    for wire in diagram.wires.values():
        for end in wire.ends:
            if isinstance(end, TerminalEnd) and end.kind is TerminalKind.ELLIPSIS:
                (node,) = wire.node_ends()
                if node.factor not in glue:
                    glue.append(node.factor)

    if len(glue) >= 2:
        return _bfs_path(adj, glue[0], glue[1], allowed)
    if len(glue) == 1:
        far = _bfs_path(adj, glue[0], None, allowed)[-1]
        return _bfs_path(adj, far, glue[0], allowed)
    start = min(allowed, key=natural_key)
    a = _bfs_path(adj, start, None, allowed)[-1]
    b = _bfs_path(adj, a, None, allowed)[-1]
    path = _bfs_path(adj, a, b, allowed)
    return path if natural_key(path[0]) <= natural_key(path[-1]) else path[::-1]


def _place_factors(diagram: Diagram, *, temporal_flow: bool = True) -> tuple[dict[str, tuple[int, int]], list[str]]:
    adj = _adjacency(diagram)
    temporal = temporal_grid(diagram, adj) if temporal_flow else None
    if temporal is not None:
        return temporal, []
    rail = _rail(diagram, adj)
    grid: dict[str, tuple[int, int]] = {f: (i, 0) for i, f in enumerate(rail)}
    occupied = set(grid.values())

    queue = deque((f, 0) for f in rail)  # (factor, inherited row direction)
    while queue:
        parent, direction = queue.popleft()
        pcol, prow = grid[parent]
        for child, wire in adj[parent]:
            if child in grid:
                continue
            role = next(e.role for e in wire.node_ends() if e.factor == parent)
            d = direction or (-1 if role == "selector" else 1)
            row = prow + d
            while (pcol, row) in occupied:
                row += d
            grid[child] = (pcol, row)
            occupied.add((pcol, row))
            queue.append((child, d))
    return grid, rail


# --------------------------------------------------------------------------
# Side assignment: which side of its box each wire end leaves from


def _assign_sides(
    diagram: Diagram, grid: dict[str, tuple[int, int]], rail: list[str]
) -> dict[tuple[str, str], str]:
    sides: dict[tuple[str, str], str] = {}
    taken: dict[str, set[str]] = {f: set() for f in grid}

    def claim(factor: str, role: str, side: str) -> None:
        sides[(factor, role)] = side
        taken[factor].add(side)

    def first_free(factor: str, preference: tuple[str, ...]) -> str:
        for side in preference:
            if side not in taken[factor]:
                return side
        return preference[0]

    structural, ports, blobs = [], [], []
    for wire in sorted(diagram.wires.values(), key=lambda w: natural_key(w.id)):
        ends = wire.node_ends()
        if len(ends) == 2:
            structural.append(wire)
        elif any(
            isinstance(e, TerminalEnd) and e.kind is TerminalKind.POINTMASS
            for e in wire.ends
        ):
            blobs.append(wire)
        else:
            ports.append(wire)

    for wire in structural:
        a, b = wire.ends
        (ax, ay), (bx, by) = grid[a.factor], grid[b.factor]
        if abs(bx - ax) >= abs(by - ay):
            claim(a.factor, a.role, "right" if bx >= ax else "left")
            claim(b.factor, b.role, "left" if bx >= ax else "right")
        else:
            claim(a.factor, a.role, "down" if by >= ay else "up")
            claim(b.factor, b.role, "up" if by >= ay else "down")

    for wire in ports:
        (node,) = wire.node_ends()
        term = next(e for e in wire.ends if isinstance(e, TerminalEnd))
        col, row = grid[node.factor]
        if term.kind is TerminalKind.ELLIPSIS and node.factor in rail:
            horizontal = ("left", "right") if node.factor == rail[0] else ("right", "left")
            side = first_free(node.factor, (*horizontal, "up", "down"))
        elif row < 0:
            side = first_free(node.factor, ("up", "left", "right", "down"))
        elif row > 0:
            side = first_free(node.factor, ("down", "left", "right", "up"))
        else:
            side = first_free(node.factor, ("up", "down", "left", "right"))
        claim(node.factor, node.role, side)

    for wire in blobs:
        (node,) = wire.node_ends()
        claim(node.factor, node.role, first_free(node.factor, _TERMINAL_PREFERENCE))
    return sides


# --------------------------------------------------------------------------
# Geometry


def _emit(
    diagram: Diagram,
    presentation: Presentation,
    style: LayoutStyle,
    grid: dict[str, tuple[int, int]],
    sides: dict[tuple[str, str], str],
) -> Scene:
    box, pitch_x, pitch_y = style.sizes(presentation.compact)
    centers = {f: (c * pitch_x, r * pitch_y) for f, (c, r) in grid.items()}

    anchors = _anchor_points(diagram, sides, centers, grid, box, (pitch_x, pitch_y))

    router = WireRouter(grid, (pitch_x, pitch_y), box)
    terminal_plans = terminal_layout(diagram.wires.values(), anchors, router, style, natural_key)
    wires: list[Path] = []
    domains = []
    beads: list[Bead] = []
    terminals = []
    ellipses: list[EllipsisMark] = []

    for wire in diagram.wires.values():
        directive = presentation.wires[wire.id]
        pts, term_item, ellipsis_item = _wire_geometry(
            wire, anchors, router, terminal_plans, style
        )
        if wire.family == "gaussian":
            domains.append((len(wires), wire.dimension, pts))
        label = "" if term_item and term_item.kind is TerminalKind.POINTMASS else wire.label
        label_at = _label_pos(pts, style) if label else None
        if wire.moment_endpoints:
            selected = {end for end, _ in wire.moment_endpoints}
            left, right = _split_at_bead(pts)
            wires.extend((
                Path(wire.id, left, label=label, label_at=label_at, label_size=style.label_size,
                     dashed=wire.ends[0] in selected),
                Path(wire.id, right, dashed=wire.ends[1] in selected),
            ))
        else:
            wires.append(Path(wire.id, pts, label=label, label_at=label_at, label_size=style.label_size))
        if term_item:
            terminals.append((term_item, terminal_plans[wire.id][2]))
        if ellipsis_item:
            ellipses.append(ellipsis_item)
        if directive.bead:
            mx, my = _midpoint(pts)
            beads.append(Bead(wire.id, mx, my, BeadShape.CIRCLE, directive.symbol))

    boxes: list[Box] = []
    internal: list[Path] = []
    domes: list[Dome] = []
    for name, node in diagram.nodes.items():
        cx, cy = centers[name]
        directive = presentation.nodes[name]
        b, i, d = _node_marginals(name, node, directive, anchors, (cx, cy), style)
        beads += b
        internal += i
        domes += d
        glyph_at = _glyph_pos(directive, (cx, cy), box)
        boxes.append(
            Box(f"factor:{name}", cx, cy, box, box, node.glyph, name, glyph_at)
        )

    composites = [
        _composite_box(c, centers, box, style) for c in diagram.composites
    ]

    captions = CaptionSpace(style.label_clearance)
    for node_box in boxes:
        captions.reserve((node_box.x - box / 2, node_box.y - box / 2,
                          node_box.x + box / 2, node_box.y + box / 2))
    for terminal, _ in terminals:
        radius = terminal.ring_radius + 1 if terminal.ring_radius is not None else 6
        captions.reserve((terminal.x - radius, terminal.y - radius,
                          terminal.x + radius, terminal.y + radius))

    for i in sorted(range(len(wires)), key=lambda n: natural_key(wires[n].ref)):
        wire_path = wires[i]
        if wire_path.label_at is not None:
            p, q = max(zip(wire_path.points, wire_path.points[1:]),
                       key=lambda pair: sum(abs(a - b) for a, b in zip(*pair)))
            wires[i] = captions.place(wire_path, 1 if p[1] == q[1] else 0)
    terminals = [captions.place(terminal, 1 - axis) for terminal, axis in terminals]
    place_domains(captions, wires, domains, style, router.lane_clearance)

    # Boxes fill white, so anything drawn inside them must come later.
    items = [*composites, *wires, *boxes, *internal, *domes, *beads, *terminals, *ellipses]
    scene = _framed(diagram.name, items, style)
    # Each bend sits on a half-pitch lane. Its rounded corner fits inside
    # this distance from the bend even when a caller requests a larger arc.
    scene.wire_corner_limit = router.lane_clearance
    return scene


def _glyph_pos(directive, center, box) -> tuple[float, float] | None:
    """Vacate the box center when beads/domes occupy it (Fig 14's top-left
    labels): internal beads -> top-left; a border dome -> nudged up."""
    cx, cy = center
    if not any(len(g.roles) > 1 for g in directive.groups):
        return None  # single-role beads hug the borders; the center is free
    if directive.placement is Placement.INTERNAL:
        return (cx - box * 0.22, cy - box * 0.26)  # joint bead sits center
    return (cx, cy - box * 0.2)  # dome bulges down; lift the glyph


def _anchor_points(diagram, sides, centers, grid, box, pitch):
    """Pixel attachment point for every (factor, role): distribute the wires
    sharing a side along it, ordered by the far endpoint's position."""

    by_side: dict[tuple[str, str], list[tuple[float, str, str]]] = {}
    for wire in diagram.wires.values():
        ends = wire.node_ends()
        for end in ends:
            side = sides[(end.factor, end.role)]
            other = next((e for e in ends if e is not end), None)
            if other is not None:
                ox, oy = centers[other.factor]
                sort = oy if side in ("left", "right") else ox
            else:
                sort = 0.0
            by_side.setdefault((end.factor, side), []).append((sort, wire.id, end.role))

    anchors: dict[tuple[str, str], tuple[float, float, str]] = {}
    for (factor, side), entries in by_side.items():
        entries.sort(key=lambda e: (e[0], natural_key(e[1])))
        cx, cy = centers[factor]
        for i, (_, _, role) in enumerate(entries):
            frac = (i + 1) / (len(entries) + 1)
            if side == "left":
                pt = (cx - box / 2, cy - box / 2 + frac * box)
            elif side == "right":
                pt = (cx + box / 2, cy - box / 2 + frac * box)
            elif side == "up":
                pt = (cx - box / 2 + frac * box, cy - box / 2)
            else:
                pt = (cx - box / 2 + frac * box, cy + box / 2)
            anchors[(factor, role)] = (*pt, side)
    return anchors


def _wire_geometry(wire, anchors, router, terminal_plans, style):
    pts: list[tuple[float, float]] = []
    terminal = None
    ellipsis = None

    nodes = wire.node_ends()
    if len(nodes) == 2:
        a, b = nodes
        pts = router.connect(
            anchors[(a.factor, a.role)], anchors[(b.factor, b.role)], a.factor, b.factor
        )
    else:
        (node,) = nodes
        term = next(e for e in wire.ends if isinstance(e, TerminalEnd))
        x, y, side = anchors[(node.factor, node.role)]
        pts, baseline, _ = terminal_plans[wire.id]
        tx, ty = pts[-1]
        ref = term.ref or wire.id
        if term.kind is TerminalKind.ELLIPSIS:
            angle = 0.0 if side in ("left", "right") else 90.0
            ellipsis = EllipsisMark(ref, tx, ty, angle)
        else:
            terminal = Terminal(ref, tx, ty, term.kind, term.label, baseline, style.caption_size,
                                dimension=wire.dimension, radius=style.continuous_terminal_radius)
    return tuple(pts), terminal, ellipsis


def _midpoint(pts) -> tuple[float, float]:
    """Midpoint of the longest segment (where a bead reads as 'on the wire')."""
    best, best_len = pts[:2], -1.0
    for p, q in zip(pts, pts[1:]):
        seg = abs(p[0] - q[0]) + abs(p[1] - q[1])
        if seg > best_len:
            best, best_len = (p, q), seg
    (x1, y1), (x2, y2) = best
    return (x1 + x2) / 2, (y1 + y2) / 2


def _split_at_bead(pts):
    """Split a wire at its edge bead, preserving bends on both half-edges."""
    index = max(
        range(len(pts) - 1),
        key=lambda i: abs(pts[i][0] - pts[i + 1][0]) + abs(pts[i][1] - pts[i + 1][1]),
    )
    midpoint = _midpoint(pts)
    return (*pts[:index + 1], midpoint), (midpoint, *pts[index + 1:])


def _label_pos(pts, style) -> tuple[float, float]:
    """Above a horizontal run, right of a vertical one (the papers' habit)."""
    best, best_len = pts[:2], -1.0
    for p, q in zip(pts, pts[1:]):
        seg = abs(p[0] - q[0]) + abs(p[1] - q[1])
        if seg > best_len:
            best, best_len = (p, q), seg
    (x1, y1), (x2, y2) = best
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if abs(x2 - x1) >= abs(y2 - y1):
        return mx + 2.0, my - 7.0
    return mx + 8.0, my + 4.0


def _node_marginals(name, node, directive, anchors, center, style):
    """Bead/dome/internal-edge geometry for one node's q-marginal groups."""
    beads: list[Bead] = []
    internal: list[Path] = []
    domes: list[Dome] = []
    cx, cy = center
    ref = f"factor:{name}"
    moment_points = {
        anchors[(name, role)][:2] for role in node.moment_roles
        if (name, role) in anchors
    }

    for group in directive.groups:
        pts = [anchors.get((name, r)) for r in group.roles]
        pts = [p for p in pts if p is not None]
        if not pts:
            continue
        if directive.placement is Placement.BORDER:
            if len(pts) == 1 or group.shape is BeadShape.SQUARE:
                for x, y, _ in pts:
                    beads.append(Bead(ref, x, y, group.shape, group.symbol))
            else:
                domes.append(Dome(ref, tuple((x, y) for x, y, _ in pts), cx, cy))
        else:
            if len(pts) == 1:
                x, y, side = pts[0]
                dx, dy = DIRECTIONS[side]
                bx, by = x - dx * style.bead_r * 1.9, y - dy * style.bead_r * 1.9
                beads.append(Bead(ref, bx, by, group.shape, group.symbol))
                if directive.internal_edges:
                    internal.append(Path(
                        ref, ((x, y), (bx, by)), kind="internal",
                        dashed=(x, y) in moment_points,
                    ))
            else:
                beads.append(Bead(ref, cx, cy, group.shape, group.symbol))
                if directive.internal_edges:
                    for x, y, _ in pts:
                        internal.append(Path(
                            ref, ((x, y), (cx, cy)), kind="internal",
                            dashed=(x, y) in moment_points,
                        ))
    return beads, internal, domes


def _composite_box(comp, centers, box, style) -> CompositeBox:
    xs, ys = zip(*(centers[m] for m in comp.members))
    pad = box / 2 + style.composite_pad
    return CompositeBox(
        f"gfe:{comp.name}",
        min(xs) - pad,
        min(ys) - pad,
        (max(xs) - min(xs)) + 2 * pad,
        (max(ys) - min(ys)) + 2 * pad,
    )


def _framed(name: str, items: list, style: LayoutStyle) -> Scene:
    """Translate everything into positive coordinates with a margin."""
    xs: list[float] = []
    ys: list[float] = []
    for item in items:
        for x, y in _item_points(item):
            xs.append(x)
            ys.append(y)
    dx = style.margin - min(xs, default=0.0)
    dy = style.margin - min(ys, default=0.0)
    moved = [_shift(item, dx, dy) for item in items]
    width = (max(xs, default=0.0) - min(xs, default=0.0)) + 2 * style.margin
    height = (max(ys, default=0.0) - min(ys, default=0.0)) + 2 * style.margin
    return Scene(name=name, width=width, height=height, items=moved)


def _item_points(item):
    if isinstance(item, Box):
        return [(item.x - item.w / 2 - 20, item.y - item.h / 2 - 20),
                (item.x + item.w / 2 + 20, item.y + item.h / 2 + 20)]
    if isinstance(item, Path):
        pts = list(item.points)
        for caption in (Label.for_item(item), item.domain_label):
            if caption:
                x, y, u, v = caption.bounds
                pts.extend(((x, y), (u, v)))
        return pts
    if isinstance(item, (Bead,)):
        return [(item.x - 10, item.y - 10), (item.x + 10, item.y + 10)]
    if isinstance(item, (Terminal, EllipsisMark)):
        points = [(item.x - 26, item.y - 16), (item.x + 26, item.y + 16)]
        if isinstance(item, Terminal):
            if (radius := item.ring_radius) is not None:
                points.extend(((item.x - radius - 1, item.y - radius - 1),
                               (item.x + radius + 1, item.y + radius + 1)))
            if caption := Label.for_item(item):
                x, y, u, v = caption.bounds
                points.extend(((x, y), (u, v)))
        return points
    if isinstance(item, Dome):
        return list(item.anchors)
    if isinstance(item, CompositeBox):
        return [(item.x, item.y), (item.x + item.w, item.y + item.h)]
    return []


def _shift(item, dx: float, dy: float):
    if isinstance(item, Path):
        pts = tuple((x + dx, y + dy) for x, y in item.points)
        label_at = (
            (item.label_at[0] + dx, item.label_at[1] + dy) if item.label_at else None
        )
        domain = item.domain_label
        if domain:
            domain = replace(domain, x=domain.x + dx, y=domain.y + dy)
        return replace(item, points=pts, label_at=label_at, domain_label=domain)
    if isinstance(item, Terminal) and item.label_at is not None:
        x, y = item.label_at
        return replace(item, x=item.x + dx, y=item.y + dy, label_at=(x + dx, y + dy))
    if isinstance(item, Dome):
        return replace(
            item,
            anchors=tuple((x + dx, y + dy) for x, y in item.anchors),
            cx=item.cx + dx,
            cy=item.cy + dy,
        )
    if isinstance(item, Box) and item.glyph_at is not None:
        return replace(
            item,
            x=item.x + dx,
            y=item.y + dy,
            glyph_at=(item.glyph_at[0] + dx, item.glyph_at[1] + dy),
        )
    return replace(item, x=item.x + dx, y=item.y + dy)
