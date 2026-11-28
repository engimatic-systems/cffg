"""Place schedule arrowheads on the straight portions of the diagram's wires."""

import itertools
from collections import defaultdict
from dataclasses import replace
from math import copysign

from cffg.viz.diagram import NodeEnd, TerminalEnd
from cffg.viz.scene import Bead, Box, EllipsisMark, MessageArrow, Path, Terminal


def _extent(item, style):
    if isinstance(item, Box):
        return (
            item.x - item.w / 2,
            item.y - item.h / 2,
            item.x + item.w / 2,
            item.y + item.h / 2,
        )
    if isinstance(item, (Bead, Terminal, EllipsisMark)):
        r = (
            13
            if isinstance(item, EllipsisMark)
            else max(style.bead_r, style.terminal_size / 2)
        )
        return item.x - r, item.y - r, item.x + r, item.y + r
    if isinstance(item, Path) and item.label_at:
        x, y = item.label_at
        # Wire labels begin at their SVG text anchor and sit on its baseline.
        return (
            x,
            y - style.label_size,
            x + style.label_size * len(item.label),
            y + style.label_size / 3,
        )
    return None


def _straight_parts(paths, radius):
    """Trim the pieces of each segment which may be replaced by a quadratic bend."""
    parts = []
    for path in paths:
        pts = path.points
        lengths = [
            abs(x - u) + abs(y - v) for (x, y), (u, v) in itertools.pairwise(pts)
        ]
        trims = [
            0,
            *(min(radius, a / 2, b / 2) for a, b in itertools.pairwise(lengths)),
            0,
        ]
        for i, ((x, y), (u, v)) in enumerate(itertools.pairwise(pts)):
            if x == u:
                parts.append((1, x, y, v, trims[i], trims[i + 1]))
            else:
                parts.append((0, y, x, u, trims[i], trims[i + 1]))
    return parts


def _place(parts, forward, obstacles, clearance):
    """Choose the closest free point to three quarters of the directed arclength."""
    total = sum(abs(end - start) for _, _, start, end, _, _ in parts)
    preferred = total * (0.75 if forward else 0.25)
    for size in (6.0, 4.0, 2.0, 1.0):
        if size / 2 > clearance:
            continue
        candidates = []
        travelled = 0.0
        for axis, fixed, start, end, trim_start, trim_end in parts:
            sign = int(copysign(1, end - start))
            direction = sign if forward else -sign
            first, last = sorted((start + sign * trim_start, end - sign * trim_end))
            # The triangle extends one size behind its tip.
            lo, hi = (first + size, last) if direction > 0 else (first, last - size)
            target = start + sign * (preferred - travelled)
            travelled += abs(end - start)
            if hi < lo:
                continue
            intervals = [(lo, hi)]
            for rect in obstacles:
                if not rect[1 - axis] - size / 2 < fixed < rect[3 - axis] + size / 2:
                    continue
                left, right = rect[axis], rect[axis + 2]
                if direction > 0:
                    right += size
                else:
                    left -= size
                available = []
                for low, high in intervals:
                    if low <= left:
                        available.append((low, min(high, left)))
                    if right <= high:
                        available.append((max(low, right), high))
                intervals = available
            for low, high in intervals:
                pos = max(low, min(target, high))
                tip = (pos, fixed) if axis == 0 else (fixed, pos)
                vector = (direction, 0) if axis == 0 else (0, direction)
                candidates.append((abs(pos - target), tip, vector))
        if candidates:
            _, tip, vector = min(candidates)
            return tip, vector, size
    raise ValueError(
        "no straight wire segment has room for the message arrow; increase layout pitch"
    )


def add_arrows(scene, diagram, overlay, style):
    by_ref = defaultdict(list)
    for item in scene.items:
        by_ref[item.ref].append(item)
    events = defaultdict(list)
    for event in overlay.events:
        events[
            event.wire_ref,
            event.source_ref,
            event.source_role,
            event.target_ref,
            event.gfe,
        ].append(event.step)

    arrows = []
    occupied = defaultdict(list)
    for (ref, source, role, target, gfe), steps in events.items():
        wire = diagram.wires[ref]
        paths = [p for p in by_ref[ref] if isinstance(p, Path)]
        refs = {ref, source, target}
        for end in wire.ends:
            if isinstance(end, TerminalEnd):
                refs.add(end.ref)
        obstacles = [
            rect for r in refs for item in by_ref[r] if (rect := _extent(item, style))
        ]
        first = wire.ends[0]
        forward = (
            isinstance(first, NodeEnd)
            and first.factor == source.removeprefix("factor:")
            and first.role == role
        )
        clearance = scene.wire_corner_limit
        tip, direction, size = _place(
            _straight_parts(paths, clearance),
            forward,
            obstacles + occupied[ref],
            clearance,
        )
        x, y = tip
        dx, dy = direction
        base_x, base_y = x - size * dx, y - size * dy
        # Reserve a visible gap even when opposite tips would only touch.
        gap = max(1.0, style.stroke_width)
        occupied[ref].append(
            (
                min(x, base_x) - abs(dy) * size / 2 - gap,
                min(y, base_y) - abs(dx) * size / 2 - gap,
                max(x, base_x) + abs(dy) * size / 2 + gap,
                max(y, base_y) + abs(dx) * size / 2 + gap,
            )
        )
        arrows.append(
            MessageArrow(
                ref, source, target, role, tuple(steps), gfe, tip, direction, size
            )
        )
    return replace(scene, items=[*scene.items, *arrows])
