"""Deterministic placement of wire and terminal text in the routed scene."""

from collections import defaultdict
from dataclasses import replace
from math import floor

from .diagram import TerminalEnd, TerminalKind
from .routing import DIRECTIONS, simplify
from .scene import Label, label_width


class CaptionSpace:
    """Index occupied boxes and place each caption on the nearest free parallel line."""

    def __init__(self, clearance):
        self.clearance = clearance
        self.cells = defaultdict(list)

    @staticmethod
    def _cells(box):
        x, y, u, v = box
        for i in range(floor(x / 96), floor(u / 96) + 1):
            for j in range(floor(y / 96), floor(v / 96) + 1):
                yield i, j

    def reserve(self, bounds):
        for cell in self._cells(bounds):
            self.cells[cell].append(bounds)

    def overlaps(self, bounds):
        x, y, u, v = bounds
        return {
            box
            for cell in self._cells(bounds)
            for box in self.cells.get(cell, ())
            if x < box[2] and box[0] < u and y < box[3] and box[1] < v
        }

    def _free(self, label, axis, sign):
        distance = 0.0
        while True:
            pos = replace(
                label,
                x=label.x + (distance if axis == 0 else 0),
                y=label.y + (distance if axis == 1 else 0),
            )
            bounds = pos.bounds
            overlaps = self.overlaps(bounds)
            if not overlaps:
                return distance, pos
            if sign > 0:
                distance += (
                    max(box[axis + 2] for box in overlaps)
                    - bounds[axis]
                    + self.clearance
                )
            else:
                distance += (
                    min(box[axis] for box in overlaps)
                    - bounds[axis + 2]
                    - self.clearance
                )

    def place(self, item, axis):
        label = Label.for_item(item)
        if label is None:
            return item
        label = self.place_label(label, axis)
        return replace(item, label_at=(label.x, label.y))

    def place_label(self, label, axis):
        if self.overlaps(label.bounds):
            options = [self._free(label, axis, sign) for sign in (-1, 1)]
            _, label = min(options, key=lambda option: abs(option[0]))
        self.reserve(label.bounds)
        return label


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def place_domains(captions, wires, domains, style, corner_limit):
    """Place domain captions after ordinary names, with wire strokes reserved."""
    if not domains:
        return
    # Reserve straight strokes and every rounded corner's control rectangle.
    stroke = max(1.0, style.label_clearance / 2)
    for wire in wires:
        for (x, y), (u, v) in zip(wire.points, wire.points[1:]):
            captions.reserve((min(x, u) - stroke, min(y, v) - stroke,
                              max(x, u) + stroke, max(y, v) + stroke))
        for p, (x, y), q in zip(wire.points, wire.points[1:], wire.points[2:]):
            radius = min(corner_limit, sum(abs(a - b) for a, b in zip(p, (x, y))) / 2,
                         sum(abs(a - b) for a, b in zip(q, (x, y))) / 2) + stroke
            captions.reserve((x - radius, y - radius, x + radius, y + radius))
    for index, dimension, points in domains:
        p, q = max(zip(points, points[1:]), key=lambda pair: sum(abs(a - b) for a, b in zip(*pair)))
        x, y = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        text = "ℝ" + str(dimension).translate(_SUPERSCRIPT)
        size = style.caption_size
        label = Label(wires[index].ref, text, x, y, "middle", "wire", size)
        if p[1] == q[1]:
            ascent = y - label.bounds[1]
            label = replace(label, y=y + ascent + style.label_clearance)
            axis = 1
        else:
            label = replace(label, x=x - label.width / 2 - style.label_clearance, y=y + 0.3 * size)
            axis = 0
        wires[index] = replace(wires[index], domain_label=captions.place_label(label, axis))


def terminal_layout(wires, anchors, router, style, natural_key):
    """Assign terminal wire endpoints and caption baselines, grouped by factor side."""
    groups = defaultdict(list)
    for wire in wires:
        nodes = wire.node_ends()
        if len(nodes) != 1:
            continue
        node = nodes[0]
        term = next(end for end in wire.ends if isinstance(end, TerminalEnd))
        x, y, side = anchors[node.factor, node.role]
        across = y if side in ("left", "right") else x
        groups[node.factor, side].append(
            (
                term.kind is TerminalKind.POINTMASS,
                across,
                natural_key(wire.id),
                wire.id,
                x,
                y,
                term,
            )
        )

    plans = {}
    for (factor, side), group in groups.items():
        dx, dy = DIRECTIONS[side]
        axis = 0 if dx else 1
        previous = 0.0
        for _, _, _, wire_id, x, y, term in sorted(group):
            desired = router.pitch[axis] * style.terminal_offset
            caption = term.label if term.kind is not TerminalKind.ELLIPSIS else ""
            width = label_width(caption, style.caption_size)
            span = width + 2 * style.label_clearance
            length = router.terminal(factor, (x, y), desired, (dx, dy))
            start = (x, y)
            if caption:
                length = max(length, previous + span)
            stop = (x + dx * length, y + dy * length)
            points = (start, stop)
            if caption and router.terminal(factor, start, length, (dx, dy)) < length:
                # A neighboring box blocks a straight extension. The half-cell
                # corridor runs parallel to the requested outward direction.
                gate, corners = router.gates((x, y, side), factor)
                corner = min(corners, key=lambda p: (abs(p[0] - x) + abs(p[1] - y), p))
                stop = (stop[0], corner[1]) if dx else (corner[0], stop[1])
                points = simplify([start, gate, corner, stop])
            baseline = None
            if caption:
                midpoint = (previous + length) / 2 if previous else length - span / 2
                if dx:
                    baseline = (x + dx * midpoint, stop[1] - style.label_clearance)
                else:
                    baseline = (
                        stop[0] + width / 2 + style.label_clearance,
                        y + dy * midpoint + style.caption_size * 0.3,
                    )
            plans[wire_id] = (tuple(points), baseline, axis)
            previous = length
    return plans
