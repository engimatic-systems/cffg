"""Place unrolled recurrences with time pinned horizontally and springs between neighbors."""

from collections import defaultdict
import re

from .diagram import TerminalEnd, TerminalKind


def indexed_name(name):
    match = re.fullmatch(r"([^0-9]+)([0-9]+)[^0-9]*", name)
    return (match[1], int(match[2])) if match else None


def temporal_grid(diagram, adjacency):
    # Composed paper slices have their own directed glue rail and composite layout.
    if diagram.composites:
        return None
    bindings = defaultdict(dict)
    observed = set()
    for wire in diagram.wires.values():
        if not wire.id.startswith("var:"):
            continue
        for terminal in wire.ends:
            if isinstance(terminal, TerminalEnd):
                if terminal.kind == TerminalKind.ELLIPSIS:
                    return None
                if terminal.kind == TerminalKind.DATA:
                    observed.add(wire.id)
        for end in wire.node_ends():
            bindings[end.factor][end.role] = wire.id

    stages = {}
    outputs = {}
    recurrence = False
    for name, node in diagram.nodes.items():
        roles = bindings[name]
        out = roles.get("out") or roles.get("a")
        stamp = out and indexed_name(out)
        incoming = indexed_name(roles.get("in", ""))
        if (
            incoming
            and stamp
            and incoming[0] == stamp[0]
            and stamp[1] == incoming[1] + 1
        ):
            recurrence = True
        stamp = stamp or indexed_name(name)
        if not stamp:
            return None
        stages[name] = stamp[1]
        outputs[name] = out
    if not recurrence or len(set(stages.values())) < 2:
        return None

    # A fan-out tree follows its source and precedes same-slice observation factors.
    producer = {}
    for name, node in diagram.nodes.items():
        roles = bindings[name]
        for role, wire in roles.items():
            if role == "out" or (node.type == "equality" and role != "a"):
                producer[wire] = name
    phase = {}

    def fanout_depth(name, visiting):
        if name in phase:
            return phase[name]
        if name in visiting or diagram.nodes[name].type != "equality":
            return 0
        parent = producer.get(bindings[name].get("a"))
        depth = 1 + (
            fanout_depth(parent, visiting | {name})
            if parent and stages[parent] == stages[name]
            else 0
        )
        phase[name] = depth
        return depth

    for name in diagram.nodes:
        phase[name] = fanout_depth(name, set())
    last = max(phase.values()) + 1
    preferred = {}
    for name, node in diagram.nodes.items():
        origin = name
        while diagram.nodes[origin].type == "equality":
            parent = producer.get(bindings[origin].get("a"))
            if not parent or phase[parent] >= phase[origin]:
                break
            origin = parent
        if outputs[name] in observed:
            phase[name] = last
            preferred[name] = 2.0
        elif diagram.nodes[origin].type in (
            "categorical",
            "matrix_cat",
            "recurrence_link",
        ):
            preferred[name] = -1.0
        else:
            preferred[name] = 0.0
    ranks = {s: i for i, s in enumerate(sorted(set(stages.values())))}
    columns = defaultdict(list)
    for name in sorted(stages):
        columns[ranks[stages[name]] * (last + 1) + phase[name]].append(name)

    # The time coordinate is fixed. Springs pull related factors toward the same
    # lane, while neighboring boxes in each column repel one another vertically.
    y = dict(preferred)
    for _ in range(18):
        forces = {}
        for names in columns.values():
            ordered = sorted(names, key=lambda n: (y[n], n))
            for n in names:
                neighbors = adjacency[n]
                pull = sum(y[peer] - y[n] for peer, _ in neighbors) / max(
                    1, len(neighbors)
                )
                forces[n] = 0.22 * pull + 0.48 * (preferred[n] - y[n])
            for a, b in zip(ordered, ordered[1:]):
                repel = max(0.0, 1.4 - (y[b] - y[a])) * 0.45
                forces[a] -= repel
                forces[b] += repel
        y = {n: value + forces[n] for n, value in y.items()}

    grid = {}
    for column, names in columns.items():
        previous = None
        for n in sorted(names, key=lambda name: (y[name], name)):
            row = round(y[n])
            if previous is not None:
                row = max(row, previous + 1)
            grid[n] = (column, row)
            previous = row
    return grid
