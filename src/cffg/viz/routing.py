"""Orthogonal routing along clear corridors between grid-aligned factor boxes."""

from math import ceil, floor, isclose


def simplify(points):
    result = []
    for point in points:
        if result and point == result[-1]:
            continue
        while len(result) >= 2:
            a, b = result[-2:]
            if not (a[0] == b[0] == point[0] or a[1] == b[1] == point[1]):
                break
            result.pop()
        if result and result[-1] == point:
            continue
        result.append(point)
    return result


class WireRouter:
    def __init__(self, grid, pitch, box):
        self.grid = grid
        self.pitch = pitch
        self.occupied = {cell: name for name, cell in grid.items()}
        self.radius = box / 2 + min(10.0, (min(pitch) - box) / 4)

    def clear(self, points, ends):
        px, py = self.pitch
        r = self.radius
        for (x, y), (u, v) in zip(points, points[1:]):
            for col in range(
                ceil((min(x, u) - r) / px), floor((max(x, u) + r) / px) + 1
            ):
                for row in range(
                    ceil((min(y, v) - r) / py), floor((max(y, v) + r) / py) + 1
                ):
                    obstacle = self.occupied.get((col, row))
                    if obstacle is not None and obstacle not in ends:
                        return False
        return True

    def gates(self, anchor, factor):
        x, y, side = anchor
        col, row = self.grid[factor]
        px, py = self.pitch
        if side in ("left", "right"):
            lane = (col + (-0.5 if side == "left" else 0.5)) * px
            return (lane, y), ((lane, (row - 0.5) * py), (lane, (row + 0.5) * py))
        lane = (row + (-0.5 if side == "up" else 0.5)) * py
        return (x, lane), (((col - 0.5) * px, lane), ((col + 0.5) * px, lane))

    def connect(self, first, second, a, b):
        start, stop = first[:2], second[:2]
        gate_a, corners_a = self.gates(first, a)
        gate_b, corners_b = self.gates(second, b)

        def score(path):
            distance = sum(
                abs(x - u) + abs(y - v) for (x, y), (u, v) in zip(path, path[1:])
            )
            return distance + 8 * len(path), len(path), path

        # Enforce the outward normal at both anchors, even on short direct paths.
        def leaves(point, nxt, side):
            dx, dy = nxt[0] - point[0], nxt[1] - point[1]
            return {
                "left": dx < 0 and isclose(dy, 0),
                "right": dx > 0 and isclose(dy, 0),
                "up": dy < 0 and isclose(dx, 0),
                "down": dy > 0 and isclose(dx, 0),
            }[side]

        candidates = []
        for turn in ((start[0], stop[1]), (stop[0], start[1])):
            path = simplify([start, turn, stop])
            if (
                len(path) > 1
                and leaves(start, path[1], first[2])
                and leaves(stop, path[-2], second[2])
                and self.clear(path, {a, b})
            ):
                candidates.append(path)
        if candidates:
            return min(candidates, key=score)
        for turn in ((gate_a[0], gate_b[1]), (gate_b[0], gate_a[1])):
            middle = simplify([gate_a, turn, gate_b])
            if self.clear(middle, set()):
                path = simplify([start, *middle, stop])
                if (
                    len(path) > 1
                    and leaves(start, path[1], first[2])
                    and leaves(stop, path[-2], second[2])
                ):
                    candidates.append(path)
        if candidates:
            return min(candidates, key=score)
        for ca in corners_a:
            for cb in corners_b:
                for corner in ((ca[0], cb[1]), (cb[0], ca[1])):
                    path = simplify([start, gate_a, ca, corner, cb, gate_b, stop])
                    candidates.append(path)
        return min(candidates, key=score)

    def terminal(self, factor, anchor, desired, direction):
        col, row = self.grid[factor]
        dx, dy = direction
        neighbor = (col + dx, row + dy)
        if neighbor not in self.occupied:
            return desired
        axis = 0 if dx else 1
        available = (
            abs(neighbor[axis] * self.pitch[axis] - anchor[axis]) - self.radius - 2.0
        )
        return min(desired, max(0.0, available))
