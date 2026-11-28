"""Grid layout: rail extraction, branch sides, terminal placement, stability.

The claims mirror the papers' layout conventions (Figs 22/26): time reads
left to right along the state chain, control priors sit above it, the
observation/goal stack hangs below, and — the editing property — growing a
model only adds rows/columns, never moves what was already there.
"""

import numpy as np
import pytest

from cffg.models.rollout import build_rollout
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec, validate
from cffg.core.compose import merge, scoped
from cffg.viz import derive
from cffg.viz.compress import present
from cffg.viz.diagram import TerminalKind
from cffg.viz.layout import LayoutStyle, layout
from cffg.viz.layout import _item_points
from cffg.viz.scene import Bead, Box, CompositeBox, EllipsisMark, Terminal


def rollout_template(steps=2, **kw):
    K, M, U = 4, 3, 2
    template, _ = build_rollout(
        "roll", steps=steps, past=kw.pop("past", 0),
        d=np.full(K, 1 / K), A=np.full((M, K), 1 / M), c=np.full(M, 1 / M),
        e=np.full(U, 1 / U), Bs=np.stack([np.eye(K)] * U), **kw,
    )
    return template


def scene_of(template, level=4):
    d = derive(template)
    return layout(d, present(d, level))


def terminals(scene, kind):
    return [i for i in scene.items if isinstance(i, Terminal) and i.kind is kind]


class TestHmmSlice:
    def test_rail_reads_left_to_right(self, hmm_slice):
        boxes = scene_of(hmm_slice).boxes()
        trans, eq, obs = (
            boxes[f"factor:{n}"] for n in ("trans", "eq", "obs")
        )
        assert trans.x < eq.x and trans.y == eq.y  # head-port factor left
        assert obs.x == eq.x and obs.y > eq.y  # observation hangs below

    def test_glue_ellipses_continue_the_rail(self, hmm_slice):
        scene = scene_of(hmm_slice)
        boxes = scene.boxes()
        marks = {m.ref: m for m in scene.items if isinstance(m, EllipsisMark)}
        assert marks["port:head"].x < boxes["factor:trans"].x
        assert marks["port:tail"].x > boxes["factor:eq"].x

    def test_terminal_sides_match_paper_convention(self, hmm_slice):
        scene = scene_of(hmm_slice)
        boxes = scene.boxes()
        pm = {t.ref: t for t in terminals(scene, TerminalKind.POINTMASS)}
        assert pm["blob:B"].y < boxes["factor:trans"].y  # left taken by glue -> up
        assert pm["blob:A"].x < boxes["factor:obs"].x  # obs branch: A to the left
        (sense,) = terminals(scene, TerminalKind.DATA)
        assert sense.y > boxes["factor:obs"].y  # clamp continues downward


class TestRollout:
    def test_rail_and_branch_rows(self):
        boxes = scene_of(rollout_template()).boxes()
        rail = ["prior", "tm1", "eq1", "tm2", "eq2"]
        xs = [boxes[f"factor:{n}"].x for n in rail]
        assert xs == sorted(xs) and len(set(xs)) == len(xs)
        assert len({boxes[f"factor:{n}"].y for n in rail}) == 1

        y0 = boxes["factor:eq1"].y
        assert boxes["factor:e1"].y < y0  # control prior above (selector role)
        assert y0 < boxes["factor:obs1"].y < boxes["factor:goal1"].y
        assert boxes["factor:obs1"].x == boxes["factor:eq1"].x

    def test_composite_box_wraps_obs_and_goal(self):
        scene = scene_of(rollout_template())
        boxes = scene.boxes()
        comp = next(
            i
            for i in scene.items
            if isinstance(i, CompositeBox) and i.ref == "gfe:obs1"
        )
        for member in ("obs1", "goal1"):
            b = boxes[f"factor:{member}"]
            assert comp.x < b.x - b.w / 2 and comp.x + comp.w > b.x + b.w / 2
            assert comp.y < b.y - b.h / 2 and comp.y + comp.h > b.y + b.h / 2

    def test_horizon_ellipsis_at_right_end(self):
        scene = scene_of(rollout_template())
        (mark,) = [i for i in scene.items if isinstance(i, EllipsisMark)]
        assert mark.x > max(b.x for b in scene.boxes().values()) - 1

    def test_delta_bead_sits_on_control_wire(self, viz_gallery):
        template = rollout_template(map_policy=True)
        viz_gallery(template, tag="MAP policy: Fig 29's ∂ beads")
        scene = scene_of(template)
        deltas = [
            i for i in scene.items if isinstance(i, Bead) and i.symbol == "delta"
        ]
        assert {b.ref for b in deltas} == {"var:u1", "var:u2"}
        boxes = scene.boxes()
        for b in deltas:
            k = b.ref[-1]
            e, tm = boxes[f"factor:e{k}"], boxes[f"factor:tm{k}"]
            assert e.y < b.y < tm.y and abs(b.x - tm.x) < tm.w


class TestStability:
    def test_growing_the_rollout_moves_nothing(self, viz_gallery):
        viz_gallery(rollout_template(2), tag="2 steps")
        viz_gallery(rollout_template(3), tag="3 steps: nothing moved")
        small, big = scene_of(rollout_template(2)), scene_of(rollout_template(3))
        sb, bb = small.boxes(), big.boxes()
        assert set(sb) < set(bb)
        for ref, box in sb.items():
            assert (box.x, box.y) == (bb[ref].x, bb[ref].y), ref
        assert big.width > small.width

    def test_layout_is_deterministic(self, hmm_slice):
        a, b = scene_of(hmm_slice), scene_of(hmm_slice)
        assert a == b

    def test_uncompressed_uses_larger_boxes(self, hmm_slice):
        style = LayoutStyle()
        full = scene_of(hmm_slice, 0).boxes()["factor:eq"]
        compact = scene_of(hmm_slice, 4).boxes()["factor:eq"]
        assert full.w == style.box and compact.w == style.compact_box


def gaussian_components(count):
    template = Template("independent_gaussians")
    template.blobs = {name: BlobRef(name, name + ".npy") for name in ("mean", "cov")}
    for index in range(count):
        name = f"x{index}"
        template.variables[name] = VariableSpec(name, dim=1)
        template.factors[f"prior{index}"] = FactorSpec(
            f"prior{index}", "gaussian", edges={"out": name},
            param_blobs={"mean": "mean", "cov": "cov"},
        )
        template.ports[name] = PortSpec(name, name, PortRole.DATA)
    validate(template)
    return template


@pytest.mark.parametrize("count", [1, 2, 30])
@pytest.mark.parametrize("level", [0, 4])
def test_disconnected_gaussians_place_every_factor_once(count, level):
    template = gaussian_components(count)
    scene = scene_of(template, level)
    boxes = list(scene.boxes().values())
    assert len(boxes) == count
    assert sum(isinstance(item, Box) for item in scene.items) == count
    assert {box.name for box in boxes} == set(template.factors)
    assert [box.name for box in boxes] == [f"prior{i}" for i in range(count)]
    assert all(a.y + a.h / 2 < b.y - b.h / 2 for a, b in zip(boxes, boxes[1:]))
    assert scene == scene_of(template, level)


@pytest.mark.parametrize("level", [0, 4])
def test_mixed_size_components_pack_full_extents_with_composites(level):
    parts = [scoped(rollout_template(steps), f"part{index}")
             for index, steps in enumerate((1, 3, 2))]
    template = Template("disconnected_rollouts")
    for part in reversed(parts):
        part.agent = None
        template = merge(template, part, template.name)
    scene = scene_of(template, level)
    assert set(scene.boxes()) == {"factor:" + name for name in template.factors}
    last_bottom = 0.
    for index, part in enumerate(parts):
        items = [item for item in scene.items if f":part{index}/" in item.ref]
        points = [point for item in items for point in _item_points(item)]
        assert min(y for _, y in points) > last_bottom
        last_bottom = max(y for _, y in points)
        assert last_bottom < scene.height
        assert all(0 < x < scene.width for x, _ in points)
        local = scene_of(part, level).boxes()
        offsets = {(box.x - local[ref].x, box.y - local[ref].y)
                   for ref, box in scene.boxes().items() if ref in local}
        assert len(offsets) == 1
        assert len(items) == len(scene_of(part, level).items)


def test_empty_direct_layout_has_a_defined_frame():
    scene = scene_of(Template("empty"))
    assert scene.items == []
    assert scene.width == scene.height == 2 * LayoutStyle().margin


@pytest.mark.parametrize("steps, level", [(3, 0), (6, 4)])
def test_rslds_time_slices_and_routed_wire_clearance(steps, level):
    import re

    from cffg.studio.models import starter
    from cffg.viz import build_scene
    from cffg.viz.scene import Path

    template = starter("rslds", "Switching", steps=steps).template
    for offsets in (None, {"z0_prior": [0, -1]}):
        scene = build_scene(template, level=level, offsets=offsets)
        boxes = scene.boxes()
        by_time = [[] for _ in range(steps)]
        for box in boxes.values():
            by_time[int(re.search(r"\d+", box.name)[0])].append(box.x)
        assert all(max(a) < min(b) for a, b in zip(by_time, by_time[1:]))
        wires = derive(template).wires
        for segment in scene.items:
            if not isinstance(segment, Path) or segment.kind != "wire":
                continue
            connected = {end.factor for end in wires[segment.ref].node_ends()}
            for (x, y), (u, v) in zip(segment.points, segment.points[1:]):
                assert x == u or y == v
                for name, box in boxes.items():
                    if box.name in connected:
                        continue
                    assert (
                        max(x, u) < box.x - box.w / 2 - 3
                        or min(x, u) > box.x + box.w / 2 + 3
                        or max(y, v) < box.y - box.h / 2 - 3
                        or min(y, v) > box.y + box.h / 2 + 3
                    ), (segment.ref, name)


def gaussian_junction(kind):
    """Registered Gaussian factor with a neighbor on each variable and three fixed parameters."""
    t = Template(kind)
    roles = {"in": "x", "out": "y"}
    if kind == "gaussian_mixture_linear":
        roles["selector"] = "z"
    for role, var in roles.items():
        name = var + "prior"
        if role == "selector":
            t.variables[var] = VariableSpec(var, cardinality=2)
            prior = FactorSpec(name, "categorical", edges={"out": var}, param_blobs={"statistic": "p"})
        else:
            t.variables[var] = VariableSpec(var, dim=1)
            prior = FactorSpec(name, "gaussian", edges={"out": var}, param_blobs={"mean": "m", "cov": "V"})
        t.factors[name] = prior
    t.factors["junction"] = FactorSpec("junction", kind, edges=roles,
                                       param_blobs={"matrix": "A", "offset": "b", "cov": "Q"})
    names = {name for factor in t.factors.values() for name in factor.param_blobs.values()}
    t.blobs = {name: BlobRef(name, name + ".npy") for name in sorted(names)}
    validate(t)
    return t


def _wire_paths(template, scene):
    """Check scene refs, endpoint normals, and every open box against each segment."""
    from cffg.viz.scene import Path

    wires = derive(template).wires
    boxes = scene.boxes()
    paths = {p.ref: p for p in scene.items if isinstance(p, Path) and p.kind == "wire"}
    assert paths.keys() == wires.keys()
    for ref, path in paths.items():
        points = path.points
        for (x, y), (u, v) in zip(points, points[1:]):
            assert (x == u) != (y == v)
            for box in boxes.values():
                assert (max(x, u) <= box.x - box.w / 2 or min(x, u) >= box.x + box.w / 2
                        or max(y, v) <= box.y - box.h / 2 or min(y, v) >= box.y + box.h / 2), (ref, box.name)
        for end, (x, y), (u, v) in zip(wires[ref].node_ends(), (points[0], points[-1]),
                                        (points[1], points[-2])):
            box = boxes[f"factor:{end.factor}"]
            assert ((abs(x - box.x) == box.w / 2 and abs(y - box.y) < box.h / 2 and (u - x) * (x - box.x) > 0)
                    or (abs(y - box.y) == box.h / 2 and abs(x - box.x) < box.w / 2 and (v - y) * (y - box.y) > 0))
    return paths


@pytest.mark.parametrize("kind,degree", [("gaussian_linear", 5), ("gaussian_mixture_linear", 6)])
def test_gaussian_junction_wires_at_every_level(kind, degree):
    from cffg.viz import build_scene, render_svg
    from cffg.viz.scene import Dome, Path

    t = gaussian_junction(kind)
    wires = derive(t).wires_of("junction")
    assert len(wires) == degree
    assert sum(len(w.node_ends()) == 1 for w in wires) == 3
    for level in range(5):
        scene = build_scene(t, level=level)
        paths = _wire_paths(t, scene)
        anchors = {
            paths[w.id].points[0 if w.ends[0].factor == "junction" else -1]
            for w in wires if len(w.node_ends()) == 2
        }
        items = scene.by_ref("factor:junction")
        if level == 4:
            dome, = [item for item in items if isinstance(item, Dome)]
            assert set(dome.anchors) == anchors
        else:
            assert {item.points[0] for item in items if isinstance(item, Path)} == anchors
    assert render_svg(t) == render_svg(t)


def test_structural_sides_and_anchors_have_priority_over_elastic_terminals():
    from cffg.viz.diagram import NodeEnd, TerminalEnd, WireView
    from cffg.viz.layout import _anchor_points, _assign_sides

    d = derive(gaussian_junction("gaussian_mixture_linear"))
    grid = {"junction": (0, 0), "xprior": (-2, 1), "yprior": (-1, -1), "zprior": (0, -2)}
    before = _assign_sides(d, grid, [])
    wid = "var:extra"
    d.wires[wid] = WireView(wid, "extra", (NodeEnd("junction", "port"), TerminalEnd(TerminalKind.OPEN)))
    after = _assign_sides(d, grid, [])
    for role, side in (("in", "left"), ("out", "left"), ("selector", "up")):
        assert before["junction", role] == after["junction", role] == side
    assert after["junction", "port"] == "down"
    assert after["junction", "cov"] == "right"  # port precedes first blob
    centers = {name: (100 * col, 84 * row) for name, (col, row) in grid.items()}
    anchors = _anchor_points(d, after, centers, grid, 36, (100, 84))
    assert anchors["junction", "out"][1] < anchors["junction", "in"][1]


def test_32_glue_ports_use_free_sides_before_sharing():
    from cffg.viz import build_scene

    t = Template("crowded")
    roles = {f"x{i}": f"x{i}" for i in range(32)}
    t.factors["constraint"] = FactorSpec("constraint", "alldifferent", edges=roles)
    for v in roles:
        t.variables[v] = VariableSpec(v, cardinality=32)
        t.ports[v] = PortSpec(v, v, PortRole.GLUE)
    validate(t)
    scene = build_scene(t)
    paths = _wire_paths(t, scene)
    box = scene.boxes()["factor:constraint"]
    by_side = {}
    for v in roles:  # natural order, including x2 before x10
        x, y = paths[f"var:{v}"].points[0]
        if abs(x - box.x) == box.w / 2:
            side = "left" if x < box.x else "right"
            fraction = (y - box.y) / box.h + 0.5
        else:
            side = "up" if y < box.y else "down"
            fraction = (x - box.x) / box.w + 0.5
        by_side.setdefault(side, []).append(fraction)
        if v == "x3":
            assert len(by_side) == 4
    for fractions in by_side.values():
        n = len(fractions)
        assert fractions == pytest.approx([(i + 1) / (n + 1) for i in range(n)])


def test_obstructed_wire_uses_shorter_clear_corridor():
    from cffg.viz.routing import WireRouter

    router = WireRouter({"left": (0, 0), "middle": (1, 0), "right": (3, 0)}, (100, 84), 36)
    def route(y):
        return router.connect((18, y, "right"), (282, y, "left"), "left", "right")
    # Above the centreline, the upper half-pitch lane saves 4 * 7 pixels.
    assert min(y for _, y in route(-7)) == -42
    assert max(y for _, y in route(7)) == 42
    assert min(y for _, y in route(0)) == -42  # coordinate-order tie break


def test_long_terminal_stops_before_next_occupied_cell():
    from cffg.viz.routing import WireRouter

    router = WireRouter({"source": (0, 0), "distant": (0, -2)}, (100, 84), 36)
    length = router.terminal("source", (0, -18), 250, (0, -1))
    assert 84 < length < 2 * 84 - 36
    assert router.terminal("source", (0, 18), 250, (0, 1)) == 250
