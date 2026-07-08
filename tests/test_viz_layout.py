"""Grid layout: rail extraction, branch sides, terminal placement, stability.

The claims mirror the papers' layout conventions (Figs 22/26): time reads
left to right along the state chain, control priors sit above it, the
observation/goal stack hangs below, and — the editing property — growing a
model only adds rows/columns, never moves what was already there.
"""

import numpy as np
import pytest

from cffg.models.rollout import build_rollout
from cffg.viz import derive
from cffg.viz.compress import present
from cffg.viz.diagram import TerminalKind
from cffg.viz.layout import LayoutStyle, layout
from cffg.viz.scene import Bead, CompositeBox, EllipsisMark, Terminal


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
