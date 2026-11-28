"""Schedule annotations trace compiled slot ownership, including composite writes."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace

import jax
import numpy as np
import pytest
from test_schedule import hmm_arrays, hmm_template

from cffg import Engine, compile_template
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    Template,
    VariableSpec,
)
from cffg.models.tmaze import build_tmaze
from cffg.schedule import ScanStage
from cffg.viz import MessageArrow, build_scene, derive_schedule_overlay, render_svg


def overlay_for(model, schedule="auto"):
    t, arrays = model
    e = Engine(compile_template(t, arrays=arrays), schedule=schedule)
    return t, e, derive_schedule_overlay(t, e)


def test_events_match_actual_batch_scatter_calls(monkeypatch):
    import cffg.schedule_program as program

    t, engine, overlay = overlay_for((hmm_template(2), hmm_arrays(2)))
    assert overlay.kind == "two_pass"
    assert derive_schedule_overlay(t, engine) == overlay
    calls = []
    scatter = program._set_disc_rows

    def record(arena, messages, slots, cards, rows):
        calls.append(tuple(map(int, np.asarray(slots).flat)))
        return scatter(arena, messages, slots, cards, rows)

    monkeypatch.setattr(program, "_set_disc_rows", record)
    unit = program.make_propagate(engine.arena, engine.schedule.stages)
    s = engine.init_state()
    jax.eval_shape(
        unit,
        s.messages,
        s.g_eta,
        s.g_lam,
        s.ef_eta,
        s.g_obs,
        s.g_mask,
        s.ef_obs,
        s.ef_mask,
        s.params,
    )
    assert len(calls) == 1 + overlay.events[-1].step
    for i, slots in enumerate(calls):
        events = overlay.select_step(i).events
        assert {e.batch for e in events} == {i}
        assert tuple(e.slot for e in events) == slots
        for event in events:
            edge = engine.arena.edges[event.wire_ref[4:]]
            source = event.source_ref.removeprefix("factor:")
            assert t.factors[source].all_edges()[event.source_role] == edge.name
            if event.target_ref.startswith("factor:"):
                target = event.target_ref.removeprefix("factor:")
                assert edge.name in t.factors[target].all_edges().values()
                assert event.slot == (edge.slot0 if source < target else edge.slot1)
            else:
                assert t.ports[event.target_ref[5:]].variable == edge.name
                assert event.slot == edge.slot0
    with pytest.raises(ValueError, match="no message writes"):
        overlay.select_step(-1)


def test_scans_expand_by_tick_and_element_at_scale():
    n = 256
    _, e, o = overlay_for((hmm_template(n), hmm_arrays(n)))
    assert any(isinstance(stage, ScanStage) for stage in e.schedule.stages)
    expected = []
    for stage in e.schedule.stages:
        rounds = (
            ((el, el.rows[i]) for i in range(stage.length) for el in stage.elems)
            if isinstance(stage, ScanStage)
            else [(stage, stage.rows)]
        )
        for unit, rows in rounds:
            expected.extend(
                map(int, np.asarray(e.arena.groups[unit.group].scatter)[rows].flat)
            )
    assert [event.slot for event in o.events] == expected
    assert len(expected) == 2 * (n * (2 + 3 + 2) + 1)


def test_synchronous_writes_match_runtime_transport(monkeypatch):
    from cffg.execution import synchronous

    t, engine, mixed = overlay_for(build_tmaze(steps=2))
    calls = []
    for name in ("_set_disc_rows", "_set_disc_block"):
        original = getattr(synchronous, name)

        def record(arena, messages, slots, *args, original=original):
            calls.extend(map(int, np.asarray(slots).flat))
            return original(arena, messages, slots, *args)

        monkeypatch.setattr(synchronous, name, record)

    for schedule in (
        mixed,
        derive_schedule_overlay(t, Engine(engine.arena, schedule="flood")),
    ):
        calls.clear()
        unit = synchronous.make_sweep(
            engine.arena, skip_scheduled=schedule.kind == "mixed"
        )
        s = engine.init_state()
        jax.eval_shape(
            unit,
            s.messages,
            s.g_eta,
            s.g_lam,
            s.g_obs,
            s.g_mask,
            s.ef_eta,
            s.ef_obs,
            s.ef_mask,
            s.params,
        )
        events = [e for e in schedule.events if e.phase == "synchronous"]
        assert Counter(e.slot for e in events) == Counter(calls)
        assert len({e.step for e in events}) == 1
        assert {e.source_ref for e in events if e.gfe} == {"factor:obs1", "factor:obs2"}
    assert list(dict.fromkeys(e.phase for e in mixed.events)) == [
        "prime",
        "synchronous",
        "island_refresh",
    ]


def test_directed_arrows_and_base_scene_at_all_compression_levels():
    for model in ((hmm_template(2), hmm_arrays(2)), build_tmaze(2)):
        t, _, o = overlay_for(model)
        for level in range(5):
            base = build_scene(t, level=level)
            scene = build_scene(t, level=level, overlay=o)
            assert scene.items[: len(base.items)] == base.items
            arrows = scene.items[len(base.items) :]
            assert all(isinstance(a, MessageArrow) for a in arrows)
            groups = {}
            for e in o.events:
                groups.setdefault((e.wire_ref, e.source_ref, e.source_role), []).append(
                    e.step
                )
            assert {
                (a.ref, a.source_ref, a.source_role): list(a.steps) for a in arrows
            } == groups
            heads = set()
            for a in arrows:
                assert (a.ref, a.tip) not in heads
                heads.add((a.ref, a.tip))
                path = next(p for p in base.by_ref(a.ref) if hasattr(p, "points"))
                assert any(
                    (x == u == a.tip[0] and min(y, v) <= a.tip[1] <= max(y, v))
                    or (y == v == a.tip[1] and min(x, u) <= a.tip[0] <= max(x, u))
                    for (x, y), (u, v) in zip(path.points, path.points[1:])
                )
        selected = o.select_step(0)
        assert len(
            [
                a
                for a in build_scene(t, overlay=selected).items
                if isinstance(a, MessageArrow)
            ]
        ) == len(selected.events)
        assert render_svg(t, overlay=o) == render_svg(t, overlay=o)


def test_seal_and_role_mismatches_rejected():
    t, e, o = overlay_for(build_tmaze(1))
    wrong = deepcopy(t)
    # Swapping two same-sized variables preserves the set of edges.
    f = wrong.factors["obs1"]
    f.edges["in"], f.edges["out"] = f.edges["out"], f.edges["in"]
    with pytest.raises(ValueError):
        derive_schedule_overlay(wrong, e)
    with pytest.raises(ValueError, match="different template"):
        build_scene(wrong, overlay=o)
    e.schedule = replace(e.schedule, reason="different")
    with pytest.raises(ValueError, match="fingerprint"):
        derive_schedule_overlay(t, e)


def test_expanded_plate_maps_writes_to_individual_copies():
    from test_plates import pixel_fragment

    from cffg.core import Expose, PlateSpec, expand_plates

    host = Template(
        "pixels", plates={"p": PlateSpec(pixel_fragment(), (2,), (Expose(),))}
    )
    engine = Engine(compile_template(host))
    with pytest.raises(ValueError, match="expand_plates"):
        derive_schedule_overlay(host, engine)
    expanded = expand_plates(host)
    overlay = derive_schedule_overlay(expanded, engine)
    assert {e.source_ref for e in overlay.events} == {
        "factor:" + name for name in expanded.factors
    }
    assert any(
        isinstance(item, MessageArrow)
        for item in build_scene(expanded, overlay=overlay).items
    )


def test_implicit_cross_arena_and_composite_writes():
    from test_dirichlet_param_edges import _make_template
    from test_gauss_vmp import gamma_gauss_prec
    from test_hybrid_engine import mixlin_template
    from test_hybrid_gfe import _prior_template
    from test_recurrence_link import _template
    from test_shared_goals import mixed_goals

    models = [
        (_prior_template(), "mixture", {"out": "gauss", "selector": "disc"}, True),
        (
            mixlin_template(),
            "mix",
            {"out": "gauss", "in": "gauss", "selector": "disc"},
            False,
        ),
        (gamma_gauss_prec(n=1), "gp0", {"out": "gauss", "prec": "ef"}, False),
        (
            (_make_template("matrix_cat", (2, 2)), {"alpha0": np.ones((2, 2))}),
            "lik",
            {"out": "disc", "in": "disc", "matrix": "disc"},
            False,
        ),
    ]
    recurrence = _template(weights=np.ones((2, 1, 1)), offsets=np.zeros((2, 1)))
    link = next(
        f.name for f in recurrence[0].factors.values() if f.type == "recurrence_link"
    )
    models.append(
        (recurrence, link, {"out": "disc", "in": "disc", "drive": "gauss"}, False)
    )
    for model, source, roles, gfe in models:
        _t, engine, overlay = overlay_for(model, schedule="flood")
        events = [e for e in overlay.events if e.source_ref == "factor:" + source]
        assert {e.source_role: e.arena for e in events if e.gfe == gfe} == roles
        assert len(events) == len(roles)
    _, engine, overlay = overlay_for(mixed_goals())
    writes = {e.slot for e in overlay.events if e.phase == "synchronous" and e.gfe}
    for p in engine.arena.goal_parameter_plans:
        assert set(p.count_scatter) | set(p.prior_scatter) <= writes


def test_colored_prime_and_delta_flood_reason():
    # A ring and disconnected prior pair give a colored component and primed forest.
    t = Template("ring_and_forest")
    t.blobs = {"d": BlobRef("d", "d.npy"), "B": BlobRef("B", "B.npy")}
    for v in ("a", "b", "c", "x"):
        t.variables[v] = VariableSpec(v, 2)
    for i, (a, b) in enumerate((("a", "b"), ("b", "c"), ("c", "a"))):
        f = f"tm{i}"
        t.factors[f] = FactorSpec(
            f, "matrix_cat", {"in": a, "out": b}, param_blobs={"matrix": "B"}
        )
    for f in ("left", "right"):
        t.factors[f] = FactorSpec(
            f, "categorical", {"out": "x"}, param_blobs={"statistic": "d"}
        )
    arrays = {"d": np.full(2, 0.5), "B": np.eye(2)}
    _, _, o = overlay_for((t, arrays))
    assert o.kind == "colored"
    assert list(dict.fromkeys(e.phase for e in o.events)) == ["prime", "colored"]
    t.constraints.append(ConstraintSpec(ConstraintKind.DELTA, "a"))
    _, engine, o = overlay_for((t, arrays))
    assert o.kind == "flood" and o.reason == engine.schedule.reason
    assert len(o.events) == 8 and {e.step for e in o.events} == {0}


def test_iterative_transition_constraint_and_mean_field_ownership():
    from test_schedule import A, D, vmp_unit_template

    from cffg.core import PortRole, PortSpec
    from cffg.models.rollout import build_rollout

    model = build_rollout(
        "learned dynamics",
        steps=1,
        epistemic_b=True,
        d=np.full(2, 0.5),
        c=np.full(2, 0.5),
        e=np.full(2, 0.5),
        A=np.eye(2),
        Bs=np.stack([np.eye(2)] * 2),
    )
    _, _, o = overlay_for(model)
    assert {
        e.source_role for e in o.events if e.source_ref == "factor:tm1" and e.gfe
    } == {"in", "out", "selector"}
    _, _, o = overlay_for((vmp_unit_template(), {"d": D, "A": A}))
    assert {
        e.source_role for e in o.events if e.phase == "synchronous" and not e.gfe
    } == {"in", "out"}

    t = Template("clamped_csp")
    t.variables = {v: VariableSpec(v, 2) for v in ("x0", "x1")}
    t.ports = {v: PortSpec(v, v, PortRole.DATA) for v in t.variables}
    t.factors["all"] = FactorSpec("all", "alldifferent", {v: v for v in t.variables})
    _, _, o = overlay_for((t, {}))
    assert {(e.source_ref, e.target_ref) for e in o.events} == {
        ("factor:all", "port:x0"),
        ("factor:all", "port:x1"),
    }


def test_wire_direction_on_detour_and_svg_titles_and_markers(tmp_path):
    import xml.etree.ElementTree as ET

    from cffg.viz import SvgStyle, derive, save_svg, to_svg
    from cffg.viz.scene import Bead, Box, EllipsisMark, Path, Terminal

    t, _, o = overlay_for(build_tmaze())
    scene = build_scene(t, overlay=o, offsets={"prior": (0, -2)})
    wires = derive(t).wires
    detours = 0
    for arrow in (a for a in scene.items if isinstance(a, MessageArrow)):
        (path,) = [p for p in scene.by_ref(arrow.ref) if isinstance(p, Path)]
        first = wires[arrow.ref].ends[0]
        forwards = ("factor:" + first.factor, first.role) == (
            arrow.source_ref,
            arrow.source_role,
        )
        tx, ty = arrow.tip
        dx, dy = arrow.direction
        intended = 1 if forwards else -1
        assert any(
            (
                (x == u == tx and min(y, v) <= ty <= max(y, v))
                or (y == v == ty and min(x, u) <= tx <= max(x, u))
            )
            and ((u - x) * dx + (v - y) * dy) * intended > 0
            for (x, y), (u, v) in zip(path.points, path.points[1:])
        )
        detours += len(path.points) > 2
        bx, by = tx - dx * arrow.size, ty - dy * arrow.size
        extent = (
            min(tx, bx) - abs(dy) * arrow.size / 2,
            min(ty, by) - abs(dx) * arrow.size / 2,
            max(tx, bx) + abs(dy) * arrow.size / 2,
            max(ty, by) + abs(dx) * arrow.size / 2,
        )
        for item in (
            scene.by_ref(arrow.ref)
            + scene.by_ref(arrow.source_ref)
            + scene.by_ref(arrow.target_ref)
        ):
            if isinstance(item, Box):
                w, h = item.w / 2, item.h / 2
            elif isinstance(item, (Bead, Terminal)):
                w = h = 6
            elif isinstance(item, EllipsisMark):
                w = h = 10
            elif isinstance(item, Path) and item.label_at:
                x, y = item.label_at
                font = SvgStyle().label_size
                left, top, right, bottom = extent
                assert (
                    right <= x
                    or left >= x + len(item.label) * font
                    or bottom <= y - font
                    or top >= y + font / 3
                )
                continue
            else:
                continue
            assert (
                extent[2] <= item.x - w
                or extent[0] >= item.x + w
                or extent[3] <= item.y - h
                or extent[1] >= item.y + h
            )
    assert detours > 0
    root = ET.fromstring(to_svg(scene, SvgStyle(stroke="#345678")))
    markers = root.findall(".//{*}marker")
    assert len(markers) == 2
    assert {m.attrib["id"]: m.find("{*}path").attrib["fill"] for m in markers} == {
        "cffg-message-gfe": "#c62828",
        "cffg-message-ordinary": "#345678",
    }
    for arrow in (a for a in scene.items if isinstance(a, MessageArrow)):
        text = (
            ("GFE" if arrow.gfe else "Ordinary")
            + f" message {arrow.source_ref}.{arrow.source_role} → {arrow.target_ref}; {arrow.ref}; steps "
        )
        assert text + ", ".join(map(str, arrow.steps)) in {
            x.text for x in root.findall(".//{*}title")
        }
    assert save_svg(t, tmp_path / "plan.svg", overlay=o).read_text() == render_svg(
        t, overlay=o
    )
