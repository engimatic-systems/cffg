"""SVG backend: well-formedness, paper notation coverage, golden files.

Goldens pin the full pipeline for the canonical models. After a deliberate
visual change, regenerate with:

    CFFG_UPDATE_GOLDENS=1 uv run pytest tests/test_viz_svg.py
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from cffg.models.tmaze import build_tmaze
from cffg.viz import build_scene, render_svg, save_svg, to_svg

GOLDEN = Path(__file__).parent / "golden"


def elements(svg: str, tag: str) -> list:
    root = ET.fromstring(svg)  # raises on malformed XML
    return [e for e in root.iter() if e.tag.endswith(f"}}{tag}")]


@pytest.fixture(scope="module")
def tmaze():
    return build_tmaze()[0]


def test_svg_is_well_formed_at_every_level(tmaze, viz_gallery):
    for level in range(5):
        svg = render_svg(tmaze, level=level)
        ET.fromstring(svg)
        viz_gallery(svg, tag=f"level {level}")  # the p.19 compression story


def test_notation_inventory(tmaze):
    svg = render_svg(tmaze, level=4)
    texts = [t.text for t in elements(svg, "text") if t.text]
    assert "Cat" in texts and "TM" in texts and "=" in texts
    # 2 planning slices: 2 dashed composite boxes + 2 P-substitution squares
    rects = elements(svg, "rect")
    assert sum(1 for r in rects if r.get("stroke-dasharray")) == 2
    squares = [r for r in rects if r.get("width") == r.get("height") == "12.0"]
    assert len(squares) == 2
    # subscripts render as absolute-size tspans (cairosvg chokes on %)
    assert all("%" not in (t.get("font-size") or "") for t in elements(svg, "tspan"))


def test_compression_removes_beads(tmaze):
    full = len(elements(render_svg(tmaze, level=0), "circle"))
    compressed = len(elements(render_svg(tmaze, level=4), "circle"))
    assert compressed < full


def test_save_svg(tmaze, tmp_path):
    out = save_svg(tmaze, tmp_path / "tmaze.svg")
    assert out.read_text().startswith("<svg")


def test_render_is_deterministic(tmaze):
    assert render_svg(tmaze) == render_svg(tmaze)


def test_svg_entrypoints_explain_template_and_scene_inputs(tmaze):
    scene = build_scene(tmaze)
    assert to_svg(scene) == render_svg(tmaze)
    with pytest.raises(TypeError, match="to_svg expects a Scene; use render_svg"):
        to_svg(tmaze)
    with pytest.raises(TypeError, match="expects a Template; use to_svg"):
        render_svg(scene)


def test_disconnected_and_empty_templates_render_without_inference():
    from cffg.core import Template
    from test_viz_layout import gaussian_components

    for template in (Template("empty"), gaussian_components(2)):
        ET.fromstring(render_svg(template))
        assert to_svg(build_scene(template)) == render_svg(template)


def test_gridworld_renders(viz_gallery):
    from cffg.models.gridworld import build_gridworld

    svg = render_svg(build_gridworld()[0], level=4)
    ET.fromstring(svg)
    viz_gallery(svg, tag="gridworld, compressed")


@pytest.mark.parametrize(
    "name,level",
    [("hmm_slice", 0), ("hmm_slice", 4), ("tmaze", 4),
     ("gaussian_linear", 0), ("gaussian_linear", 4),
     ("gaussian_mixture_linear", 0), ("gaussian_mixture_linear", 4),
     ("fig23_overlay", 4), ("fig24_overlay", 4), ("fig27_overlay", 4)],
)
def test_golden(name, level, hmm_slice, tmaze, viz_gallery):
    from test_viz_layout import gaussian_junction

    overlay = None
    if name.endswith("_overlay"):
        from cffg import Engine, compile_template
        from cffg.viz import derive_schedule_overlay

        template, arrays = build_tmaze(steps=2, past=int(name == "fig24_overlay"))
        if name == "fig23_overlay":
            arrays["e"] = arrays["e"].copy()
            arrays["e"][:] = [1, 0, 0, 0]  # Fix both future commands.
        overlay = derive_schedule_overlay(template, Engine(compile_template(template, arrays=arrays)))
        if name == "fig23_overlay":
            # Project the priming pass onto the forward prediction path for a fixed policy.
            forward_roles = {"categorical": {"out"}, "transition_mixture": {"out"}, "equality": {"b", "c"}}
            forward = tuple(
                e for e in overlay.events if e.phase == "prime"
                and e.source_role in forward_roles.get(template.factors[e.source_ref[7:]].type, ())
            )
            assert forward and all(not e.gfe for e in forward)
            overlay = replace(overlay, events=forward)
        elif name == "fig24_overlay":
            step = next(e.step for e in overlay.events if e.gfe and e.source_role == "in")
            overlay = overlay.select_step(step)
    else:
        template = gaussian_junction(name) if name.startswith("gaussian") else {"hmm_slice": hmm_slice, "tmaze": tmaze}[name]
    svg = render_svg(template, level=level, overlay=overlay)
    viz_gallery(svg, tag=f"{name}, level {level}")
    path = GOLDEN / f"{name}_l{level}.svg"
    if os.environ.get("CFFG_UPDATE_GOLDENS") == "1":
        path.parent.mkdir(exist_ok=True)
        path.write_text(svg)
    assert path.exists(), "golden missing; run with CFFG_UPDATE_GOLDENS=1"
    assert svg == path.read_text(), (
        f"render drifted from {path.name}; if intended, regenerate with "
        "CFFG_UPDATE_GOLDENS=1"
    )


def test_rounded_detour_clears_adjacent_box_even_at_large_radius():
    """Each quadratic curve stays in the rectangle containing its control polygon."""
    import re

    from cffg.viz import LayoutStyle, SvgStyle, derive
    from cffg.viz.layout import _place_factors
    from test_viz_layout import _wire_paths, gaussian_junction

    template = gaussian_junction("gaussian_linear")
    grid, _ = _place_factors(derive(template))
    columns = {"xprior": 0, "yprior": 1, "junction": 3}
    offsets = {name: (columns[name] - x, -y) for name, (x, y) in grid.items()}
    # Only 4px between boxes. The x wire has to pass the y prior.
    style = LayoutStyle(compact_box=80, compact_pitch_x=84, compact_pitch_y=84)
    scene = build_scene(template, level=4, style=style, offsets=offsets)
    paths = _wire_paths(template, scene)
    assert len(paths["var:x"].points) > 2
    svg = to_svg(scene, SvgStyle(corner_r=1000))
    group = ET.fromstring(svg).find("{*}g")
    wire_svg = group.findall("{*}path")[list(paths).index("var:x")]
    corners = re.findall(r"L([\d.,-]+) Q([\d.,-]+) ([\d.,-]+)", wire_svg.attrib["d"])
    assert corners
    rectangles = [g.find("{*}rect") for g in group.findall("{*}g")
                  if g.findtext("{*}title") in template.factors]
    assert len(rectangles) == len(template.factors)
    for corner in corners:
        xs, ys = zip(*(map(float, p.split(",")) for p in corner))
        for rect in rectangles:
            x, y, w, h = (float(rect.attrib[key]) for key in ("x", "y", "width", "height"))
            assert max(xs) <= x or min(xs) >= x + w or max(ys) <= y or min(ys) >= y + h
