"""SVG backend: well-formedness, paper notation coverage, golden files.

Goldens pin the full pipeline for the canonical models. After a deliberate
visual change, regenerate with:

    CFFG_UPDATE_GOLDENS=1 uv run pytest tests/test_viz_svg.py
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from cffg.models.tmaze import build_tmaze
from cffg.viz import render_svg, save_svg

GOLDEN = Path(__file__).parent / "golden"


def elements(svg: str, tag: str) -> list:
    root = ET.fromstring(svg)  # raises on malformed XML
    return [e for e in root.iter() if e.tag.endswith(f"}}{tag}")]


@pytest.fixture(scope="module")
def tmaze():
    return build_tmaze()[0]


def test_svg_is_well_formed_at_every_level(tmaze):
    for level in range(5):
        ET.fromstring(render_svg(tmaze, level=level))


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


@pytest.mark.parametrize(
    "name,level",
    [("hmm_slice", 0), ("hmm_slice", 4), ("tmaze", 4)],
)
def test_golden(name, level, hmm_slice, tmaze):
    template = {"hmm_slice": hmm_slice, "tmaze": tmaze}[name]
    svg = render_svg(template, level=level)
    path = GOLDEN / f"{name}_l{level}.svg"
    if os.environ.get("CFFG_UPDATE_GOLDENS"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(svg)
    assert path.exists(), "golden missing; run with CFFG_UPDATE_GOLDENS=1"
    assert svg == path.read_text(), (
        f"render drifted from {path.name}; if intended, regenerate with "
        "CFFG_UPDATE_GOLDENS=1"
    )
