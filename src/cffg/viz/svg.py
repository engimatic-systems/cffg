"""SVG backend: Scene -> paper-style vector graphics (Part I's TikZ look).

Monochrome, serif-italic math labels with numeric subscripts, quarter-arc
wire corners, domes for border-pushed joint beads. `render_svg` is the
one-call entry point; `save_png` rasterises through the optional `viz`
extra (cairosvg) and fails with an actionable message without it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from math import hypot
from pathlib import Path as FilePath

from cffg.core.ir import Template
from cffg.viz.compress import FULL_COMPRESSION, present
from cffg.viz.diagram import BeadShape, TerminalKind, derive
from cffg.viz.layout import LayoutStyle, layout
from cffg.viz.layout import _shift as _shift_item
from cffg.viz.scene import (
    Bead,
    Box,
    CompositeBox,
    Dome,
    EllipsisMark,
    Path,
    PlateBox,
    Scene,
    Terminal,
)

_SERIF = "STIXGeneral, 'STIX Two Math', 'Cambria Math', 'Times New Roman', serif"
_BEAD_SYMBOLS = {"delta": "∂", "E": "\U0001d53c"}  # ∂, 𝔼


@dataclass(frozen=True)
class SvgStyle:
    stroke: str = "#1a1a1a"
    background: str = "#ffffff"
    stroke_width: float = 1.3
    bead_r: float = 6.0
    terminal_size: float = 9.0  # pointmass square side
    corner_r: float = 10.0  # wire bend rounding
    glyph_size: float = 15.0
    label_size: float = 13.0
    tiny_size: float = 9.5


def render_svg(
    template: Template,
    *,
    level: int = FULL_COMPRESSION,
    layout_style: LayoutStyle = LayoutStyle(),
    svg_style: SvgStyle = SvgStyle(),
) -> str:
    """Template -> SVG string at the given compression level (0..4)."""
    return to_svg(build_scene(template, level=level, style=layout_style), svg_style)


def build_scene(
    template: Template, *, level: int = FULL_COMPRESSION, style: LayoutStyle = LayoutStyle()
) -> Scene:
    diagram = derive(template)
    if diagram.nodes:
        scene = layout(diagram, present(diagram, level), style)
    else:  # a pure plate host has no top-level factors to lay out
        m = style.margin
        scene = Scene(name=template.name, width=2 * m, height=2 * m, items=[])
    if template.plates:
        scene = _attach_plates(template, scene, level, style)
    return scene


def _attach_plates(
    template: Template, scene: Scene, level: int, style: LayoutStyle
) -> Scene:
    """Draw each plate as the literature's rectangle-with-count badge: the
    fragment's own diagram inside a labeled box, stacked below the host."""
    pad = 26.0
    gap = 28.0
    items = list(scene.items)
    y = scene.height + gap
    width = scene.width
    for name in sorted(template.plates):
        spec = template.plates[name]
        frag_scene = build_scene(spec.fragment, level=level, style=style)
        dx = style.margin + pad
        dy = y + pad + 14.0  # leave room for the label strip
        items += [_shift_item(it, dx, dy) for it in frag_scene.items]
        box_w = frag_scene.width + 2 * pad
        box_h = frag_scene.height + 2 * pad + 14.0
        badge = "×" + "×".join(str(d) for d in spec.dims)
        items.append(
            PlateBox(f"plate:{name}", style.margin, y, box_w, box_h, name, badge)
        )
        y += box_h + gap
        width = max(width, style.margin + box_w + style.margin)
    return Scene(name=scene.name, width=width, height=y, items=items)


def save_svg(template: Template, path: str | FilePath, **kw) -> FilePath:
    out = FilePath(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_svg(template, **kw))
    return out


def save_png(
    template: Template, path: str | FilePath, *, scale: float = 2.0, **kw
) -> FilePath:
    try:
        import cairosvg
    except ImportError as e:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "PNG output needs the 'viz' extra: uv sync --extra viz"
        ) from e
    out = FilePath(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(
        bytestring=render_svg(template, **kw).encode(), write_to=str(out), scale=scale
    )
    return out


# --------------------------------------------------------------------------


def to_svg(scene: Scene, style: SvgStyle = SvgStyle()) -> str:
    s = style
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{scene.width:.0f}" '
            f'height="{scene.height:.0f}" viewBox="0 0 {scene.width:.0f} {scene.height:.0f}" '
            f'font-family="{_SERIF}">'
        ),
        f"<title>{escape(scene.name)}</title>",
        f'<rect width="100%" height="100%" fill="{s.background}"/>',
        (
            f'<g stroke="{s.stroke}" stroke-width="{s.stroke_width}" fill="none" '
            'stroke-linecap="round">'
        ),
    ]
    for item in scene.items:
        parts.append(_render_item(item, s))
    parts += ["</g>", "</svg>"]
    return "\n".join(p for p in parts if p) + "\n"


def _render_item(item, s: SvgStyle) -> str:
    if isinstance(item, PlateBox):
        return (
            f"<g><title>{escape(item.ref)}</title>"
            f'<rect x="{item.x:.1f}" y="{item.y:.1f}" width="{item.w:.1f}" '
            f'height="{item.h:.1f}" rx="6"/>'
            f'<text x="{item.x + 8:.1f}" y="{item.y + 15:.1f}" '
            f'font-size="{s.label_size}" font-style="italic" stroke="none" '
            f'fill="{s.stroke}">{_mathify(item.label, s.label_size)}</text>'
            f'<text x="{item.x + item.w - 8:.1f}" y="{item.y + 15:.1f}" '
            f'font-size="{s.label_size}" stroke="none" fill="{s.stroke}" '
            f'text-anchor="end">{escape(item.badge)}</text></g>'
        )
    if isinstance(item, CompositeBox):
        return (
            f'<rect x="{item.x:.1f}" y="{item.y:.1f}" width="{item.w:.1f}" '
            f'height="{item.h:.1f}" rx="4" stroke-dasharray="5 4"/>'
        )
    if isinstance(item, Path):
        d = _rounded_path(item.points, s.corner_r if item.kind == "wire" else 0.0)
        label = ""
        if item.label and item.label_at:
            lx, ly = item.label_at
            label = (
                f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="{s.label_size}" '
                f'font-style="italic" stroke="none" fill="{s.stroke}">'
                f"{_mathify(item.label, s.label_size)}</text>"
            )
        return f'<path d="{d}"/>{label}'
    if isinstance(item, Box):
        title = f"<title>{escape(item.name)}</title>"
        gx, gy = item.glyph_at or (item.x, item.y)
        size = s.glyph_size * (0.85 if item.glyph_at else 1.0)
        eq = item.glyph == "="
        return (
            f"<g>{title}"
            f'<rect x="{item.x - item.w / 2:.1f}" y="{item.y - item.h / 2:.1f}" '
            f'width="{item.w:.1f}" height="{item.h:.1f}" fill="{s.background}"/>'
            f'<text x="{gx:.1f}" y="{gy:.1f}" font-size="{size:.1f}" '
            f'{"" if eq else "font-style=\"italic\" "}stroke="none" fill="{s.stroke}" '
            'text-anchor="middle" dominant-baseline="central">'
            f"{_mathify(item.glyph, size)}</text></g>"
        )
    if isinstance(item, Dome):
        return _dome(item, s)
    if isinstance(item, Bead):
        return _bead(item, s)
    if isinstance(item, Terminal):
        return _terminal(item, s)
    if isinstance(item, EllipsisMark):
        dots = []
        for k in (-1, 0, 1):
            dx, dy = (k * 7.0, 0) if item.angle == 0 else (0, k * 7.0)
            dots.append(
                f'<circle cx="{item.x + dx:.1f}" cy="{item.y + dy:.1f}" r="2.1" '
                f'fill="{s.stroke}" stroke="none"/>'
            )
        return "".join(dots)
    return ""


def _bead(item: Bead, s: SvgStyle) -> str:
    r = s.bead_r
    if item.shape is BeadShape.SQUARE:
        shape = (
            f'<rect x="{item.x - r:.1f}" y="{item.y - r:.1f}" width="{2 * r:.1f}" '
            f'height="{2 * r:.1f}" fill="{"#ffffff"}"/>'
        )
    else:
        shape = f'<circle cx="{item.x:.1f}" cy="{item.y:.1f}" r="{r:.1f}" fill="#ffffff"/>'
    symbol = _BEAD_SYMBOLS.get(item.symbol, "")
    text = (
        f'<text x="{item.x:.1f}" y="{item.y:.1f}" font-size="{2 * r * 0.95:.1f}" '
        f'stroke="none" fill="{s.stroke}" text-anchor="middle" '
        f'dominant-baseline="central">{symbol}</text>'
        if symbol
        else ""
    )
    return shape + text


def _terminal(item: Terminal, s: SvgStyle) -> str:
    if item.kind is TerminalKind.POINTMASS:
        h = s.terminal_size / 2
        mark = (
            f'<rect x="{item.x - h:.1f}" y="{item.y - h:.1f}" '
            f'width="{2 * h:.1f}" height="{2 * h:.1f}" fill="{s.stroke}"/>'
        )
        lx, ly = item.x, item.y - h - 5
    elif item.kind is TerminalKind.DATA:
        mark = (
            f'<circle cx="{item.x:.1f}" cy="{item.y:.1f}" r="{s.bead_r:.1f}" '
            f'fill="{s.stroke}"/>'
            f'<text x="{item.x:.1f}" y="{item.y:.1f}" font-size="{s.bead_r * 1.7:.1f}" '
            'stroke="none" fill="#ffffff" text-anchor="middle" '
            f'dominant-baseline="central">{_BEAD_SYMBOLS["delta"]}</text>'
        )
        lx, ly = item.x, item.y + s.bead_r + 11
    else:  # OPEN: dangling edge terminated by an (empty) bead, §5.2
        mark = (
            f'<circle cx="{item.x:.1f}" cy="{item.y:.1f}" r="{s.bead_r:.1f}" '
            'fill="#ffffff"/>'
        )
        lx, ly = item.x, item.y + s.bead_r + 11
    label = (
        f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="{s.tiny_size}" stroke="none" '
        f'fill="{s.stroke}" text-anchor="middle" font-style="italic">'
        f"{_mathify(item.label, s.tiny_size)}</text>"
        if item.label
        else ""
    )
    return f"<g><title>{escape(item.ref)}</title>{mark}{label}</g>"


def _dome(item: Dome, s: SvgStyle) -> str:
    """The border-pushed joint bead (Fig 19): one quadratic between the two
    farthest anchors — a dome for opposite sides, a corner-rounding curve for
    adjacent ones — plus straight stubs joining any remaining anchors to it."""
    anchors = sorted(item.anchors)
    pairs = [(a, b) for i, a in enumerate(anchors) for b in anchors[i + 1 :]]
    a, b = max(pairs, key=lambda p: _dist(*p))
    rest = [p for p in anchors if p not in (a, b)]
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    if abs(a[0] - b[0]) < 1.0 or abs(a[1] - b[1]) < 1.0:  # opposite sides
        if rest:  # bulge away from the remaining anchors
            rx = sum(p[0] for p in rest) / len(rest) - mx
            ry = sum(p[1] for p in rest) / len(rest) - my
            n = hypot(rx, ry) or 1.0
            ux, uy = -rx / n, -ry / n
        elif abs(a[1] - b[1]) < 1.0:
            ux, uy = 0.0, 1.0  # horizontal chord: dome opens downward (Fig 22)
        else:
            ux, uy = 1.0, 0.0
        half = _dist(a, b) / 2
        cx, cy = mx + ux * half * 1.6, my + uy * half * 1.6
    else:  # adjacent sides: hug the shared corner (the f_d look, Fig 19)
        cx = a[0] if abs(a[0] - item.cx) > abs(b[0] - item.cx) else b[0]
        cy = a[1] if abs(a[1] - item.cy) > abs(b[1] - item.cy) else b[1]

    d = [f"M{a[0]:.1f},{a[1]:.1f} Q{cx:.1f},{cy:.1f} {b[0]:.1f},{b[1]:.1f}"]
    apex = ((a[0] + 2 * cx + b[0]) / 4, (a[1] + 2 * cy + b[1]) / 4)
    for p in rest:
        d.append(f"M{p[0]:.1f},{p[1]:.1f} L{apex[0]:.1f},{apex[1]:.1f}")
    return f'<path d="{" ".join(d)}"/>'


def _rounded_path(points, r: float) -> str:
    pts = list(points)
    if len(pts) < 2:
        return ""
    d = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for prev, bend, nxt in zip(pts, pts[1:], pts[2:]):
        rr = min(r, _dist(prev, bend) / 2, _dist(bend, nxt) / 2)
        if rr < 1.0:
            d.append(f"L{bend[0]:.1f},{bend[1]:.1f}")
            continue
        ax, ay = _toward(bend, prev, rr)
        bx, by = _toward(bend, nxt, rr)
        d.append(f"L{ax:.1f},{ay:.1f} Q{bend[0]:.1f},{bend[1]:.1f} {bx:.1f},{by:.1f}")
    d.append(f"L{pts[-1][0]:.1f},{pts[-1][1]:.1f}")
    return " ".join(d)


def _dist(a, b) -> float:
    return hypot(b[0] - a[0], b[1] - a[1])


def _toward(frm, to, dist) -> tuple[float, float]:
    d = _dist(frm, to) or 1.0
    return frm[0] + (to[0] - frm[0]) * dist / d, frm[1] + (to[1] - frm[1]) * dist / d


def _mathify(text: str, size: float) -> str:
    """Trailing digits become subscripts: zt1 -> zt₁ (paper style). Absolute
    font-size and dy, not percentages/baseline-shift: cairosvg mangles both."""
    m = re.fullmatch(r"(.*?[^\d])(\d+)", text)
    if not m:
        return escape(text)
    base, sub = m.groups()
    return (
        f'{escape(base)}<tspan font-size="{size * 0.7:.1f}" '
        f'dy="{size * 0.22:.1f}">{escape(sub)}</tspan>'
    )
