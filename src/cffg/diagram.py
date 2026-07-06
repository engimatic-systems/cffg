"""Deterministic SVG diagrams for CFFG templates.

The renderer is intentionally dependency-free. It lays out factors, variables,
parameter blobs, constraints, and ports on stable lanes inside a fixed-aspect
viewBox so small structural changes tend to appear as local visual changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import re
from statistics import median

from cffg.core.ir import Template


@dataclass(frozen=True)
class DiagramStyle:
    width: int = 1280
    height: int = 720
    margin_x: float = 90.0
    margin_y: float = 70.0
    column_gap: float = 620.0
    row_gap: float = 58.0
    node_width: float = 132.0
    node_height: float = 34.0
    font_family: str = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"


@dataclass(frozen=True)
class _Node:
    key: str
    kind: str
    label: str
    detail: str
    column: int
    order: str


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    label: str
    kind: str = "structural"


def render_template_svg(template: Template, *, style: DiagramStyle = DiagramStyle()) -> str:
    """Render `template` as a standalone SVG string.

    Layout is deterministic: node columns come from scoped time suffixes when
    available, otherwise from sorted connectivity. Nodes use fixed lanes by
    semantic kind, and the SVG viewBox expands while the visible aspect ratio
    remains fixed through `preserveAspectRatio`.
    """

    nodes, edges = _diagram_graph(template)
    positions, view_w, view_h = _layout(nodes, style)
    return _render_svg(template.name, nodes, edges, positions, view_w, view_h, style)


def save_template_svg(
    template: Template,
    path: str | Path,
    *,
    style: DiagramStyle = DiagramStyle(),
    overwrite: bool = True,
) -> bool:
    """Write a template diagram. Returns False when skipped by `overwrite=False`."""

    out = Path(path)
    if out.exists() and not overwrite:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_template_svg(template, style=style))
    return True


def _diagram_graph(template: Template) -> tuple[list[_Node], list[_Edge]]:
    columns: dict[str, int] = {}
    for name in template.variables:
        columns[f"var:{name}"] = _column_from_name(name)
    for name, factor in template.factors.items():
        connected = [columns.get(f"var:{v}", _column_from_name(v)) for v in factor.all_edges().values()]
        columns[f"factor:{name}"] = int(round(median(connected))) if connected else _column_from_name(name)
    for name, port in template.ports.items():
        columns[f"port:{name}"] = columns.get(f"var:{port.variable}", _column_from_name(port.variable))
    for name, blob in template.blobs.items():
        users = [
            columns[f"factor:{factor.name}"]
            for factor in template.factors.values()
            if name in factor.param_blobs.values()
        ]
        columns[f"blob:{name}"] = int(round(median(users))) if users else _column_from_name(blob.name)
    for i, constraint in enumerate(template.constraints):
        target_key = (
            f"factor:{constraint.target}"
            if constraint.target in template.factors
            else f"var:{constraint.target}"
        )
        columns[f"constraint:{i}"] = columns.get(target_key, _column_from_name(constraint.target))

    nodes: list[_Node] = []
    for name, blob in sorted(template.blobs.items()):
        nodes.append(_Node(f"blob:{name}", "blob", name, blob.path, columns[f"blob:{name}"], name))
    for i, constraint in enumerate(template.constraints):
        detail = ", ".join(f"{k}={v}" for k, v in constraint.detail)
        nodes.append(
            _Node(
                f"constraint:{i}",
                "constraint",
                constraint.kind.value,
                f"{constraint.target}" + (f" | {detail}" if detail else ""),
                columns[f"constraint:{i}"],
                f"{constraint.kind.value}:{constraint.target}:{i}",
            )
        )
    for name, factor in sorted(template.factors.items()):
        nodes.append(
            _Node(
                f"factor:{name}",
                "factor",
                name,
                factor.type,
                columns[f"factor:{name}"],
                f"{_natural_key(name)}",
            )
        )
    for name, variable in sorted(template.variables.items()):
        detail = f"card {variable.cardinality}" if variable.cardinality > 0 else ""
        nodes.append(
            _Node(
                f"var:{name}",
                "variable",
                name,
                detail,
                columns[f"var:{name}"],
                f"{_natural_key(name)}",
            )
        )
    for name, port in sorted(template.ports.items()):
        nodes.append(
            _Node(
                f"port:{name}",
                "port",
                name,
                port.role.value,
                columns[f"port:{name}"],
                f"{port.role.value}:{_natural_key(name)}",
            )
        )

    edges: list[_Edge] = []
    for factor in sorted(template.factors.values(), key=lambda f: _natural_key(f.name)):
        for role, variable in sorted(factor.edges.items()):
            edges.append((_factor_var_edge(factor.name, variable, role, "structural")))
        for role, variable in sorted(factor.param_edges.items()):
            edges.append((_factor_var_edge(factor.name, variable, role, "parameter")))
        for role, blob in sorted(factor.param_blobs.items()):
            edges.append(_Edge(f"blob:{blob}", f"factor:{factor.name}", role, "parameter"))
    for name, port in sorted(template.ports.items()):
        edges.append(_Edge(f"var:{port.variable}", f"port:{name}", port.role.value, "port"))
    for i, constraint in enumerate(template.constraints):
        target = (
            f"factor:{constraint.target}"
            if constraint.target in template.factors
            else f"var:{constraint.target}"
        )
        edges.append(_Edge(f"constraint:{i}", target, constraint.kind.value, "constraint"))
    return nodes, edges


def _factor_var_edge(factor: str, variable: str, role: str, kind: str) -> _Edge:
    return _Edge(f"factor:{factor}", f"var:{variable}", role, kind)


def _layout(
    nodes: list[_Node], style: DiagramStyle
) -> tuple[dict[str, tuple[float, float]], float, float]:
    lanes = {
        "blob": 0,
        "constraint": 1,
        "factor": 2,
        "variable": 3,
        "port": 4,
    }
    x_offsets = {
        "blob": 0.0,
        "constraint": 100.0,
        "factor": 210.0,
        "variable": 410.0,
        "port": 560.0,
    }
    max_col = max((node.column for node in nodes), default=0)
    positions: dict[str, tuple[float, float]] = {}
    for kind, lane in lanes.items():
        by_col: dict[int, list[_Node]] = {}
        for node in nodes:
            if node.kind == kind:
                by_col.setdefault(node.column, []).append(node)
        for col, group in by_col.items():
            ordered = sorted(group, key=lambda n: n.order)
            x = style.margin_x + col * style.column_gap + x_offsets[kind]
            base_y = style.margin_y + lane * (style.row_gap * 2.0)
            for i, node in enumerate(ordered):
                positions[node.key] = (x, base_y + i * style.row_gap)

    max_rows = 1
    for kind in lanes:
        for col in range(max_col + 1):
            max_rows = max(max_rows, sum(1 for n in nodes if n.kind == kind and n.column == col))
    content_w = style.margin_x * 2 + max_col * style.column_gap + style.node_width + 620
    content_h = style.margin_y * 2 + 5 * style.row_gap * 2 + max_rows * style.row_gap
    view_w = max(style.width, content_w)
    view_h = max(style.height, content_h)
    aspect = style.width / style.height
    if view_w / view_h < aspect:
        view_w = view_h * aspect
    else:
        view_h = view_w / aspect
    return positions, view_w, view_h


def _render_svg(
    title: str,
    nodes: list[_Node],
    edges: list[_Edge],
    positions: dict[str, tuple[float, float]],
    view_w: float,
    view_h: float,
    style: DiagramStyle,
) -> str:
    css = f"""
    .bg {{ fill: #fbfaf7; }}
    .title {{ font: 700 24px {style.font_family}; fill: #1f2933; }}
    .legend {{ font: 500 13px {style.font_family}; fill: #52606d; }}
    .node text {{ font-family: {style.font_family}; text-anchor: middle; }}
    .label {{ font-size: 13px; font-weight: 700; fill: #17212b; }}
    .detail {{ font-size: 10px; fill: #52606d; }}
    .factor rect {{ fill: #e8f1ff; stroke: #2f68b7; }}
    .variable ellipse {{ fill: #f4f0ff; stroke: #7c5cc4; }}
    .blob path {{ fill: #eef8ef; stroke: #3b8a4b; }}
    .constraint rect {{ fill: #fff4df; stroke: #b7791f; stroke-dasharray: 5 3; }}
    .port rect {{ fill: #f4f6f8; stroke: #697386; }}
    .edge {{ fill: none; stroke: #7b8794; stroke-width: 1.4; }}
    .edge.parameter {{ stroke: #3b8a4b; stroke-dasharray: 4 3; }}
    .edge.constraint {{ stroke: #b7791f; stroke-dasharray: 5 4; }}
    .edge.port {{ stroke: #697386; stroke-dasharray: 2 4; }}
    .edge-label {{ font: 10px {style.font_family}; fill: #52606d; paint-order: stroke; stroke: #fbfaf7; stroke-width: 3px; }}
    """
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{style.width}" '
            f'height="{style.height}" viewBox="0 0 {view_w:.0f} {view_h:.0f}" '
            'preserveAspectRatio="xMidYMid meet">'
        ),
        f"<style>{css}</style>",
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#7b8794"/></marker></defs>',
        f'<rect class="bg" width="{view_w:.0f}" height="{view_h:.0f}"/>',
        f'<text class="title" x="32" y="42">{escape(title)}</text>',
        '<text class="legend" x="32" y="64">rect=factor  ellipse=variable  green=parameter/blob  dashed=constraint/port</text>',
    ]

    for edge in sorted(edges, key=lambda e: (e.kind, e.source, e.target, e.label)):
        if edge.source not in positions or edge.target not in positions:
            continue
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        y1 += style.node_height / 2
        y2 += style.node_height / 2
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        parts.append(
            f'<path class="edge {edge.kind}" marker-end="url(#arrow)" d="M{x1:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"/>'
        )
        if edge.label:
            parts.append(f'<text class="edge-label" x="{mx:.1f}" y="{my - 4:.1f}">{escape(edge.label)}</text>')

    for node in sorted(nodes, key=lambda n: (n.kind, n.column, n.order)):
        x, y = positions[node.key]
        parts.append(_render_node(node, x, y, style))
    parts.append("</svg>\n")
    return "\n".join(parts)


def _render_node(node: _Node, x: float, y: float, style: DiagramStyle) -> str:
    w, h = style.node_width, style.node_height
    label = _shorten(node.label, 24)
    detail = _shorten(node.detail, 28)
    if node.kind == "variable":
        shape = f'<ellipse cx="{x:.1f}" cy="{y + h / 2:.1f}" rx="{w / 2:.1f}" ry="{h / 2:.1f}"/>'
    elif node.kind == "blob":
        shape = (
            f'<path d="M{x - w / 2:.1f},{y + 7:.1f} '
            f'C{x - w / 2:.1f},{y - 2:.1f} {x + w / 2:.1f},{y - 2:.1f} {x + w / 2:.1f},{y + 7:.1f} '
            f'L{x + w / 2:.1f},{y + h - 7:.1f} '
            f'C{x + w / 2:.1f},{y + h + 2:.1f} {x - w / 2:.1f},{y + h + 2:.1f} {x - w / 2:.1f},{y + h - 7:.1f} Z"/>'
        )
    else:
        rx = 12 if node.kind == "port" else 4
        shape = f'<rect x="{x - w / 2:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"/>'
    detail_text = (
        f'<text class="detail" x="{x:.1f}" y="{y + 27:.1f}">{escape(detail)}</text>'
        if detail
        else ""
    )
    return (
        f'<g class="node {node.kind}">{shape}'
        f'<text class="label" x="{x:.1f}" y="{y + 16:.1f}">{escape(label)}</text>'
        f"{detail_text}</g>"
    )


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def _column_from_name(name: str) -> int:
    if "/" in name:
        prefix = name.rsplit("/", 1)[0]
        match = re.search(r"(?:^|/)t(\d+)(?:/|$)", prefix)
        if match:
            return int(match.group(1))
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def _natural_key(value: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(p) if p.isdigit() else p for p in parts)
