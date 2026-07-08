"""Scene graph: typed drawable primitives with pixel coordinates.

The layout stage produces a Scene; backends (SVG today, an interactive GUI
later) consume it. Every item carries a `ref` back to the template element it
depicts ("factor:tm1", "var:x2", "blob:A", "port:sense", "gfe:obs1") so a
GUI can hit-test a primitive and resolve it to the model — the scene is the
contract that keeps renderers dumb.

Coordinates are y-down pixels; (0, 0) is the top-left of the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cffg.viz.diagram import BeadShape, TerminalKind


@dataclass(frozen=True)
class Box:
    """A factor node: square outline, glyph centered, name for tooltips."""

    ref: str
    x: float  # center
    y: float  # center
    w: float
    h: float
    glyph: str
    name: str
    glyph_at: tuple[float, float] | None = None  # None: centered; set when
    # beads/domes occupy the center (papers put the label top-left, Fig 14)


@dataclass(frozen=True)
class Path:
    """An open polyline (a wire, or an internal edge extension). Corners are
    bend points; backends may round them (the papers' quarter-arc corners)."""

    ref: str
    points: tuple[tuple[float, float], ...]
    kind: str = "wire"  # "wire" | "internal"
    label: str = ""
    label_at: tuple[float, float] | None = None


@dataclass(frozen=True)
class Bead:
    """A q-marginal bead: on a wire, inside a box, or straddling its border."""

    ref: str
    x: float
    y: float
    shape: BeadShape
    symbol: str = ""  # "", "delta", "E", ...
    filled: bool = False


@dataclass(frozen=True)
class Dome:
    """A joint node-marginal bead pushed to the border (Fig 19): an arc
    through `anchors` (the member edges' attachment points) bulging into the
    box toward (cx, cy)."""

    ref: str
    anchors: tuple[tuple[float, float], ...]
    cx: float
    cy: float


@dataclass(frozen=True)
class Terminal:
    """A wire ending: pointmass square, filled data bead, or open bead."""

    ref: str
    x: float
    y: float
    kind: TerminalKind
    label: str = ""


@dataclass(frozen=True)
class EllipsisMark:
    """Continuation dots at a glue port: the model rolls out further."""

    ref: str
    x: float
    y: float
    angle: float = 0.0  # degrees; 0 = horizontal continuation


@dataclass(frozen=True)
class CompositeBox:
    """Dashed rounded box around a GFE composite's members."""

    ref: str
    x: float  # top-left
    y: float
    w: float
    h: float


class ItemKind(str, Enum):
    pass  # reserved: scenes are heterogeneous lists, discriminated by type


SceneItem = Box | Path | Bead | Dome | Terminal | EllipsisMark | CompositeBox


@dataclass
class Scene:
    """The drawable result: items in painter's order (first = bottom)."""

    name: str
    width: float
    height: float
    items: list[SceneItem] = field(default_factory=list)

    def by_ref(self, ref: str) -> list[SceneItem]:
        return [i for i in self.items if i.ref == ref]

    def boxes(self) -> dict[str, Box]:
        return {i.ref: i for i in self.items if isinstance(i, Box)}
