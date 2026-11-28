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
    dashed: bool = False
    label_size: float = 13.0
    domain_label: Label | None = None


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
    label_at: tuple[float, float] | None = None
    label_size: float = 9.5
    dimension: int = 0
    radius: float = 6.0

    @property
    def ring_radius(self) -> float | None:
        if self.dimension and self.kind in (TerminalKind.DATA, TerminalKind.OPEN):
            return self.radius + 3.0
        return None


def label_width(text: str, size: float) -> float:
    # Leave room for italic overhang and full-size subscript characters.
    return (len(text) + 0.35) * size


@dataclass(frozen=True)
class Label:
    """A caption's SVG baseline, anchor and conservative headless text extent."""

    ref: str
    text: str
    x: float
    y: float
    anchor: str
    font_role: str
    size: float

    @property
    def width(self) -> float:
        return label_width(self.text, self.size)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = self.x - (self.width / 2 if self.anchor == "middle" else self.size * 0.12)
        # SVG includes the font ascent and the lowered run for trailing digits.
        return left, self.y - 1.25 * self.size, left + self.width, self.y + 0.65 * self.size

    @classmethod
    def for_item(cls, item: Path | Terminal) -> Label | None:
        if not item.label or item.label_at is None:
            return None
        terminal = isinstance(item, Terminal)
        return cls(item.ref, item.label, *item.label_at, "middle" if terminal else "start",
                   "terminal" if terminal else "wire", item.label_size)


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


@dataclass(frozen=True)
class PlateBox:
    """The literature's rectangle-with-count badge (cff-cpxj): a labeled box
    drawn around a replicated fragment's own diagram, with an `×N` (or `×R×C`)
    badge in the corner. Drawn from the plate SPEC, never the expansion."""

    ref: str
    x: float  # top-left
    y: float
    w: float
    h: float
    label: str  # plate name
    badge: str  # "×16", "×4×4", ...


class ItemKind(str, Enum):
    pass  # reserved: scenes are heterogeneous lists, discriminated by type


SceneItem = Box | Path | Bead | Dome | Terminal | EllipsisMark | CompositeBox | PlateBox


@dataclass
class Scene:
    """The drawable result: items in painter's order (first = bottom)."""

    name: str
    width: float
    height: float
    items: list[SceneItem] = field(default_factory=list)
    wire_corner_limit: float | None = None

    def by_ref(self, ref: str) -> list[SceneItem]:
        return [i for i in self.items if i.ref == ref]

    def labels(self) -> list[Label]:
        """Wire and terminal captions in drawing order, including repeated blob owners."""
        labels = []
        for item in self.items:
            if isinstance(item, (Path, Terminal)) and (caption := Label.for_item(item)):
                labels.append(caption)
            if isinstance(item, Path) and item.domain_label:
                labels.append(item.domain_label)
        return labels

    def boxes(self) -> dict[str, Box]:
        return {i.ref: i for i in self.items if isinstance(i, Box)}
