"""CFFG diagram rendering in the papers' visual algebra (Part I §5).

Pipeline: `Template` -> semantic diagram (`derive`) -> compression passes
(Part I p.19 recipe) -> grid layout -> scene graph -> SVG/PNG. Each stage is
importable on its own so GUIs can stop at the scene graph and hit-test.
"""

from cffg.viz.compress import FULL_COMPRESSION, Presentation, present
from cffg.viz.diagram import (
    BeadGroup,
    BeadShape,
    Diagram,
    NodeEnd,
    NodeView,
    CompositeView,
    TerminalEnd,
    TerminalKind,
    WireView,
    derive,
)
from cffg.viz.layout import LayoutStyle, layout
from cffg.viz.scene import Scene
from cffg.viz.svg import SvgStyle, build_scene, render_svg, save_png, save_svg, to_svg

__all__ = [
    "BeadGroup",
    "BeadShape",
    "CompositeView",
    "Diagram",
    "FULL_COMPRESSION",
    "LayoutStyle",
    "NodeEnd",
    "NodeView",
    "Presentation",
    "Scene",
    "SvgStyle",
    "TerminalEnd",
    "TerminalKind",
    "WireView",
    "build_scene",
    "derive",
    "layout",
    "present",
    "render_svg",
    "save_png",
    "save_svg",
    "to_svg",
]
