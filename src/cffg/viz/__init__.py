"""CFFG diagram rendering in the papers' visual algebra (Part I §5).

Pipeline: `Template` -> semantic diagram (`derive`) -> compression passes
(Part I p.19 recipe) -> grid layout -> scene graph -> SVG/PNG. Each stage is
importable on its own so GUIs can stop at the scene graph and hit-test.
"""

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

__all__ = [
    "BeadGroup",
    "BeadShape",
    "CompositeView",
    "Diagram",
    "NodeEnd",
    "NodeView",
    "TerminalEnd",
    "TerminalKind",
    "WireView",
    "derive",
]
