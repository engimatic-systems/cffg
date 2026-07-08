"""CFFG compression (Part I §5.6, p.19) as presentation directives.

The recipe's four passes turn the fully-annotated diagram into the compact
form the papers use for real models. Level k applies passes 1..k:

1. Summarise bead chains by their terminating beads — empty on-wire
   normalisation beads disappear (symbol-carrying beads and terminals stay).
2. Nodes with no factorisation constraints lose their empty internal beads.
3. The same nodes lose their internal edge extensions (bare boxes).
4. All remaining internal beads are pushed to the node border; boxes shrink.

"No factorisation constraints" is `NodeView.default_factorisation`: a joint
marginal over *all* edges. A node with a pointmass edge (parameter blob or
data clamp) is not default — its q-marginal omits that edge — so it keeps its
beads, exactly the f_d case in Figs 16-17.

Passes never change *what the diagram means*, only what is drawn; hence they
produce directives over the semantic model rather than a mutated copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cffg.viz.diagram import BeadGroup, Diagram

FULL_COMPRESSION = 4


class Placement(str, Enum):
    INTERNAL = "internal"  # beads inside the box, wires extended through the border
    BORDER = "border"  # beads straddle the border at their edges' attachment
    # points; multi-role groups render as an arc/dome spanning them (Fig 19)


@dataclass(frozen=True)
class NodeDirective:
    groups: tuple[BeadGroup, ...]  # beads actually drawn
    placement: Placement
    internal_edges: bool  # draw the edge extensions through the border


@dataclass(frozen=True)
class WireDirective:
    bead: bool  # draw the on-wire bead at the wire midpoint
    symbol: str  # "" empty, "delta", ...


@dataclass
class Presentation:
    """What to draw, per element, at one compression level. Geometry-free:
    the layout stage consumes this together with the Diagram itself."""

    level: int
    compact: bool  # shrink boxes (pass 4's payoff, Fig 19 right)
    nodes: dict[str, NodeDirective]
    wires: dict[str, WireDirective]


def present(diagram: Diagram, level: int = FULL_COMPRESSION) -> Presentation:
    if not 0 <= level <= FULL_COMPRESSION:
        raise ValueError(f"compression level must be 0..4, got {level}")

    nodes: dict[str, NodeDirective] = {}
    for name, node in diagram.nodes.items():
        groups = node.groups
        if level >= 2 and node.default_factorisation:
            groups = tuple(g for g in groups if g.symbol)  # pass 2
        stripped = level >= 3 and node.default_factorisation and not groups  # pass 3
        nodes[name] = NodeDirective(
            groups=groups,
            placement=Placement.BORDER if level >= 4 else Placement.INTERNAL,
            internal_edges=level < 4 and not stripped,
        )

    wires: dict[str, WireDirective] = {}
    for wid, wire in diagram.wires.items():
        if wire.bead is None:  # pointmass wires never carry a q bead
            wires[wid] = WireDirective(bead=False, symbol="")
        else:  # pass 1: only symbol-carrying beads survive
            drawn = wire.bead != "" or level < 1
            wires[wid] = WireDirective(bead=drawn, symbol=wire.bead)

    return Presentation(
        level=level, compact=level >= FULL_COMPRESSION, nodes=nodes, wires=wires
    )
