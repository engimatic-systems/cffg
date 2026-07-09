"""Semantic diagram model: a Template expressed in CFFG visual algebra.

This is the level-independent middle representation between the IR and the
renderer. It answers "what is drawn" (Part I §5): factor boxes with their
q-marginal bead structure, wires with on-wire beads, pointmass/data terminals,
rollout ellipses at glue ports, and GFE composite boxes. It does *not* answer
"where" (layout) or "at which compression level" (the p.19 passes) — those
are separate stages that consume this model.

The derivation makes the papers' default explicit: with no annotations, q has
a joint marginal per node and a normalisation bead per internal edge. Every
ConstraintSpec, port, and parameter binding then perturbs that default:

- ``MEAN_FIELD``       -> one bead per q-edge instead of the joint bead
- ``P_SUBSTITUTION``   -> the substituted role's bead becomes a square (Fig 13)
- ``DELTA`` on an edge -> the on-wire bead carries a delta symbol (Fig 11)
- ``param_blobs``      -> black-square pointmass terminal (Fig 10 discussion);
                          the role drops out of q, so the node loses its
                          default factorisation (the f_d case, Fig 16)
- DATA ports           -> filled, terminating data bead (Fig 10)
- GLUE ports           -> ellipsis: the model continues by composition
- GOAL/PRIOR ports     -> dangling edge terminated by an (empty) bead (§5.2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cffg.core.ir import (
    ConstraintKind,
    ConstraintSpec,
    FACTOR_TYPES,
    FactorSpec,
    PortRole,
    Template,
)

# Conventional glyphs used by the papers' figures (Part I Figs 22/26) and the
# wider FFG community (Loeliger 2007: 𝒩 for normal factors). Box glyphs are
# capped at 3 characters so labels never overflow the box (test_viz_diagram
# enforces this for every registered factor type — add an entry here when you
# register a type; the fallback truncates, it does not save you).
GLYPHS = {
    "categorical": "Cat",
    "matrix_cat": "T",
    "transition_mixture": "TM",
    "equality": "=",
    "dirichlet": "Dir",
    "gaussian": "𝒩",
    "gaussian_linear": "𝒩A",
    "gamma": "Γ",
    "beta": "β",
    "poisson_obs": "Poi",
    "bernoulli_obs": "Ber",
}


class BeadShape(str, Enum):
    CIRCLE = "circle"  # marginal of q
    SQUARE = "square"  # P-substituted: p(x|rest) used in expectation (Fig 13)


class TerminalKind(str, Enum):
    POINTMASS = "pointmass"  # black square: delta-distributed model parameter
    DATA = "data"  # filled delta bead: observation clamp, terminates the edge
    OPEN = "open"  # dangling edge terminated by a bead (form-constraint anchor)
    ELLIPSIS = "ellipsis"  # glue port: the graph continues by composition


@dataclass(frozen=True)
class NodeEnd:
    """A wire endpoint attached to a factor box at a named role."""

    factor: str
    role: str


@dataclass(frozen=True)
class TerminalEnd:
    """A wire endpoint that terminates outside any factor box."""

    kind: TerminalKind
    label: str = ""  # blob or port name
    ref: str = ""  # "blob:<name>" or "port:<name>" back-reference


@dataclass(frozen=True)
class WireView:
    """A drawable edge. `id` is "var:<name>" or "blob:<factor>.<role>"."""

    id: str
    label: str  # variable name; empty for parameter wires (terminal is labeled)
    ends: tuple[NodeEnd | TerminalEnd, NodeEnd | TerminalEnd]
    bead: str | None = None  # None: no bead in q (pointmass wires); "": empty
    # bead symbols: "" (default normalisation), "delta", "E" (moment matching)

    def node_ends(self) -> list[NodeEnd]:
        return [e for e in self.ends if isinstance(e, NodeEnd)]


@dataclass(frozen=True)
class BeadGroup:
    """One factor of the node's q-marginal: q^n(s_a^n) over `roles` (§5.1)."""

    roles: tuple[str, ...]
    shape: BeadShape = BeadShape.CIRCLE
    symbol: str = ""


@dataclass(frozen=True)
class NodeView:
    factor: str
    type: str
    glyph: str
    groups: tuple[BeadGroup, ...]  # factorisation of q(s_a) over q-roles
    pointmass_roles: tuple[str, ...]  # roles absent from q (blobs, data clamps)
    default_factorisation: bool  # joint over *all* edges: nothing to draw
    # when compressed (steps 2-3 of the p.19 recipe apply to this node)


@dataclass(frozen=True)
class CompositeView:
    """A GFE composite (Part II §4): dashed box around obs + goal factors."""

    name: str
    members: tuple[str, ...]  # (observation factor, goal factor)
    x_variable: str  # the P-substituted edge joining them


@dataclass
class Diagram:
    name: str
    nodes: dict[str, NodeView] = field(default_factory=dict)
    wires: dict[str, WireView] = field(default_factory=dict)
    composites: list[CompositeView] = field(default_factory=list)

    def wires_of(self, factor: str) -> list[WireView]:
        return [w for w in self.wires.values() if any(e.factor == factor for e in w.node_ends())]


_PORT_TERMINALS = {
    PortRole.DATA: TerminalKind.DATA,
    PortRole.GLUE: TerminalKind.ELLIPSIS,
    PortRole.GOAL: TerminalKind.OPEN,
    PortRole.PRIOR: TerminalKind.OPEN,
}


def derive(template: Template) -> Diagram:
    """Express `template` in the visual algebra, at full (level-0) detail."""

    node_constraints: dict[str, list[ConstraintSpec]] = {}
    edge_constraints: dict[str, list[ConstraintSpec]] = {}
    for c in template.constraints:
        bucket = edge_constraints if c.target in template.variables else node_constraints
        bucket.setdefault(c.target, []).append(c)

    data_clamped = {
        p.variable for p in template.ports.values() if p.role is PortRole.DATA
    }

    diagram = Diagram(name=template.name)
    for f in template.factors.values():
        diagram.nodes[f.name] = _node_view(f, node_constraints.get(f.name, []), data_clamped)

    endpoints: dict[str, list[NodeEnd]] = {v: [] for v in template.variables}
    for f in template.factors.values():
        for role, var in f.all_edges().items():
            endpoints[var].append(NodeEnd(f.name, role))

    for var, ends in endpoints.items():
        diagram.wires[f"var:{var}"] = _variable_wire(
            template, var, ends, edge_constraints.get(var, [])
        )
    for f in template.factors.values():
        for role, blob in sorted(f.param_blobs.items()):
            wid = f"blob:{f.name}.{role}"
            diagram.wires[wid] = WireView(
                id=wid,
                label=blob,  # the parameter's name labels its wire (Fig 22's d, A, c)
                ends=(
                    NodeEnd(f.name, role),
                    TerminalEnd(TerminalKind.POINTMASS, label=blob, ref=f"blob:{blob}"),
                ),
                bead=None,  # pointmass in the model: not a factor of q
            )

    diagram.composites = _detect_composites(template, node_constraints)
    return diagram


def _psub_role(constraints: list[ConstraintSpec]) -> str | None:
    for c in constraints:
        if c.kind is ConstraintKind.P_SUBSTITUTION:
            return c.detail_dict().get("substituted_role", "out")
    return None


def _node_view(
    f: FactorSpec, constraints: list[ConstraintSpec], data_clamped: set[str]
) -> NodeView:
    ftype = FACTOR_TYPES.get(f.type)
    # Stable role order: the type's declared order, then any extras.
    declared = (*ftype.edge_roles, *ftype.param_roles) if ftype else ()
    all_roles = [r for r in declared if r in f.all_edges() or r in f.param_blobs]
    all_roles += [r for r in {**f.all_edges(), **f.param_blobs} if r not in all_roles]

    pointmass = tuple(
        r
        for r in all_roles
        if r in f.param_blobs or f.all_edges().get(r) in data_clamped
    )
    q_roles = [r for r in all_roles if r not in pointmass]

    psub = _psub_role(constraints)
    mean_field = any(c.kind is ConstraintKind.MEAN_FIELD for c in constraints)

    groups: list[BeadGroup] = []
    plain = [r for r in q_roles if r != psub]
    if mean_field:
        groups += [BeadGroup((r,)) for r in plain]
    elif plain:
        groups.append(BeadGroup(tuple(plain)))
    if psub is not None:
        groups.append(BeadGroup((psub,), shape=BeadShape.SQUARE))

    return NodeView(
        factor=f.name,
        type=f.type,
        glyph=GLYPHS.get(f.type, f.type[:3]),  # 3-char cap; tooltip has the name
        groups=tuple(groups),
        pointmass_roles=pointmass,
        default_factorisation=not mean_field and psub is None and not pointmass,
    )


def _variable_wire(
    template: Template,
    var: str,
    ends: list[NodeEnd],
    constraints: list[ConstraintSpec],
) -> WireView:
    symbol = ""
    bead: str | None = ""
    for c in constraints:
        if c.kind is ConstraintKind.DELTA:
            symbol = "delta"

    pair: tuple[NodeEnd | TerminalEnd, NodeEnd | TerminalEnd]
    ports = template.ports_of(var)
    if len(ends) == 2:
        pair = (ends[0], ends[1])
    elif ends and ports:
        port = ports[0]
        kind = _PORT_TERMINALS[port.role]
        if kind is TerminalKind.DATA:
            bead = None  # the filled terminal *is* the (pointmass) bead
        pair = (ends[0], TerminalEnd(kind, label=port.name, ref=f"port:{port.name}"))
    elif ends:
        # Degree-1 edge without a port: invalid per cffg.core.validate, but
        # drawing it (terminated by a bead, §5.2) beats crashing the renderer.
        pair = (ends[0], TerminalEnd(TerminalKind.OPEN, label=var))
    else:
        raise ValueError(f"variable {var!r} is connected to no factor")
    return WireView(id=f"var:{var}", label=var, ends=pair, bead=symbol if bead == "" else bead)


def _detect_composites(
    template: Template, node_constraints: dict[str, list[ConstraintSpec]]
) -> list[CompositeView]:
    """Template-level mirror of the compiler's GFE-composite detection
    (cffg.compiler: P-substituted matrix_cat facing a fixed-statistic goal).
    Tolerant where the compiler errors: an unmatched P-substitution simply
    renders as a bare square bead with no dashed box.
    """

    composites: list[CompositeView] = []
    for fname, constraints in sorted(node_constraints.items()):
        f = template.factors.get(fname)
        if f is None or f.type != "matrix_cat" or _psub_role(constraints) != "out":
            continue
        x_var = f.edges["out"]
        facing = [
            g
            for g in template.factors.values()
            if g.name != fname and x_var in g.all_edges().values()
        ]
        if not facing:
            continue
        goal = facing[0]
        if goal.type == "categorical" and "statistic" in goal.param_blobs:
            composites.append(
                CompositeView(name=fname, members=(fname, goal.name), x_variable=x_var)
            )
    return composites
