"""CFFG intermediate representation.

A Template is a Forney-style factor graph fragment (factors = nodes, variables =
edges) plus the constraint annotations that make it a *constrained* FFG
(Koudahl et al. 2023, §5): factorization/form constraints and P-substitution
markers that together specify the variational family Q and the free-energy
functional, not just the generative model f.

Boundary edges are declared as typed ports. The port invariant (enforced by
cffg.core.validate) is the single mechanism behind unrolling, rollout gluing,
"no dangling edges," and template composition: every variable is either fully
connected (degree 2) or exposed through exactly one port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PortRole(str, Enum):
    """What a dangling boundary edge is waiting for."""

    DATA = "data"  # sensor clamp: observed one-hot value bound at runtime
    GOAL = "goal"  # goal prior over a future observation
    PRIOR = "prior"  # parameter/state prior factor
    GLUE = "glue"  # bonds to another template instance (unrolling, composition)


@dataclass(frozen=True)
class VariableSpec:
    """An edge in the FFG: a named random variable.

    Two domain families: discrete one-hot (`cardinality` = k) or Gaussian
    (`dim` = d, a point in R^d). Exactly one of the two is set; `family`
    names which. Messages, arenas, and rules dispatch on the family — a
    discrete edge lives in the log-probability arena, a Gaussian edge in
    the information-form (eta, Lambda) arena (cff-4e5z).
    """

    name: str
    cardinality: int = 0
    dim: int = 0

    def __post_init__(self):
        if (self.cardinality > 0) == (self.dim > 0):
            raise ValueError(
                f"variable {self.name!r}: exactly one of cardinality ({self.cardinality}) "
                f"and dim ({self.dim}) must be positive"
            )

    @property
    def family(self) -> str:
        return "gaussian" if self.dim else "discrete"

    @property
    def domain_key(self) -> tuple[str, int]:
        """Hashable domain identity: what must match across an equality node
        or a glue bond."""
        return (self.family, self.dim or self.cardinality)


@dataclass(frozen=True)
class BlobRef:
    """Named reference to an array stored outside the template (never inline).

    `init` (cff-wx28) is an optional canonical-initializer spec (see
    cffg.inits.spec): when the file is absent at compile time, the array is
    materialized from the spec — templates with warm-start parameters are
    self-contained, and a checkpoint written to `path` overrides the warm
    start on reload."""

    name: str
    path: str  # relative to the bundle root
    init: tuple[tuple[str, str], ...] = ()  # sorted (key, value) pairs; () = none


@dataclass(frozen=True)
class FactorSpec:
    """A node in the FFG.

    `edges` maps the factor type's roles to variable names. `params` maps a
    parameter role to its source: an edge (learned, e.g. A under a Dirichlet
    prior) or a blob (fixed, e.g. the T-maze B matrices). Each parameter role
    must be bound by exactly one of the two.
    """

    name: str
    type: str  # key into FACTOR_TYPES
    edges: dict[str, str] = field(default_factory=dict)  # role -> variable
    param_edges: dict[str, str] = field(default_factory=dict)  # role -> variable
    param_blobs: dict[str, str] = field(default_factory=dict)  # role -> blob name

    def all_edges(self) -> dict[str, str]:
        """All connected variables by role, structural and parameter alike."""
        return {**self.edges, **self.param_edges}


class ConstraintKind(str, Enum):
    DELTA = "delta"  # pointmass form constraint on an edge marginal (MAP reads)
    MEAN_FIELD = "mean_field"  # naive factorization of a node marginal
    P_SUBSTITUTION = "p_substitution"  # q(x) -> p(x|rest); local VFE becomes GFE
    MOMENT_MATCHING = "moment_matching"  # reserved for EP; no rules registered yet


# Which kinds target an edge (variable name) vs a node (factor name).
EDGE_CONSTRAINTS = frozenset({ConstraintKind.DELTA})
NODE_CONSTRAINTS = frozenset(
    {ConstraintKind.MEAN_FIELD, ConstraintKind.P_SUBSTITUTION, ConstraintKind.MOMENT_MATCHING}
)


@dataclass(frozen=True)
class ConstraintSpec:
    kind: ConstraintKind
    target: str  # variable name (edge kinds) or factor name (node kinds)
    detail: tuple[tuple[str, str], ...] = ()
    # detail is a sorted tuple of key/value pairs so specs stay hashable and
    # canonically serializable, e.g. (("substituted_role", "out"),).

    def detail_dict(self) -> dict[str, str]:
        return dict(self.detail)


@dataclass(frozen=True)
class PortSpec:
    name: str
    variable: str
    role: PortRole


@dataclass(frozen=True)
class FactorType:
    """Structural signature of a factor kind. Math lives in cffg.rules (M2+)
    and cffg.gauss (Gaussian family, cff-4e5z).

    `family` constrains the structural edges' domains: "discrete" and
    "gaussian" require every edge in that family; "any" (equality) accepts
    either family but `symmetric_domain` then forces one shared domain."""

    name: str
    edge_roles: tuple[str, ...]  # required structural edges
    param_roles: tuple[str, ...] = ()  # each bound by exactly one edge or blob
    symmetric_domain: bool = False  # all structural edges share one domain
    family: str = "discrete"  # "discrete" | "gaussian" | "any"


FACTOR_TYPES: dict[str, FactorType] = {}


def register_factor_type(ft: FactorType) -> FactorType:
    if ft.name in FACTOR_TYPES:
        raise ValueError(f"factor type {ft.name!r} already registered")
    overlap = set(ft.edge_roles) & set(ft.param_roles)
    if overlap:
        raise ValueError(f"factor type {ft.name!r}: roles {sorted(overlap)} are both edge and param")
    FACTOR_TYPES[ft.name] = ft
    return ft


# The discrete vocabulary of the papers (Part II §5, Part I App. C).
register_factor_type(FactorType("categorical", edge_roles=("out",), param_roles=("statistic",)))
register_factor_type(FactorType("matrix_cat", edge_roles=("out", "in"), param_roles=("matrix",)))
register_factor_type(
    FactorType("transition_mixture", edge_roles=("out", "in", "selector"), param_roles=("matrices",))
)
register_factor_type(FactorType("dirichlet", edge_roles=("out",), param_roles=("counts",)))
register_factor_type(
    FactorType("equality", edge_roles=("a", "b", "c"), symmetric_domain=True, family="any")
)

# The Gaussian vocabulary (cff-4e5z): enough for linear-Gaussian state-space
# models. `gaussian` is the prior/leaf node N(out; mean, cov); `gaussian_linear`
# is the conditional N(out; matrix @ in + offset, cov) — one type covers both
# Kalman transitions and emissions.
register_factor_type(
    FactorType("gaussian", edge_roles=("out",), param_roles=("mean", "cov"), family="gaussian")
)
register_factor_type(
    FactorType(
        "gaussian_linear",
        edge_roles=("out", "in"),
        param_roles=("matrix", "offset", "cov"),
        family="gaussian",
    )
)


@dataclass
class Template:
    """A CFFG fragment: the unit of serialization, composition, and compilation."""

    name: str
    variables: dict[str, VariableSpec] = field(default_factory=dict)
    factors: dict[str, FactorSpec] = field(default_factory=dict)
    constraints: list[ConstraintSpec] = field(default_factory=list)
    ports: dict[str, PortSpec] = field(default_factory=dict)
    blobs: dict[str, BlobRef] = field(default_factory=dict)
    schema_version: str = "1"

    def degree(self, variable: str) -> int:
        """Number of factor endpoints attached to an edge (FFG max is 2)."""
        return sum(
            1 for f in self.factors.values() for v in f.all_edges().values() if v == variable
        )

    def ports_of(self, variable: str) -> list[PortSpec]:
        return [p for p in self.ports.values() if p.variable == variable]
