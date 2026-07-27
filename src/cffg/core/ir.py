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

    Three domain families, tagged by which field is set (exactly one):
    discrete one-hot (`cardinality` = k), Gaussian (`dim` = d, a point in
    R^d), or scalar exponential-family (`ef` = a family name, cff-8s6a).
    Messages, arenas, and rules dispatch on the family — a discrete edge
    lives in the log-probability arena, a Gaussian edge in the
    information-form (eta, Lambda) arena (cff-4e5z), an EF edge in the
    natural-parameter arena. EF names come in two kinds (cffg.ef): message
    families (gamma, beta) whose edges carry natural-parameter messages,
    and observation families (poisson, bernoulli) whose edges are
    conditioned DATA ports only — their predictives leave the family, so
    no message ever flows on them.
    """

    name: str
    cardinality: int = 0
    dim: int = 0
    ef: str = ""

    def __post_init__(self):
        if (self.cardinality > 0) + (self.dim > 0) + bool(self.ef) != 1:
            raise ValueError(
                f"variable {self.name!r}: exactly one of cardinality ({self.cardinality}), "
                f"dim ({self.dim}), and ef ({self.ef!r}) must be set"
            )

    @property
    def family(self) -> str:
        return "gaussian" if self.dim else "ef" if self.ef else "discrete"

    @property
    def domain_key(self) -> tuple[str, int]:
        """Hashable domain identity: what must match across an equality node
        or a glue bond. EF domains are scalar; the family name is the key."""
        if self.ef:
            return (f"ef:{self.ef}", 1)
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
class LearnableSpec:
    """A declared learnable parameter: one conjugate posterior, and every blob
    through which the model reads it (cff-x0sv).

    The unit of learning is the *blob set*, not the factor, because blobs are
    how the IR already ties parameters — build_rollout names one `Bs` blob on
    every slice's transition, and plates tie by name across copies. Counts
    from every factor instance reading a blob in this set accumulate into one
    posterior, which is what makes learning transfer across states.

    A learnable spans several blobs when one parameter is read in more than
    one form. Two cases exist today:

    - consumption form — the discrete groups read expected log-parameters
      (E[log θ] = ψ(α) − ψ(α₀)) while the GFE composites read the raw counts α
      that their novelty term differentiates. build_rollout carries both as
      `A`/`Ac` and `Bs`/`Bc`.
    - joint family — an MNIW over a gaussian_linear node governs its
      `matrix`, `offset`, and `cov` blobs together, so all three name the
      same learnable.

    `family` keys cffg.learn.FAMILIES; the per-factor-type sufficient
    statistic comes from the factor type, not from this spec.

    `sources` restricts which factor instances *supply* statistics; empty
    means every reader does. Tying is about reading — a habit prior applies
    at every step of the horizon, so every planning slice reads the blob —
    while the statistic is the executed control alone. Where those differ it
    is a property of the model, so the model declares it.
    """

    name: str
    family: str  # "dirichlet" | ... (key into cffg.learn.FAMILIES)
    blobs: tuple[str, ...]  # every blob this posterior writes
    sources: tuple[str, ...] = ()  # factor instances supplying stats; () = all readers


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

    `family` constrains the structural edges' domains: "discrete",
    "gaussian", and "ef" require every edge in that family ("ef" types
    additionally pin each role to a specific EF family via
    EF_EDGE_FAMILIES); "any" (equality) accepts any family but
    `symmetric_domain` then forces one shared domain."""

    name: str
    edge_roles: tuple[str, ...]  # required structural edges
    param_roles: tuple[str, ...] = ()  # each bound by exactly one edge or blob
    symmetric_domain: bool = False  # all structural edges share one domain
    family: str = "discrete"  # "discrete" | "gaussian" | "any" | "constraint"
    variadic: bool = False  # edges are x0..x{n-1} for a runtime-chosen n
    # (hard-constraint factors: alldifferent over any arity — cff-e8zk)


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

# The hard-constraint vocabulary (cff-e8zk): indicator factors over a runtime-
# chosen number of discrete edges x0..x{n-1}. family="constraint" marks them as
# indicator factors (log-potential entries in {0, NEG}) — exempt from the
# column-stochastic check (unnormalized by design), carrying structural metadata
# only (no dense potential tensor is ever materialized). Their combinatorial
# message and node-energy functions live in cffg.csp. `variadic` edges are
# validated to be the contiguous roles x0..x{n-1}; per-type cardinality rules
# (alldifferent: shared K >= n; exactly/at_most: binary; table: enumerated) are
# checked at compile.
register_factor_type(
    FactorType("alldifferent", edge_roles=(), family="constraint", variadic=True)
)
register_factor_type(
    FactorType(
        "exactly_k", edge_roles=(), param_roles=("k",), family="constraint", variadic=True
    )
)
register_factor_type(
    FactorType(
        "at_most_k", edge_roles=(), param_roles=("k",), family="constraint", variadic=True
    )
)
register_factor_type(
    FactorType(
        "table", edge_roles=(), param_roles=("table",), family="constraint", variadic=True
    )
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

# The scalar exponential-family vocabulary (cff-8s6a): prior leaves over
# rates/precisions (gamma) and probabilities (beta), plus their conjugate
# observation factors. poisson_obs/bernoulli_obs are *conditioned* likelihoods
# (Gaussian DATA convention, cff-4e5z): the "out" edge must be a DATA port —
# the unobserved predictives (negative-binomial / Beta-Bernoulli mixture)
# leave the exponential family, so no message rule exists for them.
register_factor_type(
    FactorType("gamma", edge_roles=("out",), param_roles=("shape", "rate"), family="ef")
)
register_factor_type(
    FactorType("beta", edge_roles=("out",), param_roles=("a", "b"), family="ef")
)
register_factor_type(FactorType("poisson_obs", edge_roles=("out", "rate"), family="ef"))
register_factor_type(FactorType("bernoulli_obs", edge_roles=("out", "prob"), family="ef"))

# Per-role EF family constraints for family="ef" types: which `VariableSpec.ef`
# tag each structural edge must carry (validated in cffg.core.validate).
EF_EDGE_FAMILIES: dict[str, dict[str, str]] = {
    "gamma": {"out": "gamma"},
    "beta": {"out": "beta"},
    "poisson_obs": {"out": "poisson", "rate": "gamma"},
    "bernoulli_obs": {"out": "bernoulli", "prob": "beta"},
    # gaussian_prec is family="hybrid" (its out edge is Gaussian), but its
    # prec edge must carry an ef=gamma tag — the per-role check keys on f.type,
    # not on the factor's family, so it applies here just the same (cff-7c5o).
    "gaussian_prec": {"prec": "gamma"},
}

# The hybrid Gaussian-mixture vocabulary (cff-pksr): a discrete selector edge
# steers a continuous Gaussian output. `gaussian_mixture` is the mixture-of-
# priors leaf N(out; m_s, V_s); `gaussian_mixture_linear` is the switching
# conditional N(out; A_s in + b_s, V_s) — the continuous sibling of
# transition_mixture. Messages on the Gaussian edges are moment-matched
# collapses; the selector message is exact (cffg.mixg, DERIVATIONS.md D3).
register_factor_type(
    FactorType(
        "gaussian_mixture",
        edge_roles=("out", "selector"),
        param_roles=("mean", "cov"),
        family="hybrid",
    )
)
register_factor_type(
    FactorType(
        "gaussian_mixture_linear",
        edge_roles=("out", "in", "selector"),
        param_roles=("matrix", "offset", "cov"),
        family="hybrid",
    )
)

# The mixed Gaussian/EF vocabulary (cff-7c5o): a scalar Gaussian node whose
# precision lives on an ef:gamma edge. `gaussian_prec` is f(x; m, τ) =
# N(x; m, τ⁻¹) — the EF abstraction's first mixed client, joining the
# information-form (out) and natural-parameter (prec) arenas under a
# mean-field q(x)q(τ) VMP bead. Messages are closed-form (cffg.gauss,
# DERIVATIONS.md D4); the free energy is the exact mean-field VFE.
register_factor_type(
    FactorType(
        "gaussian_prec",
        edge_roles=("out", "prec"),
        param_roles=("mean",),
        family="hybrid",
    )
)

# Per-role domain-family constraints for family="hybrid" types: unlike EF
# (which pins the `ef` tag), each structural edge here must carry a whole
# VariableSpec.family (validated in cffg.core.validate).
HYBRID_EDGE_FAMILIES: dict[str, dict[str, str]] = {
    "gaussian_mixture": {"out": "gaussian", "selector": "discrete"},
    "gaussian_mixture_linear": {"out": "gaussian", "in": "gaussian", "selector": "discrete"},
    # out is a Gaussian edge, prec an EF edge (pinned to ef=gamma above).
    "gaussian_prec": {"out": "gaussian", "prec": "ef"},
}


# --- Plates: compact replication of a fragment (cff-cpxj) --------------------
# A plate declares "duplicate this fragment N times (or R×C) and wire the copies
# together by these rules." The wiring vocabulary is a small closed set of
# frozen rules — patterns, not arbitrary index arithmetic — expanded naively
# (materialized copies) in the compiler's grounding phase for phase 1.


@dataclass(frozen=True)
class Chain:
    """Bond copy i's `port_a` to copy (i+offset)'s `port_b` along `axis`.

    A 1-D chain plate reproduces unrolling's glue pattern; a grid is one Chain
    per axis. Both endpoints must be GLUE ports of the fragment (validated).
    Boundary copies keep their unbonded chain ports exposed (like unroll keeps
    t0's head and t{n-1}'s tail)."""

    port_a: str
    port_b: str
    axis: int = 0
    offset: int = 1


@dataclass(frozen=True)
class Broadcast:
    """Every copy's `port` faces one shared parent-level edge named `hub`.

    FFG edges have max degree 2, so expansion SYNTHESIZES an equality tree (a
    chain of ternary equality factors) tying all copies' branch edges to the
    hub. The hub edge is exposed as a GLUE port on the parent for the rest of
    the model to terminate (a prior, a goal, further gluing)."""

    port: str
    hub: str


@dataclass(frozen=True)
class Expose:
    """All remaining unbonded fragment ports become parent ports named
    `plate/<i>/port` (or `plate/<i>/<j>/port` for 2-D), roles preserved.

    Leftover ports are always exposed in phase 1 (a dangling internal edge is
    otherwise invalid); Expose() is the explicit, serializable marker of that
    intent."""


WiringRule = Chain | Broadcast | Expose


@dataclass(frozen=True)
class PlateSpec:
    """A replicated fragment: `dims` copies (1-D or 2-D in phase 1) of
    `fragment`, wired by `bonds`. Parameters are TIED by default (copies share
    the fragment's blob names); `per_copy_blobs` opts specific blobs out into
    per-copy indexed names."""

    fragment: "Template"
    dims: tuple[int, ...]
    bonds: tuple[WiringRule, ...] = ()
    per_copy_blobs: tuple[str, ...] = ()


@dataclass
class Template:
    """A CFFG fragment: the unit of serialization, composition, and compilation."""

    name: str
    variables: dict[str, VariableSpec] = field(default_factory=dict)
    factors: dict[str, FactorSpec] = field(default_factory=dict)
    constraints: list[ConstraintSpec] = field(default_factory=list)
    ports: dict[str, PortSpec] = field(default_factory=dict)
    blobs: dict[str, BlobRef] = field(default_factory=dict)
    plates: dict[str, PlateSpec] = field(default_factory=dict)
    learnables: dict[str, LearnableSpec] = field(default_factory=dict)
    schema_version: str = "1"

    def degree(self, variable: str) -> int:
        """Number of factor endpoints attached to an edge (FFG max is 2)."""
        return sum(
            1 for f in self.factors.values() for v in f.all_edges().values() if v == variable
        )

    def ports_of(self, variable: str) -> list[PortSpec]:
        return [p for p in self.ports.values() if p.variable == variable]
