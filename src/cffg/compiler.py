"""Template → executable arena.

Grounds a validated template into a flat message arena (PGMax-inspired):
messages live in one [n_slots, k_max] log-space array — each FFG edge owns two
slots, one per side, holding the message flowing *into* the edge from that
side. Factors are grouped by (type, cardinality signature) so each group
updates as a single vectorized einsum over its members, driven by numpy
gather/scatter index tables that jit treats as constants.

Parameters are *inputs* to the compiled step, not baked constants: instances
sharing a structure hash can be vmapped with different parameters (the colony
contract, docs/ARCHITECTURE.md §5b).

Port policy at compile time:
- DATA ports: allowed; the edge's outer slot is a runtime clamp (uniform when
  unbound = missing observation).
- GLUE ports: allowed; an unbonded glue edge is an FFG half-edge, which is
  implicit marginalization (Loeliger 2004) — the outer slot stays uniform.
- PRIOR / GOAL ports: rejected; they declare that a factor is *missing*, so
  compiling would silently change the model. Compose the prior (M2) or bind
  the goal machinery (M3) first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cffg import ef
from cffg.core.ir import FACTOR_TYPES, ConstraintKind, PortRole, Template
from cffg.core.validate import validate

NEG = -1.0e30  # log-space "zero"; finite so NEG - NEG == 0 (no inf-inf NaNs)


class CompileError(ValueError):
    pass


@dataclass(frozen=True)
class EdgeInfo:
    name: str
    card: int
    slot0: int  # side-0 message: into the edge from its first endpoint
    slot1: int  # side-1 message: second endpoint, or the external side
    interior: bool  # both sides are factors (degree 2)
    dim: int = 0  # Gaussian dimension; 0 = discrete. Gaussian slots index the
    # (eta, Lambda) arena, discrete slots the log-probability arena — the two
    # numberings are independent (cff-4e5z).
    ef: str = ""  # EF family name; slots then index the natural-parameter
    # arena ef_eta [n_efslots, 2], a third independent numbering (cff-8s6a).


@dataclass(frozen=True)
class GroupPlan:
    """All factors of one (type, cardinality-signature, mode), run together."""

    name: str
    type: str
    roles: tuple[str, ...]
    cards: tuple[int, ...]  # cardinality per role
    members: tuple[str, ...]  # factor names, defines the stacking order
    gather: np.ndarray  # [n_members, n_roles] slot to read (opposite side)
    scatter: np.ndarray  # [n_members, n_roles] slot to write (own side)
    mode: str = "bp"  # "bp" (sum-product) | "vmp" (mean-field bead)
    gather_both: np.ndarray | None = None  # [M, R, 2] both slots (vmp: marginals)


@dataclass(frozen=True)
class GaussGroupPlan:
    """All Gaussian-family factors of one (type, dim-signature), run together.

    Types: "gaussian" (prior), "gaussian_linear", "gaussian_linear_obs" (a
    gaussian_linear whose out edge is a DATA port — compiled to *conditioning*
    on the runtime observation, never a huge-precision clamp, so the
    differential-entropy ledger stays finite; see cffg.gauss), and "equality"
    (family-polymorphic, routed here when its edges are Gaussian)."""

    name: str
    type: str
    roles: tuple[str, ...]
    dims: tuple[int, ...]  # dimension per role
    members: tuple[str, ...]
    gather: np.ndarray  # [M, R] gaussian slot to read (opposite side)
    scatter: np.ndarray  # [M, R] gaussian slot to write (own side)
    obs_rows: np.ndarray | None = None  # [M] rows into the g_obs array


@dataclass(frozen=True)
class MixGroupPlan:
    """All hybrid Gaussian-mixture factors of one (type, dim-signature,
    selector cardinality), run together (cff-pksr). A hybrid node straddles
    two arenas: its out(/in) edges live in the Gaussian (eta, Lambda) arena
    (g_gather/g_scatter, read-opposite/write-own like GaussGroupPlan), its
    selector edge in the discrete log-probability arena (s_gather/s_scatter).

    Types: "gaussian_mixture" (roles out, selector; dims (do,)) and
    "gaussian_mixture_linear" (roles out, in, selector; dims (do, di)).
    Parameters are per-regime stacks keyed under f"{name}:W" [M, K, do, do]
    plus f"{name}:m" (mixture) or f"{name}:A"/f"{name}:b" (linear)."""

    name: str
    type: str
    dims: tuple[int, ...]  # (do,) or (do, di)
    k_sel: int  # selector cardinality K (the per-regime stack axis)
    members: tuple[str, ...]
    g_gather: np.ndarray  # [M, R_g] gaussian slot to read (opposite side)
    g_scatter: np.ndarray  # [M, R_g] gaussian slot to write (own side)
    s_gather: np.ndarray  # [M] selector discrete slot to read (opposite side)
    s_scatter: np.ndarray  # [M] selector discrete slot to write (own side)


@dataclass(frozen=True)
class PrecGroupPlan:
    """All gaussian_prec factors, run together (cff-7c5o). A scalar Gaussian
    node f(x; m, τ) = N(x; m, τ⁻¹) whose precision lives on an ef:gamma edge —
    the first factor straddling the Gaussian (eta, Lambda) arena (its `out`
    edge) and the EF natural-parameter arena (its `prec` edge). Both VMP
    messages read *marginals* (both slots of each edge, like the discrete vmp
    groups), so the gather tables carry both slots. Parameters: the scalar
    means stacked [M, 1] under f"{name}:m"."""

    name: str
    members: tuple[str, ...]
    x_gather_both: np.ndarray  # [M, 2] both Gaussian slots of the out edge
    tau_gather_both: np.ndarray  # [M, 2] both EF slots of the prec edge
    x_scatter: np.ndarray  # [M] own Gaussian slot on the out edge
    tau_scatter: np.ndarray  # [M] own EF slot on the prec edge


@dataclass(frozen=True)
class EfGroupPlan:
    """All EF-family factors of one (type, family), run together (cff-8s6a).

    Types: "gamma"/"beta" (prior leaves sending the constant natural
    parameters stacked in params["<name>:eta"]), "poisson_obs"/
    "bernoulli_obs" (conditioned likelihoods: the "out" edge is a DATA port,
    the message is a constant conjugate increment to the parameter edge, and
    a missing observation drops the node — the Gaussian conditioning
    convention), and "equality" (family-polymorphic, routed here when its
    edges are EF: sum-minus-self on natural parameters)."""

    name: str
    type: str
    family: str  # message family of the edges it reads/writes
    roles: tuple[str, ...]  # message-carrying roles only (obs "out" is data)
    members: tuple[str, ...]
    gather: np.ndarray  # [M, R] ef slot to read (opposite side)
    scatter: np.ndarray  # [M, R] ef slot to write (own side)
    obs_rows: np.ndarray | None = None  # [M] rows into the ef_obs array


@dataclass(frozen=True)
class GfeTmPlan:
    """A P-substituted transition_mixture node (cff-fk2u): the dynamics-
    curiosity composite. Solves the Corollary-1 fixed point over the joint
    (z, u) simplex with the downstream state message in the goal prior's
    role; `learned=True` (detail parameter_uncertainty=dirichlet) reads
    per-action Dirichlet counts α [U, K', K] and adds the count-novelty
    channel. FE contribution is a GFE term with optimistic-bound semantics
    (LIMITATIONS S1c)."""

    name: str
    cards: tuple[int, int, int]  # (K_out, K_in, U)
    members: tuple[str, ...]
    gather: np.ndarray  # [M, 3] incoming slots: (out=μ_down, in=μ_in, sel=μ_e)
    scatter: np.ndarray  # [M, 3] outgoing slots, same role order
    learned: bool = False


@dataclass(frozen=True)
class GfePlan:
    """A P-substituted goal-observation composite (Part II §4): matrix_cat
    observation node facing a categorical goal node across the x edge. The
    composite owns both factors' updates and their joint FE term.

    `learned=True` (P-bead detail parameter_uncertainty=dirichlet): the matrix
    parameter is a Dirichlet *count* array α, not probabilities. Messages then
    use Ā = E[A] for probability terms and the closed-form coupled term
    E[A log A] for the pseudo-entropy — the parameter-novelty channel."""

    name: str
    cards: tuple[int, int]  # (M_obs, K_state)
    members: tuple[str, ...]  # obs-factor names
    gather_z: np.ndarray  # [M] incoming state-message slot (μ_D)
    scatter_z: np.ndarray  # [M] outgoing backward-message slot (μ 2)
    scatter_x: np.ndarray  # [M] obs-side x slot (posterior predictive q_x)
    learned: bool = False


@dataclass
class Arena:
    template_name: str
    k_max: int
    n_slots: int
    edges: dict[str, EdgeInfo]
    groups: list[GroupPlan]
    data_slots: dict[str, int]  # DATA port name -> external slot
    log_params: dict[str, np.ndarray] = field(default_factory=dict)  # group -> [M, ...]
    # Gaussian arena (cff-4e5z): messages are (eta [n_gslots, d_max],
    # Lambda [n_gslots, d_max, d_max]) information-form pairs, zero-padded
    # past each edge's dim; vacuous = zeros, so init and half-edges are free.
    d_max: int = 1
    n_gslots: int = 0
    gauss_groups: list[GaussGroupPlan] = field(default_factory=list)
    gauss_data: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Gaussian DATA port -> (row in g_obs, dim of the observed edge)
    n_obs_rows: int = 0
    mix_groups: list[MixGroupPlan] = field(default_factory=list)
    # Hybrid Gaussian-mixture groups (cff-pksr): each writes both the Gaussian
    # arena (out/in edges) and the discrete arena (selector edge).
    prec_groups: list["PrecGroupPlan"] = field(default_factory=list)
    # gaussian_prec groups (cff-7c5o): each writes both the Gaussian arena
    # (out edge) and the EF arena (prec edge).
    gauss_interior: list[tuple[int, np.ndarray]] = field(default_factory=list)
    # [(dim, [E, 2] slot pairs)] interior Gaussian edges for the entropy ledger
    # EF arena (cff-8s6a): messages are natural parameters ef_eta
    # [n_efslots, 2], zero-padded past each family's n_params; vacuous =
    # zeros, so init and half-edges are free (η = 0 is the empty product).
    n_efslots: int = 0
    ef_groups: list[EfGroupPlan] = field(default_factory=list)
    ef_data: dict[str, tuple[int, str]] = field(default_factory=dict)
    # EF DATA port -> (row in the ef_obs/ef_mask arrays, observation family)
    n_ef_obs: int = 0
    ef_interior: list[tuple[str, np.ndarray]] = field(default_factory=list)
    # [(family, [E, 2] slot pairs)] interior EF edges for the entropy ledger
    gfe_groups: list[GfePlan] = field(default_factory=list)
    gfe_tm_groups: list[GfeTmPlan] = field(default_factory=list)
    gfe_x_edges: frozenset[str] = frozenset()  # entropy absorbed into composite term
    delta_pairs: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )  # [n_delta_edges, 2] slot pairs projected to a pointmass each sweep
    member_of: dict[str, tuple[str, int]] = field(default_factory=dict)
    # factor name -> (params key, member index): lets the runtime swap ONE
    # member's parameters (same-cardinality factors share a stacked group;
    # a whole-group write would clobber siblings).

    @property
    def interior_slot_pairs(self) -> np.ndarray:
        return np.array(
            [
                [e.slot0, e.slot1]
                for e in self.edges.values()
                if e.interior and not e.dim and not e.ef and e.name not in self.gfe_x_edges
            ],
            dtype=np.int64,
        ).reshape(-1, 2)


def compile_template(
    t: Template,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    root: str | Path | None = None,
) -> Arena:
    """Ground `t` into an Arena. Blob data comes from `arrays` (name -> array,
    wins) or is loaded from `root`/blob.path."""
    validate(t)
    problems: list[str] = []

    mean_field = {
        c.target for c in t.constraints if c.kind is ConstraintKind.MEAN_FIELD
    }
    psub = {
        c.target: c.detail_dict()
        for c in t.constraints
        if c.kind is ConstraintKind.P_SUBSTITUTION
    }

    for f in t.factors.values():
        if f.param_edges:
            problems.append(
                f"factor {f.name!r} has parameter edges {sorted(f.param_edges.values())}: "
                "learned parameters are separate tickets (cff-5iuz, cff-gjjk)"
            )
        if f.name in mean_field and f.type == "equality":
            problems.append(
                f"factor {f.name!r}: mean-field bead on an equality node is ill-defined "
                "(geometric expectation of a hard constraint)"
            )
        if f.name in mean_field and FACTOR_TYPES[f.type].family == "gaussian":
            problems.append(
                f"factor {f.name!r}: mean-field bead on a Gaussian node is Gaussian VMP "
                "(ticket cff-7c5o)"
            )
        if f.name in psub and FACTOR_TYPES[f.type].family == "gaussian":
            problems.append(
                f"factor {f.name!r}: P-substitution on Gaussian nodes is open research "
                "(ticket cff-dfpi)"
            )
        if FACTOR_TYPES[f.type].family == "ef" and f.name in (mean_field | set(psub)):
            problems.append(
                f"factor {f.name!r}: beads on EF nodes are undefined in M-scope — "
                "conjugate BP is already exact on EF subgraphs (cff-8s6a)"
            )
        if FACTOR_TYPES[f.type].family == "hybrid" and f.name in (mean_field | set(psub)):
            problems.append(
                f"factor {f.name!r}: hybrid beads are undefined; collapse rules are "
                "already assumed-density (cff-pksr)"
            )
        if f.type in ("poisson_obs", "bernoulli_obs"):
            out_var = f.edges["out"]
            out_ports = [p for p in t.ports_of(out_var) if p.role is PortRole.DATA]
            if t.degree(out_var) == 2 or not out_ports:
                problems.append(
                    f"factor {f.name!r}: {f.type} is a conditioned likelihood — its 'out' "
                    "edge must be a DATA port (unobserved predictives leave the family)"
                )
    for p in t.ports.values():
        if p.role in (PortRole.PRIOR, PortRole.GOAL):
            problems.append(
                f"port {p.name!r} ({p.role.value}) is unterminated: compose the missing "
                "factor before compiling"
            )
    if problems:
        raise CompileError(f"cannot compile {t.name!r}:\n  - " + "\n  - ".join(problems))

    # Edge table: endpoints in deterministic order; external side is side 1.
    endpoints: dict[str, list[tuple[str, str]]] = {v: [] for v in t.variables}
    for f in sorted(t.factors.values(), key=lambda f: f.name):
        for role, var in sorted(f.edges.items()):
            endpoints[var].append((f.name, role))

    edges: dict[str, EdgeInfo] = {}
    slot_of: dict[tuple[str, str], int] = {}  # (factor, role) -> its own slot
    opposite: dict[tuple[str, str], int] = {}  # (factor, role) -> slot it reads
    data_slots: dict[str, int] = {}
    gauss_data_ports: dict[str, str] = {}  # variable -> DATA port name (gaussian)
    ef_data_ports: dict[str, str] = {}  # variable -> DATA port name (ef obs)
    next_slot = 0
    next_gslot = 0
    next_efslot = 0
    for vname in sorted(t.variables):
        ends = endpoints[vname]
        v = t.variables[vname]
        if v.dim:  # gaussian slots have their own numbering (own arena)
            s0, s1 = next_gslot, next_gslot + 1
            next_gslot += 2
        elif v.ef:  # ef slots likewise (third arena)
            s0, s1 = next_efslot, next_efslot + 1
            next_efslot += 2
        else:
            s0, s1 = next_slot, next_slot + 1
            next_slot += 2
        edges[vname] = EdgeInfo(vname, v.cardinality, s0, s1, len(ends) == 2, v.dim, v.ef)
        slot_of[ends[0]] = s0
        if len(ends) == 2:
            slot_of[ends[1]] = s1
            opposite[ends[0]], opposite[ends[1]] = s1, s0
        else:  # dangling: outer side s1 is external (clamp or uniform/vacuous)
            opposite[ends[0]] = s1
            for port in t.ports_of(vname):
                if port.role is not PortRole.DATA:
                    continue
                if v.dim:
                    # Gaussian observations compile to conditioning: legal only
                    # where a conditioned rule exists (gaussian_linear "out").
                    fname, role = ends[0]
                    ftype_here = t.factors[fname].type
                    if ftype_here == "gaussian_linear" and role == "out":
                        gauss_data_ports[vname] = port.name
                    elif ftype_here in ("gaussian_mixture", "gaussian_mixture_linear"):
                        problems.append(
                            f"port {port.name!r}: observed mixture outputs are not "
                            f"supported yet (Gaussian DATA port on {fname!r}/{role!r})"
                        )
                    elif ftype_here == "gaussian_prec":
                        problems.append(
                            f"port {port.name!r}: observed gaussian_prec outputs: bind "
                            "observations via a gaussian_linear emission or an equality branch"
                        )
                    else:
                        problems.append(
                            f"port {port.name!r}: Gaussian DATA ports must clamp a "
                            f"gaussian_linear 'out' edge, not {fname!r}/{role!r}"
                        )
                elif v.ef:
                    # EF observations compile to conditioning too: legal only
                    # on a conjugate observation factor's "out" edge.
                    fname, role = ends[0]
                    if t.factors[fname].type in ("poisson_obs", "bernoulli_obs") and role == "out":
                        ef_data_ports[vname] = port.name
                    else:
                        problems.append(
                            f"port {port.name!r}: EF DATA ports must clamp a poisson_obs/"
                            f"bernoulli_obs 'out' edge, not {fname!r}/{role!r}"
                        )
                else:
                    data_slots[port.name] = s1

    resolved = _resolve_arrays(t, arrays, root)
    member_of: dict[str, tuple[str, int]] = {}

    # GFE composites: P-substituted obs factor + facing goal factor across x.
    consumed: set[str] = set()
    gfe_x_edges: set[str] = set()
    gfe_buckets: dict[tuple[int, int], list[tuple[str, str]]] = {}  # cards -> (obs, goal)
    tm_buckets: dict[tuple, list[str]] = {}
    for fname, detail in sorted(psub.items()):
        f = t.factors[fname]
        if f.type == "transition_mixture" and detail.get("substituted_role") == "out":
            cards3 = tuple(
                t.variables[f.edges[r]].cardinality for r in ("out", "in", "selector")
            )
            learned = detail.get("parameter_uncertainty") == "dirichlet"
            tm_buckets.setdefault((cards3, learned), []).append(fname)
            consumed.add(fname)
            continue
        if f.type != "matrix_cat" or detail.get("substituted_role") != "out":
            problems.append(
                f"p_substitution on {fname!r}: supported on matrix_cat and "
                "transition_mixture nodes with substituted_role 'out' only"
            )
            continue
        x_var = f.edges["out"]
        facing = [fn for fn, _ in endpoints[x_var] if fn != fname]
        goal = t.factors[facing[0]] if facing else None
        if goal is None or goal.type != "categorical" or "statistic" not in goal.param_blobs:
            problems.append(
                f"p_substitution on {fname!r}: edge {x_var!r} must face a categorical "
                "goal factor with a fixed statistic blob (learned goals: cff-gjjk)"
            )
            continue
        ko, ki = (t.variables[f.edges[r]].cardinality for r in ("out", "in"))
        learned = detail.get("parameter_uncertainty") == "dirichlet"
        gfe_buckets.setdefault((ko, ki, learned), []).append((fname, goal.name))
        consumed.update((fname, goal.name))
        gfe_x_edges.add(x_var)
    if problems:
        raise CompileError(f"cannot compile {t.name!r}:\n  - " + "\n  - ".join(problems))

    gfe_groups: list[GfePlan] = []
    log_params: dict[str, np.ndarray] = {}
    for (ko, ki, learned), pairs in sorted(gfe_buckets.items()):
        gname = f"gfe{'L' if learned else ''}:{ko}x{ki}"
        obs_names = tuple(o for o, _ in pairs)
        gfe_groups.append(
            GfePlan(
                name=gname,
                cards=(ko, ki),
                members=obs_names,
                gather_z=np.array([opposite[(o, "in")] for o, _ in pairs], dtype=np.int64),
                scatter_z=np.array([slot_of[(o, "in")] for o, _ in pairs], dtype=np.int64),
                scatter_x=np.array([slot_of[(o, "out")] for o, _ in pairs], dtype=np.int64),
                learned=learned,
            )
        )
        la, lc = [], []
        for obs_name, goal_name in pairs:
            a = resolved[next(iter(t.factors[obs_name].param_blobs.values()))]
            c = resolved[t.factors[goal_name].param_blobs["statistic"]]
            _expect(obs_name, a, (ko, ki))
            if learned:
                if not np.all(a > 0):
                    raise CompileError(f"factor {obs_name!r}: Dirichlet counts must be > 0")
                la.append(a)  # raw counts α, not probabilities
            else:
                _stochastic(obs_name, a)
                la.append(_log_of(a))
            _expect(goal_name, c, (ko,))
            _stochastic(goal_name, c[:, None])
            lc.append(_log_of(c))
        log_params[f"{gname}:counts" if learned else f"{gname}:A"] = np.stack(la)
        log_params[f"{gname}:c"] = np.stack(lc)
        for i, (obs_name, goal_name) in enumerate(pairs):
            member_of[obs_name] = (f"{gname}:counts" if learned else f"{gname}:A", i)
            member_of[goal_name] = (f"{gname}:c", i)

    gfe_tm_groups: list[GfeTmPlan] = []
    for ((ko, ki, ku), learned), members in sorted(tm_buckets.items()):
        gname = f"gfeTM{'L' if learned else ''}:{ko}x{ki}x{ku}"
        gather = np.array(
            [[opposite[(m, r)] for r in ("out", "in", "selector")] for m in members],
            dtype=np.int64,
        )
        scatter = np.array(
            [[slot_of[(m, r)] for r in ("out", "in", "selector")] for m in members],
            dtype=np.int64,
        )
        stacked = []
        for m in members:
            arr = resolved[t.factors[m].param_blobs["matrices"]]
            _expect(m, arr, (ku, ko, ki))
            if learned:
                if not np.all(arr > 0):
                    raise CompileError(f"factor {m!r}: Dirichlet counts must be > 0")
                stacked.append(arr)  # raw counts α [U, K', K]
            else:
                for u in range(ku):
                    _stochastic(f"{m}[u={u}]", arr[u])
                stacked.append(_log_of(arr))
        log_params[f"{gname}:counts" if learned else f"{gname}:B"] = np.stack(stacked)
        gfe_tm_groups.append(
            GfeTmPlan(gname, (ko, ki, ku), tuple(members), gather, scatter, learned)
        )

    # Group remaining factors by (type, cardinality signature, mode); the
    # Gaussian family (and equality nodes on Gaussian edges) go to their own
    # arena's buckets, keyed by (type, dim signature, observed?); the EF
    # family (and equality nodes on EF edges) to theirs, keyed by (type,
    # message family).
    buckets: dict[tuple, list[str]] = {}
    gauss_buckets: dict[tuple, list[str]] = {}
    ef_buckets: dict[tuple[str, str], list[str]] = {}
    mix_buckets: dict[tuple, list[str]] = {}
    prec_members: list[str] = []
    for fname in sorted(t.factors):
        if fname in consumed:
            continue
        f = t.factors[fname]
        ft = FACTOR_TYPES[f.type]
        roles = ft.edge_roles
        gaussian = ft.family == "gaussian" or (
            ft.family == "any" and t.variables[f.edges[roles[0]]].dim > 0
        )
        if gaussian:
            dims = tuple(t.variables[f.edges[r]].dim for r in roles)
            observed = f.type == "gaussian_linear" and f.edges["out"] in gauss_data_ports
            gtype = "gaussian_linear_obs" if observed else f.type
            gauss_buckets.setdefault((gtype, dims), []).append(fname)
            continue
        if f.type == "gaussian_prec":  # mixed Gaussian/EF node: its own arena pair
            d_out = t.variables[f.edges["out"]].dim
            if d_out != 1:
                raise CompileError(
                    f"factor {fname!r}: gaussian_prec is scalar (dim-1 out edge); "
                    "vector precision-mean priors are cff-7c5o follow-up work"
                )
            prec_members.append(fname)
            continue
        if ft.family == "hybrid":  # Gaussian-mixture nodes: their own arena pair
            dims = tuple(t.variables[f.edges[r]].dim for r in roles if r != "selector")
            k_sel = t.variables[f.edges["selector"]].cardinality
            mix_buckets.setdefault((f.type, dims, k_sel), []).append(fname)
            continue
        ef_family = ft.family == "ef" or (
            ft.family == "any" and bool(t.variables[f.edges[roles[0]]].ef)
        )
        if ef_family:
            if f.type in ("poisson_obs", "bernoulli_obs"):
                family = {"poisson_obs": "gamma", "bernoulli_obs": "beta"}[f.type]
            else:  # gamma/beta priors, or equality routed by its edges' family
                family = f.type if f.type in ef.FAMILIES else t.variables[f.edges[roles[0]]].ef
                if family not in ef.FAMILIES:
                    raise CompileError(
                        f"factor {fname!r}: no messages flow on observation-family "
                        f"({family!r}) edges — they are conditioned DATA ports only"
                    )
            ef_buckets.setdefault((f.type, family), []).append(fname)
            continue
        cards = tuple(t.variables[f.edges[r]].cardinality for r in roles)
        mode = "vmp" if fname in mean_field else "bp"
        buckets.setdefault((f.type, cards, mode), []).append(fname)

    gauss_groups: list[GaussGroupPlan] = []
    gauss_data: dict[str, int] = {}
    n_obs_rows = 0
    for (gtype, dims), members in sorted(gauss_buckets.items()):
        base = "gaussian_linear" if gtype == "gaussian_linear_obs" else gtype
        roles = FACTOR_TYPES[base].edge_roles
        gname = f"gauss:{gtype}:{'x'.join(map(str, dims))}"
        gather = np.array([[opposite[(m, r)] for r in roles] for m in members], dtype=np.int64)
        scatter = np.array([[slot_of[(m, r)] for r in roles] for m in members], dtype=np.int64)
        obs_rows = None
        if gtype == "gaussian_linear_obs":
            rows = []
            for m in members:
                port = gauss_data_ports[t.factors[m].edges["out"]]
                gauss_data[port] = (n_obs_rows, dims[0])
                rows.append(n_obs_rows)
                n_obs_rows += 1
            obs_rows = np.array(rows, dtype=np.int64)
        gauss_groups.append(
            GaussGroupPlan(gname, gtype, roles, dims, tuple(members), gather, scatter, obs_rows)
        )
        for key, arr in _stack_gauss_params(t, base, dims, members, resolved).items():
            log_params[f"{gname}:{key}"] = arr
        # member_of stays discrete-only: per-member Gaussian parameter updates
        # are the learning ticket's call (cff-7c5o).

    mix_groups: list[MixGroupPlan] = []
    for (mtype, dims, k_sel), members in sorted(mix_buckets.items()):
        gauss_roles = tuple(r for r in FACTOR_TYPES[mtype].edge_roles if r != "selector")
        gname = f"mix:{mtype}:{'x'.join(map(str, dims))}:{k_sel}"
        g_gather = np.array(
            [[opposite[(m, r)] for r in gauss_roles] for m in members], dtype=np.int64
        )
        g_scatter = np.array(
            [[slot_of[(m, r)] for r in gauss_roles] for m in members], dtype=np.int64
        )
        s_gather = np.array([opposite[(m, "selector")] for m in members], dtype=np.int64)
        s_scatter = np.array([slot_of[(m, "selector")] for m in members], dtype=np.int64)
        mix_groups.append(
            MixGroupPlan(
                gname, mtype, dims, k_sel, tuple(members),
                g_gather, g_scatter, s_gather, s_scatter,
            )
        )
        for key, arr in _stack_mix_params(t, mtype, dims, k_sel, members, resolved).items():
            log_params[f"{gname}:{key}"] = arr
        # member_of stays discrete-only, as with the Gaussian and EF families.

    prec_groups: list[PrecGroupPlan] = []
    if prec_members:
        members = sorted(prec_members)
        gname = "prec:gaussian_prec"
        x_scatter = np.array([slot_of[(m, "out")] for m in members], dtype=np.int64)
        x_gather_both = np.stack(
            [x_scatter, np.array([opposite[(m, "out")] for m in members], dtype=np.int64)],
            axis=-1,
        )
        tau_scatter = np.array([slot_of[(m, "prec")] for m in members], dtype=np.int64)
        tau_gather_both = np.stack(
            [tau_scatter, np.array([opposite[(m, "prec")] for m in members], dtype=np.int64)],
            axis=-1,
        )
        prec_groups.append(
            PrecGroupPlan(
                gname, tuple(members), x_gather_both, tau_gather_both, x_scatter, tau_scatter
            )
        )
        log_params[f"{gname}:m"] = _stack_prec_means(t, members, resolved)
        # member_of stays discrete-only, as with the Gaussian and EF families.

    ef_groups: list[EfGroupPlan] = []
    ef_data: dict[str, int] = {}
    n_ef_obs = 0
    _EF_ROLES = {
        "gamma": ("out",),
        "beta": ("out",),
        "poisson_obs": ("rate",),
        "bernoulli_obs": ("prob",),
        "equality": ("a", "b", "c"),
    }
    for (ftype, family), members in sorted(ef_buckets.items()):
        roles = _EF_ROLES[ftype]
        gname = f"ef:{ftype}:{family}"
        gather = np.array([[opposite[(m, r)] for r in roles] for m in members], dtype=np.int64)
        scatter = np.array([[slot_of[(m, r)] for r in roles] for m in members], dtype=np.int64)
        obs_rows = None
        if ftype in ("poisson_obs", "bernoulli_obs"):
            rows = []
            for m in members:
                out_var = t.factors[m].edges["out"]
                ef_data[ef_data_ports[out_var]] = (n_ef_obs, t.variables[out_var].ef)
                rows.append(n_ef_obs)
                n_ef_obs += 1
            obs_rows = np.array(rows, dtype=np.int64)
        elif ftype != "equality":  # gamma/beta prior leaves: stack constant η_p
            log_params[f"{gname}:eta"] = _stack_ef_priors(t, ftype, members, resolved)
        ef_groups.append(
            EfGroupPlan(gname, ftype, family, roles, tuple(members), gather, scatter, obs_rows)
        )
        # member_of stays discrete-only: per-member EF prior updates are the
        # conjugate-learning ticket's call (cff-x0sv), as with Gaussian.

    groups: list[GroupPlan] = []
    for (ftype, cards, mode), members in sorted(buckets.items()):
        roles = FACTOR_TYPES[ftype].edge_roles
        gname = f"{mode}:{ftype}:{'x'.join(map(str, cards))}"
        gather = np.array(
            [[opposite[(m, r)] for r in roles] for m in members], dtype=np.int64
        )
        scatter = np.array(
            [[slot_of[(m, r)] for r in roles] for m in members], dtype=np.int64
        )
        both = np.stack([scatter, gather], axis=-1)  # own & opposite = the edge pair
        groups.append(
            GroupPlan(gname, ftype, roles, cards, tuple(members), gather, scatter, mode, both)
        )
        for i, m in enumerate(members):
            member_of[m] = (gname, i)
        stacked = _stack_params(t, ftype, cards, members, resolved)
        if stacked is not None:
            log_params[gname] = stacked

    delta_edges = sorted(
        c.target for c in t.constraints if c.kind is ConstraintKind.DELTA
    )
    for v in delta_edges:
        if edges[v].dim:
            raise CompileError(
                f"delta constraint on Gaussian edge {v!r}: the MAP of a Gaussian "
                "marginal is its mean — read gaussian_marginals instead (hybrid "
                "pointmass forms: cff-pksr)"
            )
        if edges[v].ef:
            raise CompileError(
                f"delta constraint on EF edge {v!r}: read the mode from "
                "ef_marginals instead (hybrid pointmass forms: cff-pksr)"
            )
    delta_pairs = np.array(
        [[edges[v].slot0, edges[v].slot1] for v in delta_edges], dtype=np.int64
    ).reshape(-1, 2)

    # Interior Gaussian edges, grouped by dim, for the Bethe entropy ledger.
    by_dim: dict[int, list[list[int]]] = {}
    for e in edges.values():
        if e.dim and e.interior:
            by_dim.setdefault(e.dim, []).append([e.slot0, e.slot1])
    gauss_interior = [
        (d, np.array(pairs, dtype=np.int64)) for d, pairs in sorted(by_dim.items())
    ]

    # Interior EF edges, grouped by family, likewise. Observation-family
    # edges are always exterior DATA ports (enforced above), so every
    # interior EF edge has a message family with a closed-form entropy.
    by_fam: dict[str, list[list[int]]] = {}
    for e in edges.values():
        if e.ef and e.interior:
            by_fam.setdefault(e.ef, []).append([e.slot0, e.slot1])
    ef_interior = [
        (fam, np.array(pairs, dtype=np.int64)) for fam, pairs in sorted(by_fam.items())
    ]

    k_max = max((e.card for e in edges.values() if not e.dim), default=1)
    d_max = max((e.dim for e in edges.values() if e.dim), default=1)
    return Arena(
        template_name=t.name,
        k_max=k_max,
        n_slots=next_slot,
        edges=edges,
        groups=groups,
        data_slots=data_slots,
        log_params=log_params,
        d_max=d_max,
        n_gslots=next_gslot,
        gauss_groups=gauss_groups,
        gauss_data=gauss_data,
        n_obs_rows=n_obs_rows,
        mix_groups=mix_groups,
        prec_groups=prec_groups,
        gauss_interior=gauss_interior,
        n_efslots=next_efslot,
        ef_groups=ef_groups,
        ef_data=ef_data,
        n_ef_obs=n_ef_obs,
        ef_interior=ef_interior,
        gfe_groups=gfe_groups,
        gfe_tm_groups=gfe_tm_groups,
        gfe_x_edges=frozenset(gfe_x_edges),
        delta_pairs=delta_pairs,
        member_of=member_of,
    )


def _resolve_arrays(
    t: Template, arrays: dict[str, np.ndarray] | None, root: str | Path | None
) -> dict[str, np.ndarray]:
    """Resolution order per blob: explicit arrays > existing file > init spec.
    Files beat init specs so a continual-learning checkpoint written to the
    blob path overrides the warm start on reload (cff-wx28)."""
    needed = {b for f in t.factors.values() for b in f.param_blobs.values()}
    out: dict[str, np.ndarray] = {}
    for name in sorted(needed):
        if arrays is not None and name in arrays:
            out[name] = np.asarray(arrays[name], dtype=np.float64)
            continue
        path = Path(root) / t.blobs[name].path if root is not None else None
        if path is not None and path.exists():
            out[name] = _load_blob(path, name)
            continue
        if t.blobs[name].init:
            from cffg import inits

            shape = _blob_shape(t, name)
            if shape is None:
                raise CompileError(f"blob {name!r}: cannot infer shape for init spec")
            try:
                out[name] = np.asarray(
                    inits.from_spec(t.blobs[name].init, shape), dtype=np.float64
                )
            except ValueError as e:
                raise CompileError(f"blob {name!r}: init spec failed: {e}") from None
            continue
        raise CompileError(
            f"blob {name!r}: no array override, no file"
            + (f" at {path}" if path is not None else " (no root directory)")
            + ", and no init spec"
        )
    return out


def _blob_shape(t: Template, blob_name: str) -> tuple[int, ...] | None:
    """Expected array shape for a blob, inferred from the first factor role
    that references it (the compiler validates full agreement later)."""
    for f in t.factors.values():
        for role, b in f.param_blobs.items():
            if b != blob_name:
                continue
            card = lambda r: t.variables[f.edges[r]].cardinality  # noqa: E731
            dim = lambda r: t.variables[f.edges[r]].dim  # noqa: E731
            if f.type in ("categorical", "dirichlet"):
                return (card("out"),)
            if f.type == "matrix_cat":
                return (card("out"), card("in"))
            if f.type == "transition_mixture":
                return (card("selector"), card("out"), card("in"))
            if f.type == "gaussian":
                d = dim("out")
                return {"mean": (d,), "cov": (d, d)}[role]
            if f.type == "gaussian_linear":
                do, di = dim("out"), dim("in")
                return {"matrix": (do, di), "offset": (do,), "cov": (do, do)}[role]
            if f.type == "gaussian_mixture":
                d, k = dim("out"), card("selector")
                return {"mean": (k, d), "cov": (k, d, d)}[role]
            if f.type == "gaussian_mixture_linear":
                do, di, k = dim("out"), dim("in"), card("selector")
                return {"matrix": (k, do, di), "offset": (k, do), "cov": (k, do, do)}[role]
            if f.type in ("gamma", "beta"):
                return ()  # scalar prior parameters
            if f.type == "gaussian_prec":
                return (dim("out"),)  # scalar mean (dim-1 out edge)
    return None


def _load_blob(path: Path, name: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file

        tensors = load_file(path)
        key = name if name in tensors else next(iter(tensors)) if len(tensors) == 1 else None
        if key is None:
            raise CompileError(f"blob {name!r}: {path} holds {sorted(tensors)}, need {name!r}")
        return np.asarray(tensors[key], dtype=np.float64)
    raise CompileError(f"blob {name!r}: unsupported format {path.suffix!r}")


def _stack_params(
    t: Template,
    ftype: str,
    cards: tuple[int, ...],
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Stack per-member parameter arrays into [M, ...] log-space, with shape
    and column-stochasticity checks ("true FE" needs normalized conditionals)."""
    if ftype == "equality":
        return None
    log_of = _log_of

    stacked = []
    for m in members:
        f = t.factors[m]
        arr = resolved[next(iter(f.param_blobs.values()))]
        if ftype == "categorical":
            (k,) = cards
            _expect(m, arr, (k,))
            _stochastic(m, arr[:, None])  # one column: the whole vector sums to 1
            stacked.append(log_of(arr))
        elif ftype == "matrix_cat":
            ko, ki = cards
            _expect(m, arr, (ko, ki))
            _stochastic(m, arr)
            stacked.append(log_of(arr))
        elif ftype == "transition_mixture":
            ko, ki, ku = cards
            _expect(m, arr, (ku, ko, ki))  # blob is a [K_u] stack of B_u
            for u in range(ku):
                _stochastic(f"{m}[u={u}]", arr[u])
            stacked.append(log_of(arr.transpose(1, 2, 0)))  # -> [ko, ki, ku]
        else:
            raise CompileError(f"factor {m!r}: no BP rule for type {ftype!r} in M2")
    return np.stack(stacked)


def _stack_gauss_params(
    t: Template,
    ftype: str,
    dims: tuple[int, ...],
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-member Gaussian parameters, stacked [M, ...]. Covariances are
    validated SPD and stored as precisions W = V⁻¹ (the form every rule
    consumes); the compile-time Cholesky is also the SPD check."""
    if ftype == "equality":
        return {}
    out: dict[str, list[np.ndarray]] = {}
    for m in members:
        f = t.factors[m]
        blobs = {role: resolved[f.param_blobs[role]] for role in sorted(f.param_blobs)}
        cov = blobs["cov"]
        if ftype == "gaussian":
            (d,) = dims
            _expect(m, blobs["mean"], (d,))
            _expect(m, cov, (d, d))
            out.setdefault("m", []).append(blobs["mean"])
        else:  # gaussian_linear
            do, di = dims
            _expect(m, blobs["matrix"], (do, di))
            _expect(m, blobs["offset"], (do,))
            _expect(m, cov, (do, do))
            out.setdefault("A", []).append(blobs["matrix"])
            out.setdefault("b", []).append(blobs["offset"])
        if not np.allclose(cov, cov.T, atol=1e-12):
            raise CompileError(f"factor {m!r}: covariance not symmetric")
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            raise CompileError(f"factor {m!r}: covariance not positive definite") from None
        out.setdefault("W", []).append(np.linalg.inv(cov))
    return {k: np.stack(v) for k, v in out.items()}


def _stack_mix_params(
    t: Template,
    ftype: str,
    dims: tuple[int, ...],
    k_sel: int,
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-member hybrid parameters, stacked [M, K, ...] over the K regimes
    (cff-pksr). Blobs are per-regime stacks: mean [K, d]/cov [K, d, d] for
    gaussian_mixture, matrix [K, do, di]/offset [K, do]/cov [K, do, do] for
    gaussian_mixture_linear. Each cov[k] is validated symmetric + SPD (the
    Cholesky is the check) and stored as a precision W = V⁻¹, the form every
    mixg rule consumes."""
    out: dict[str, list[np.ndarray]] = {}
    for m in members:
        f = t.factors[m]
        blobs = {role: resolved[f.param_blobs[role]] for role in sorted(f.param_blobs)}
        cov = blobs["cov"]
        if ftype == "gaussian_mixture":
            (d,) = dims
            _expect(m, blobs["mean"], (k_sel, d))
            _expect(m, cov, (k_sel, d, d))
            out.setdefault("m", []).append(blobs["mean"])
        else:  # gaussian_mixture_linear
            do, di = dims
            _expect(m, blobs["matrix"], (k_sel, do, di))
            _expect(m, blobs["offset"], (k_sel, do))
            _expect(m, cov, (k_sel, do, do))
            out.setdefault("A", []).append(blobs["matrix"])
            out.setdefault("b", []).append(blobs["offset"])
        w_regimes = []
        for k in range(k_sel):
            ck = cov[k]
            if not np.allclose(ck, ck.T, atol=1e-12):
                raise CompileError(f"factor {m!r}: covariance not symmetric")
            try:
                np.linalg.cholesky(ck)
            except np.linalg.LinAlgError:
                raise CompileError(f"factor {m!r}: covariance not positive definite") from None
            w_regimes.append(np.linalg.inv(ck))
        out.setdefault("W", []).append(np.stack(w_regimes))
    return {k: np.stack(v) for k, v in out.items()}


def _stack_prec_means(
    t: Template, members: list[str], resolved: dict[str, np.ndarray]
) -> np.ndarray:
    """Stack per-member gaussian_prec means [M, 1]. The mean is the scalar m in
    N(x; m, τ⁻¹); the out edge is dim-1, so any size-1 blob is accepted and
    reshaped to a length-1 vector (the form gauss.prec_to_x consumes)."""
    means = []
    for m in members:
        arr = resolved[t.factors[m].param_blobs["mean"]]
        if arr.size != 1:
            raise CompileError(
                f"factor {m!r}: parameter 'mean' must be a scalar, got shape {arr.shape}"
            )
        means.append(arr.reshape(1))
    return np.stack(means)


def _stack_ef_priors(
    t: Template, ftype: str, members: list[str], resolved: dict[str, np.ndarray]
) -> np.ndarray:
    """Stack per-member prior natural parameters [M, 2]. Priors must be
    proper — an improper prior makes every downstream belief and ledger term
    undefined, so it is a compile error, not a runtime +inf."""
    fam = ef.FAMILIES[ftype]
    p_roles = ("shape", "rate") if ftype == "gamma" else ("a", "b")
    etas = []
    for m in members:
        vals = []
        for role in p_roles:
            arr = resolved[t.factors[m].param_blobs[role]]
            if arr.size != 1:
                raise CompileError(
                    f"factor {m!r}: parameter {role!r} must be a scalar, got shape {arr.shape}"
                )
            vals.append(float(arr.reshape(())))
        if not all(v > 0 for v in vals):
            raise CompileError(
                f"factor {m!r}: improper {ftype} prior {tuple(vals)}; parameters must be > 0"
            )
        etas.append(fam.to_natural(*vals))
    return np.stack(etas)


def _log_of(arr: np.ndarray) -> np.ndarray:
    return np.where(arr > 0.0, np.log(np.maximum(arr, 1e-300)), NEG)


def _expect(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if arr.shape != shape:
        raise CompileError(f"factor {name!r}: parameter shape {arr.shape} != expected {shape}")


def _stochastic(name: str, arr: np.ndarray) -> None:
    sums = arr.sum(axis=0)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise CompileError(
            f"factor {name!r}: conditional not column-stochastic (col sums {sums}); "
            "true-FE semantics require normalized factors"
        )
