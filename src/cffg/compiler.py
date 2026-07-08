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
    gauss_interior: list[tuple[int, np.ndarray]] = field(default_factory=list)
    # [(dim, [E, 2] slot pairs)] interior Gaussian edges for the entropy ledger
    gfe_groups: list[GfePlan] = field(default_factory=list)
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
                if e.interior and not e.dim and e.name not in self.gfe_x_edges
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
    next_slot = 0
    next_gslot = 0
    for vname in sorted(t.variables):
        ends = endpoints[vname]
        v = t.variables[vname]
        if v.dim:  # gaussian slots have their own numbering (own arena)
            s0, s1 = next_gslot, next_gslot + 1
            next_gslot += 2
        else:
            s0, s1 = next_slot, next_slot + 1
            next_slot += 2
        edges[vname] = EdgeInfo(vname, v.cardinality, s0, s1, len(ends) == 2, v.dim)
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
                    if t.factors[fname].type == "gaussian_linear" and role == "out":
                        gauss_data_ports[vname] = port.name
                    else:
                        problems.append(
                            f"port {port.name!r}: Gaussian DATA ports must clamp a "
                            f"gaussian_linear 'out' edge, not {fname!r}/{role!r}"
                        )
                else:
                    data_slots[port.name] = s1

    resolved = _resolve_arrays(t, arrays, root)
    member_of: dict[str, tuple[str, int]] = {}

    # GFE composites: P-substituted obs factor + facing goal factor across x.
    consumed: set[str] = set()
    gfe_x_edges: set[str] = set()
    gfe_buckets: dict[tuple[int, int], list[tuple[str, str]]] = {}  # cards -> (obs, goal)
    for fname, detail in sorted(psub.items()):
        f = t.factors[fname]
        if f.type != "matrix_cat" or detail.get("substituted_role") != "out":
            problems.append(
                f"p_substitution on {fname!r}: M3 supports matrix_cat nodes with "
                "substituted_role 'out' only"
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

    # Group remaining factors by (type, cardinality signature, mode); the
    # Gaussian family (and equality nodes on Gaussian edges) go to their own
    # arena's buckets, keyed by (type, dim signature, observed?).
    buckets: dict[tuple, list[str]] = {}
    gauss_buckets: dict[tuple, list[str]] = {}
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
        gauss_interior=gauss_interior,
        gfe_groups=gfe_groups,
        gfe_x_edges=frozenset(gfe_x_edges),
        delta_pairs=delta_pairs,
        member_of=member_of,
    )


def _resolve_arrays(
    t: Template, arrays: dict[str, np.ndarray] | None, root: str | Path | None
) -> dict[str, np.ndarray]:
    needed = {b for f in t.factors.values() for b in f.param_blobs.values()}
    out: dict[str, np.ndarray] = {}
    for name in sorted(needed):
        if arrays is not None and name in arrays:
            out[name] = np.asarray(arrays[name], dtype=np.float64)
            continue
        if root is None:
            raise CompileError(f"blob {name!r}: no array override and no root directory")
        path = Path(root) / t.blobs[name].path
        if not path.exists():
            raise CompileError(f"blob {name!r}: file not found: {path}")
        out[name] = _load_blob(path, name)
    return out


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
