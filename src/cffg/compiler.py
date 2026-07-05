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

from cffg.core.ir import FACTOR_TYPES, PortRole, Template
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


@dataclass(frozen=True)
class GroupPlan:
    """All factors of one (type, cardinality-signature), executed together."""

    name: str
    type: str
    roles: tuple[str, ...]
    cards: tuple[int, ...]  # cardinality per role
    members: tuple[str, ...]  # factor names, defines the stacking order
    gather: np.ndarray  # [n_members, n_roles] slot to read (opposite side)
    scatter: np.ndarray  # [n_members, n_roles] slot to write (own side)


@dataclass
class Arena:
    template_name: str
    k_max: int
    n_slots: int
    edges: dict[str, EdgeInfo]
    groups: list[GroupPlan]
    data_slots: dict[str, int]  # DATA port name -> external slot
    log_params: dict[str, np.ndarray] = field(default_factory=dict)  # group -> [M, ...]

    @property
    def interior_slot_pairs(self) -> np.ndarray:
        return np.array(
            [[e.slot0, e.slot1] for e in self.edges.values() if e.interior], dtype=np.int64
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

    for f in t.factors.values():
        if f.param_edges:
            problems.append(
                f"factor {f.name!r} has parameter edges {sorted(f.param_edges.values())}: "
                "learned parameters need Dirichlet VMP (M3)"
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
    next_slot = 0
    for vname in sorted(t.variables):
        ends = endpoints[vname]
        card = t.variables[vname].cardinality
        s0, s1 = next_slot, next_slot + 1
        next_slot += 2
        edges[vname] = EdgeInfo(vname, card, s0, s1, interior=len(ends) == 2)
        slot_of[ends[0]] = s0
        if len(ends) == 2:
            slot_of[ends[1]] = s1
            opposite[ends[0]], opposite[ends[1]] = s1, s0
        else:  # dangling: outer side s1 is external (clamp or uniform)
            opposite[ends[0]] = s1
            for port in t.ports_of(vname):
                if port.role is PortRole.DATA:
                    data_slots[port.name] = s1

    # Group factors by (type, cardinality signature).
    buckets: dict[tuple, list[str]] = {}
    for fname in sorted(t.factors):
        f = t.factors[fname]
        roles = FACTOR_TYPES[f.type].edge_roles
        cards = tuple(t.variables[f.edges[r]].cardinality for r in roles)
        buckets.setdefault((f.type, cards), []).append(fname)

    groups: list[GroupPlan] = []
    log_params: dict[str, np.ndarray] = {}
    resolved = _resolve_arrays(t, arrays, root)
    for (ftype, cards), members in sorted(buckets.items()):
        roles = FACTOR_TYPES[ftype].edge_roles
        gname = f"{ftype}:{'x'.join(map(str, cards))}"
        gather = np.array(
            [[opposite[(m, r)] for r in roles] for m in members], dtype=np.int64
        )
        scatter = np.array(
            [[slot_of[(m, r)] for r in roles] for m in members], dtype=np.int64
        )
        groups.append(GroupPlan(gname, ftype, roles, cards, tuple(members), gather, scatter))
        stacked = _stack_params(t, ftype, cards, members, resolved)
        if stacked is not None:
            log_params[gname] = stacked

    k_max = max(e.card for e in edges.values())
    return Arena(
        template_name=t.name,
        k_max=k_max,
        n_slots=next_slot,
        edges=edges,
        groups=groups,
        data_slots=data_slots,
        log_params=log_params,
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

    def log_of(arr: np.ndarray) -> np.ndarray:
        return np.where(arr > 0.0, np.log(np.maximum(arr, 1e-300)), NEG)

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
