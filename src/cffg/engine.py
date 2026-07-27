"""The inference VM: flooding message passing over a compiled arena.

State is pure data (an Equinox pytree of flat, shape-segmented message arrays
plus per-group log-parameters); the step is a pure function of it. Prefix
offsets and logical-slot tables from the Arena are captured as numpy constants
inside the jitted step, so instances sharing a structure batch under
`eqx.filter_vmap` (colony contract).

Free energy: `bfe` evaluates F = Σ_a F[q_a] + Σ_{interior edges} H[q_i]
exactly — node terms from the rule registry, plus the Bethe edge-entropy
correction (interior edges have degree 2, hence counting number d_i − 1 = 1).
On tree graphs at the BP fixed point, F = −log Z.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cffg import csp, ef, efrules, gauss, kernels, mixg
from cffg.compiler import NEG, Arena
from cffg.schedule import BatchStage, Schedule, _fingerprint, build_schedule
from cffg.rules import (
    BP_UPDATES,
    NODE_ENERGIES,
    VMP_ENERGIES,
    VMP_UPDATES,
    gfe_energy,
    gfe_tm_energy,
    gfe_tm_update,
    gfe_update,
)

_TINY = 1e-12


class State(eqx.Module):
    """Pure-data instance state. The Gaussian arrays are zero-size on
    discrete-only models — same pytree structure either way, so colonies mix
    freely. g_obs/g_mask are the *conditioned* Gaussian observations (see
    cffg.gauss: conditioning, not clamping) with mask 0 = missing."""

    messages: jax.Array  # [sum_slot K] log-space discrete segments
    params: dict[str, jax.Array]  # group name -> [M, ...] log-factors
    g_eta: jax.Array  # [sum_slot d] information-vector segments
    g_lam: jax.Array  # [sum_slot d*d] flattened precision blocks
    g_obs: jax.Array  # [sum_observation d] conditioned observation segments
    g_mask: jax.Array  # [n_obs_rows] 1.0 = bound, 0.0 = missing
    ef_eta: jax.Array  # [n_efslots, 2] EF natural parameters (cff-8s6a)
    ef_obs: jax.Array  # [n_ef_obs] conditioned counts/flips
    ef_mask: jax.Array  # [n_ef_obs] 1.0 = bound, 0.0 = missing


class RunStatus(str, Enum):
    """Mutually exclusive public termination outcomes for :meth:`Engine.run`."""

    CONVERGED = "converged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DIVERGED = "diverged"
    INFEASIBLE = "infeasible"


class InferenceError(RuntimeError):
    """Raised when a caller requires a converged state and did not get one."""

    def __init__(self, result: "RunResult"):
        super().__init__(
            f"inference {result.status.value} after {result.iterations} iterations "
            f"(residual={result.residual!r})"
        )
        self.result = result


@dataclass(frozen=True)
class RunResult:
    """Status-bearing result of an inference run.

    This intentionally is not tuple-compatible: migrating callers must name
    ``state`` and therefore cross the status-bearing API boundary explicitly.
    ``free_energies`` contains only sweeps on which the ledger is defined
    (improper continuous beliefs during propagation have no FE entry).
    """

    state: State
    status: RunStatus
    iterations: int
    message_deltas: tuple[float, ...]
    free_energies: tuple[float, ...]
    residual: float
    inner_status: RunStatus = RunStatus.CONVERGED
    inner_residual: float = 0.0

    @property
    def delta_trace(self) -> tuple[float, ...]:
        return self.message_deltas

    @property
    def fe_trace(self) -> tuple[float, ...]:
        return self.free_energies

    def require_converged(self) -> State:
        """Return the state only when the run proved convergence."""
        if self.status is not RunStatus.CONVERGED:
            raise InferenceError(self)
        return self.state

    def require_converged_with_trace(self) -> tuple[State, tuple[float, ...]]:
        """Explicit migration helper for the old ``state, deltas`` API."""
        return self.require_converged(), self.message_deltas


def _replace(state: State, **kw) -> State:
    return State(**{f: kw.get(f, getattr(state, f)) for f in (
        "messages", "params", "g_eta", "g_lam", "g_obs", "g_mask",
        "ef_eta", "ef_obs", "ef_mask",
    )})


def _disc_rows(
    arena: Arena,
    messages: jax.Array,
    slots: np.ndarray,
    cards: tuple[int, ...],
) -> jax.Array:
    """Gather logical discrete slots into one shape-local padded block.

    Padding exists only in this transient rule input and only to the maximum
    role cardinality of one homogeneous factor group. Persistent storage is
    exactly segmented, so an unrelated model-wide outlier cannot widen it.
    """
    width = max(cards)
    blocks = []
    for role, card in enumerate(cards):
        block = messages[arena.message_indices(slots[..., role], card)]
        blocks.append(
            jnp.pad(block, [(0, 0)] * (block.ndim - 1) + [(0, width - card)],
                    constant_values=NEG)
        )
    return jnp.stack(blocks, axis=-2)


def _disc_both(
    arena: Arena,
    messages: jax.Array,
    slots: np.ndarray,
    cards: tuple[int, ...],
) -> jax.Array:
    """Gather both logical slots per VMP edge as ``[M, R, 2, local_k]``."""
    width = max(cards)
    blocks = []
    for role, card in enumerate(cards):
        block = messages[arena.message_indices(slots[:, role, :], card)]
        blocks.append(jnp.pad(block, ((0, 0), (0, 0), (0, width - card)),
                              constant_values=NEG))
    return jnp.stack(blocks, axis=1)


def _set_disc_rows(
    arena: Arena,
    messages: jax.Array,
    slots: np.ndarray,
    cards: tuple[int, ...],
    rows: jax.Array,
) -> jax.Array:
    """Scatter a shape-local ``[..., R, local_k]`` rule output exactly."""
    out = messages
    for role, card in enumerate(cards):
        idx = arena.message_indices(slots[..., role], card)
        out = out.at[idx].set(rows[..., role, :card])
    return out


def _set_disc_block(
    arena: Arena,
    messages: jax.Array,
    slots: np.ndarray,
    card: int,
    rows: jax.Array,
) -> jax.Array:
    return messages.at[arena.message_indices(slots, card)].set(rows[..., :card])


def _gauss_rows(
    arena: Arena,
    eta: jax.Array,
    lam: jax.Array,
    slots: np.ndarray,
    dims: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Gather Gaussian segments into one shape-local padded rule block."""
    width = max(dims)
    eta_blocks, lam_blocks = [], []
    for role, dim in enumerate(dims):
        e = eta[arena.g_eta_indices(slots[..., role], dim)]
        l = lam[arena.g_lam_indices(slots[..., role], dim)]
        eta_blocks.append(jnp.pad(e, [(0, 0)] * (e.ndim - 1) + [(0, width - dim)]))
        lam_blocks.append(
            jnp.pad(l, [(0, 0)] * (l.ndim - 2) + [(0, width - dim), (0, width - dim)])
        )
    return jnp.stack(eta_blocks, axis=-2), jnp.stack(lam_blocks, axis=-3)


def _set_gauss_block(
    arena: Arena,
    eta: jax.Array,
    lam: jax.Array,
    slots: np.ndarray,
    dim: int,
    eta_rows: jax.Array,
    lam_rows: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return (
        eta.at[arena.g_eta_indices(slots, dim)].set(eta_rows[..., :dim]),
        lam.at[arena.g_lam_indices(slots, dim)].set(lam_rows[..., :dim, :dim]),
    )


def _obs_rows(arena: Arena, observations: jax.Array, rows: np.ndarray, dim: int) -> jax.Array:
    return observations[arena.g_obs_indices(rows, dim)]


class Engine:
    """Compiled, executable model. Construction is cheap; `step` is jitted.

    ``schedule`` selects the message-passing schedule (cff-d56d): "auto"
    builds the strongest schedule the topology admits — exact two-pass for
    acyclic BP over the discrete/Gaussian/EF arenas, colored sweeps for
    loopy pure BP, block-phased mixed scheduling when iterative structures
    (vmp/GFE/constraints/hybrid/prec) are present — and floods otherwise.
    "flood" forces flooding; "two_pass" requires the exact schedule and
    raises when the model is outside its contract. A prebuilt
    :class:`~cffg.schedule.Schedule` is validated against this arena's
    structure fingerprint and rejected on mismatch."""

    def __init__(self, arena: Arena, schedule: str | Schedule = "auto"):
        self.arena = arena
        self._step = eqx.filter_jit(_make_step(arena))
        self._bfe = eqx.filter_jit(_make_bfe(arena))
        if isinstance(schedule, Schedule):
            if schedule.kind != "flood":  # flooding is always safe as given
                # Recompute the fingerprint from the prebuilt schedule's own
                # content: a doctored stage list cannot ride in on a genuine
                # stored fingerprint, and a schedule built for another arena
                # cannot match this one's structure hash.
                expected = build_schedule(arena, "auto")
                if _fingerprint(arena, schedule.describe()) != expected.fingerprint:
                    raise ValueError(
                        "prebuilt schedule does not match this arena "
                        "(stale, doctored, or built for another model); "
                        "rebuild it with cffg.schedule.build_schedule"
                    )
            self.schedule = schedule
        else:
            self.schedule = build_schedule(arena, schedule)
        self._propagate = None
        self._prime = None
        self._iter_step = self._step
        self.scan_registerized: tuple[bool, ...] = ()
        kind = self.schedule.kind
        self._exact_unit = None
        if kind == "two_pass":
            raw = _make_propagate(arena, self.schedule.stages)
            self.scan_registerized = raw.scan_registerized
            self._propagate = eqx.filter_jit(raw)
            self._exact_unit = eqx.filter_jit(_make_exact_unit(arena, raw))
        elif kind in ("colored", "mixed"):
            island = _make_propagate(arena, self.schedule.stages)
            self.scan_registerized = island.scan_registerized
            if self.schedule.prime_stages:
                prime_raw = (
                    island
                    if self.schedule.prime_stages == self.schedule.stages
                    else _make_propagate(arena, self.schedule.prime_stages)
                )
                self._prime = eqx.filter_jit(prime_raw)
            if kind == "mixed":
                # Reviewer-ordered outer iteration (cff-d56d, 2026-07-27):
                # update the iterative phase synchronously, then exactly
                # re-equilibrate the BP islands, so the loop's residual/FE
                # is never evaluated on a stale island.
                phase_step = _make_step(arena, skip_scheduled=True)

                def iter_step(
                    messages, g_eta, g_lam, g_obs, g_mask,
                    ef_eta, ef_obs, ef_mask, params, damp=0.0, harden=True,
                ):
                    messages, g_eta, g_lam, ef_eta, ir, ic = phase_step(
                        messages, g_eta, g_lam, g_obs, g_mask,
                        ef_eta, ef_obs, ef_mask, params, damp, harden,
                    )
                    messages, g_eta, g_lam, ef_eta = island(
                        messages, g_eta, g_lam, ef_eta,
                        g_obs, g_mask, ef_obs, ef_mask, params,
                    )
                    return messages, g_eta, g_lam, ef_eta, ir, ic
            else:
                # Colored Gauss-Seidel sweep; damping blends the sweep's
                # result with its input (fixed points are preserved).
                def iter_step(
                    messages, g_eta, g_lam, g_obs, g_mask,
                    ef_eta, ef_obs, ef_mask, params, damp=0.0, harden=True,
                ):
                    m2, e2, l2, f2 = island(
                        messages, g_eta, g_lam, ef_eta,
                        g_obs, g_mask, ef_obs, ef_mask, params,
                    )
                    m2 = damp * messages + (1.0 - damp) * m2
                    e2 = damp * g_eta + (1.0 - damp) * e2
                    l2 = damp * g_lam + (1.0 - damp) * l2
                    f2 = damp * ef_eta + (1.0 - damp) * f2
                    zero = jnp.asarray(0.0, dtype=m2.dtype)
                    return m2, e2, l2, f2, zero, jnp.asarray(0)

                iter_step.__doc__ = "one colored sweep over the cyclic islands"
            self._iter_step = eqx.filter_jit(iter_step)
        # Discrete edge slot pairs bucketed by cardinality: the zero-support
        # check runs per bucket instead of per edge (Python-loop-free).
        by_card: dict[int, list[list[int]]] = {}
        for e in arena.edges.values():
            if not e.dim and not e.ef:
                by_card.setdefault(e.card, []).append([e.slot0, e.slot1])
        self._support_edge_groups = [
            (card, np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
            for card, pairs in sorted(by_card.items())
        ]

    # -- state construction ---------------------------------------------

    def init_state(self) -> State:
        a = self.arena
        return State(
            messages=self._uniform_messages(),
            params={name: jnp.asarray(p) for name, p in a.log_params.items()},
            g_eta=jnp.zeros(a.n_g_eta_values),  # zeros = vacuous
            g_lam=jnp.zeros(a.n_g_lam_values),
            g_obs=jnp.zeros(a.n_g_obs_values),
            g_mask=jnp.zeros(a.n_obs_rows),
            ef_eta=jnp.zeros((a.n_efslots, 2)),  # zeros = vacuous
            ef_obs=jnp.zeros(a.n_ef_obs),
            ef_mask=jnp.zeros(a.n_ef_obs),
        )

    def _uniform_messages(self) -> jax.Array:
        rows = np.empty(self.arena.n_message_values)
        for e in self.arena.edges.values():
            if e.dim or e.ef:
                continue  # gaussian/EF slots live in their own arenas
            rows[self.arena.message_indices(e.slot0, e.card)] = -np.log(e.card)
            rows[self.arena.message_indices(e.slot1, e.card)] = -np.log(e.card)
        return jnp.asarray(rows)

    def bind_data(self, state: State, observations: dict) -> State:
        """Bind DATA ports: discrete ports take an int (None = missing =
        uniform); Gaussian ports take a length-dim vector, EF ports a scalar
        count/flip (None = missing = the likelihood node drops out)."""
        msgs, g_obs, g_mask = state.messages, state.g_obs, state.g_mask
        ef_obs, ef_mask = state.ef_obs, state.ef_mask
        for port, value in observations.items():
            if port in self.arena.ef_data:
                row, fam = self.arena.ef_data[port]
                if value is None:
                    ef_mask = ef_mask.at[row].set(0.0)
                    continue
                validator = kernels.observation_validator(fam)
                try:
                    val = float(value) if validator is None else validator(value)
                except ValueError as err:
                    raise ValueError(f"port {port!r}: {err}") from None
                ef_obs = ef_obs.at[row].set(val)
                ef_mask = ef_mask.at[row].set(1.0)
                continue
            if port in self.arena.gauss_data:
                row, dim = self.arena.gauss_data[port]
                if value is None:
                    g_mask = g_mask.at[row].set(0.0)
                else:
                    val = np.asarray(value, dtype=np.float64).reshape(-1)
                    if val.shape[0] != dim:
                        raise ValueError(
                            f"port {port!r}: observation has dim {val.shape[0]}, edge has {dim}"
                        )
                    idx = self.arena.g_obs_indices(row, dim)
                    g_obs = g_obs.at[idx].set(jnp.asarray(val, dtype=g_obs.dtype))
                    g_mask = g_mask.at[row].set(1.0)
                continue
            if port not in self.arena.data_slots:
                raise KeyError(
                    f"no DATA port {port!r}; have "
                    f"{sorted([*self.arena.data_slots, *self.arena.gauss_data, *self.arena.ef_data])}"
                )
            slot = self.arena.data_slots[port]
            card = next(
                e.card
                for e in self.arena.edges.values()
                if not e.dim and not e.ef and slot in (e.slot0, e.slot1)
            )
            row = np.full(card, NEG)
            if value is None:
                row[:card] = -np.log(card)
            else:
                if not 0 <= value < card:
                    raise ValueError(f"port {port!r}: value {value} out of range [0, {card})")
                row[value] = 0.0
            idx = self.arena.message_indices(slot, card)
            msgs = msgs.at[idx].set(jnp.asarray(row, dtype=msgs.dtype))
        return _replace(
            state, messages=msgs, g_obs=g_obs, g_mask=g_mask, ef_obs=ef_obs, ef_mask=ef_mask
        )

    # -- execution --------------------------------------------------------

    def step(self, state: State, damp: float = 0.0, harden: bool = True) -> State:
        msgs, g_eta, g_lam, ef_eta, _, _ = self._step(
            state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
            state.ef_eta, state.ef_obs, state.ef_mask, state.params, damp, harden,
        )
        return _replace(state, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta)

    def run(
        self,
        state: State,
        n_iters: int,
        *,
        tol: float | None = None,
        damp: float = 0.0,
        harden_last: int | None = None,
    ) -> RunResult:
        """Flood up to ``n_iters`` sweeps and return explicit termination data.

        In sum/GFE mode, ``tol`` applies to free-energy stationarity; message
        delta must also be below the dtype-scaled tolerance so an accidentally
        flat ledger cannot certify incoherent messages.  In max mode there is
        no FE: the message fixed-point residual is the stopping criterion and
        the decoded objective is validated when stationarity is reached.

        With ``tol=None`` the requested fixed sweep budget is always consumed
        and the status is ``budget_exhausted`` unless the run fails earlier.
        Callers that require actionable beliefs should use
        ``result.require_converged()``.

        `damp` (0..1) blends in the previous messages; use ~0.5 for models with
        multiple GFE composites, whose outer coordination can limit-cycle.

        δ form-constraints must not fire while beliefs are still garbage — a
        premature pointmass locks in its own arbitrary argmax. With
        `harden_last=k` (default: k = n_iters // 2 when δ edges exist), the
        first phase runs soft to coherence, then δ edges harden for the final
        k sweeps so decisions commit sequentially.

        Two-pass schedules (cff-d56d): when the engine compiled an exact
        directed schedule and the call keeps its contract (``damp=0``, no
        hardening phase, at least one iteration), the run executes the
        schedule once instead of flooding: each directed message is updated
        O(1) times, a certification sweep measures the true fixed-point
        residual, and the status is ``converged`` when that residual passes
        the dtype- and depth-scaled certificate (also with ``tol=None`` —
        the schedule proves the fixed point, so no sweep budget applies)."""
        if n_iters < 0:
            raise ValueError("n_iters must be non-negative")
        if harden_last is None:
            harden_last = n_iters // 2 if len(self.arena.delta_pairs) else 0
        if not 0 <= harden_last <= n_iters:
            raise ValueError("harden_last must be between 0 and n_iters")
        if (
            self._propagate is not None
            and n_iters >= 1
            and damp == 0.0
            and harden_last == 0
        ):
            return self._run_exact(state, tol)

        deltas: list[float] = []
        energies: list[float] = []
        status = RunStatus.BUDGET_EXHAUSTED
        residual = float("inf")
        inner_status = RunStatus.CONVERGED
        inner_residual = 0.0

        dtype = np.asarray(state.messages).dtype
        # A requested tolerance below the active dtype's resolution cannot be
        # falsified.  This floor protects truthful convergence in consumer f32.
        effective_tol = None if tol is None else max(float(tol), 64.0 * np.finfo(dtype).eps)

        if not self._state_is_finite(state):
            return RunResult(
                state, RunStatus.DIVERGED, 0, (), (), float("inf"),
                RunStatus.DIVERGED, float("inf"),
            )
        if self._has_zero_support(state):
            return RunResult(
                state, RunStatus.INFEASIBLE, 0, (), (float("inf"),), float("inf")
            )

        if self._prime is not None and n_iters >= 1:
            # Colored/mixed priming: equilibrate the forest islands once
            # (exact, O(E)) before the iterative loop starts counting.
            msgs, g_eta, g_lam, ef_eta = self._prime(
                state.messages, state.g_eta, state.g_lam, state.ef_eta,
                state.g_obs, state.g_mask, state.ef_obs, state.ef_mask,
                state.params,
            )
            state = _replace(
                state, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta
            )

        def phase(n: int, harden: bool, st: State) -> tuple[State, bool]:
            nonlocal status, residual, inner_status, inner_residual
            previous_fe: float | None = None
            for _ in range(n):
                msgs, g_eta, g_lam, ef_eta, inner_r, inner_code = self._iter_step(
                    st.messages, st.g_eta, st.g_lam, st.g_obs, st.g_mask,
                    st.ef_eta, st.ef_obs, st.ef_mask, st.params, damp, harden,
                )
                nxt = _replace(
                    st, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta
                )
                delta = max(
                    (
                        float(jnp.max(jnp.abs(getattr(nxt, f) - getattr(st, f))))
                        for f in ("messages", "g_eta", "g_lam", "ef_eta")
                        if getattr(st, f).size
                    ),
                    default=0.0,
                )
                deltas.append(delta)
                st = nxt
                inner_residual = float(inner_r)
                code = int(inner_code)
                inner_status = (
                    RunStatus.CONVERGED if code == 0 else
                    RunStatus.BUDGET_EXHAUSTED if code == 1 else
                    RunStatus.DIVERGED
                )

                if (
                    not np.isfinite(delta)
                    or not self._state_is_finite(st)
                    or not np.isfinite(inner_residual)
                    or inner_status is RunStatus.DIVERGED
                ):
                    status = RunStatus.DIVERGED
                    residual = float("inf")
                    return st, True
                if self._has_zero_support(st):
                    status = RunStatus.INFEASIBLE
                    residual = float("inf")
                    energies.append(float("inf"))
                    return st, True
                if inner_status is RunStatus.BUDGET_EXHAUSTED:
                    status = RunStatus.BUDGET_EXHAUSTED
                    residual = inner_residual
                    return st, True

                if self.arena.semiring == "max":
                    residual = delta
                    if effective_tol is not None and delta <= effective_tol:
                        objective = self.objective(st)
                        if not np.isfinite(objective):
                            status = RunStatus.DIVERGED
                            residual = float("inf")
                        else:
                            status = RunStatus.CONVERGED
                        return st, True
                    continue

                fe = self.bfe(st)
                if np.isfinite(fe):
                    energies.append(fe)
                    if previous_fe is not None:
                        residual = abs(fe - previous_fe)
                        if (
                            effective_tol is not None
                            and residual <= effective_tol
                            and delta <= effective_tol
                        ):
                            status = RunStatus.CONVERGED
                            return st, True
                    previous_fe = fe
                elif previous_fe is not None or not (
                    self.arena.gauss_groups or self.arena.ef_groups or self.arena.prec_groups
                ):
                    status = RunStatus.DIVERGED
                    residual = float("inf")
                    return st, True
            return st, False

        state, stopped = phase(n_iters - harden_last, harden=False, st=state)
        if not stopped and harden_last:
            state, stopped = phase(harden_last, harden=True, st=state)
        elif stopped and harden_last and status is RunStatus.CONVERGED:
            # Soft coherence is only the start state for the hardening phase.
            status = RunStatus.BUDGET_EXHAUSTED
            state, stopped = phase(harden_last, harden=True, st=state)

        if (
            status is RunStatus.BUDGET_EXHAUSTED
            and self.arena.semiring != "max"
            and not energies
            and deltas
            and (self.arena.gauss_groups or self.arena.ef_groups or self.arena.prec_groups)
        ):
            # A continuous/EF ledger that never became defined is not a
            # usable budget result; it is a numerically failed inference.
            status = RunStatus.DIVERGED
            residual = float("inf")
        return RunResult(
            state=state,
            status=status,
            iterations=len(deltas),
            message_deltas=tuple(deltas),
            free_energies=tuple(energies),
            residual=residual,
            inner_status=inner_status,
            inner_residual=inner_residual,
        )

    @staticmethod
    def _certificate_floor(state: State) -> float:
        """Dtype- and scale-relative certificate for the fixed-point residual.

        The certification sweep recomputes each message once from already-
        final inputs, so its residual is a few ulps of the values involved —
        *local* rounding, independent of graph diameter. The floor is
        therefore 64 eps times the largest live magnitude in the state
        (log-messages near the NEG log-zero sentinel recompute exactly and
        are excluded from the scale). A diameter-scaled floor would grow
        past genuine staleness signals on long float32 chains — the false-
        convergence defect this replaces (cff-d56d review, 2026-07-27)."""
        dtype = np.asarray(state.messages).dtype
        scale = 1.0
        msgs = np.asarray(state.messages)
        live = msgs[np.abs(msgs) < 0.5 * abs(NEG)]
        if live.size:
            scale = max(scale, float(np.max(np.abs(live))))
        for arr in (state.g_eta, state.g_lam, state.ef_eta):
            a = np.asarray(arr)
            if a.size:
                scale = max(scale, float(np.max(np.abs(a))))
        return 64.0 * float(np.finfo(dtype).eps) * scale

    def _run_exact(self, state: State, tol: float | None) -> RunResult:
        """Execute the two-pass directed schedule once and certify it.

        On an acyclic BP graph one inward+outward execution reaches the BP
        fixed point, so there is no sweep loop: the run costs O(E). A
        certification sweep (one synchronous flooding step) then re-derives
        every directed message from the propagated state; its max change is
        the true fixed-point residual, so a scheduling defect — any stale
        message — surfaces as a non-converged status rather than as silently
        wrong marginals."""

        if not self._state_is_finite(state):
            return RunResult(
                state, RunStatus.DIVERGED, 0, (), (), float("inf"),
                RunStatus.DIVERGED, float("inf"),
            )

        # One dispatch: pre/post support flags, propagation, certification
        # sweep, residual, and ledger execute as a single XLA computation
        # (review item (c)).
        msgs, g_eta, g_lam, ef_eta, delta_dev, fe_dev, pre_bad, post_bad = (
            self._exact_unit(
                state.messages, state.g_eta, state.g_lam, state.g_obs,
                state.g_mask, state.ef_eta, state.ef_obs, state.ef_mask,
                state.params,
            )
        )
        if bool(pre_bad):  # proven-empty support before any propagation
            return RunResult(
                state, RunStatus.INFEASIBLE, 0, (), (float("inf"),), float("inf")
            )
        st = _replace(state, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta)
        delta = float(delta_dev)
        if not np.isfinite(delta) or not self._state_is_finite(st):
            return RunResult(st, RunStatus.DIVERGED, 1, (delta,), (), float("inf"))
        if bool(post_bad):
            # The fused ledger value is discarded: Z = 0 means F = +inf, and
            # a finite NEG-sentinel score must never leak (cff-j4mu).
            return RunResult(
                st, RunStatus.INFEASIBLE, 1, (), (float("inf"),), float("inf")
            )

        floor = self._certificate_floor(st)
        certificate = floor if tol is None else max(float(tol), floor)
        status = (
            RunStatus.CONVERGED if delta <= certificate else RunStatus.BUDGET_EXHAUSTED
        )
        if self.arena.semiring == "max":
            if status is RunStatus.CONVERGED and not np.isfinite(self.objective(st)):
                return RunResult(st, RunStatus.DIVERGED, 1, (delta,), (), float("inf"))
            return RunResult(st, status, 1, (delta,), (), delta)

        fe = float(fe_dev)
        if not np.isfinite(fe):
            return RunResult(st, RunStatus.DIVERGED, 1, (delta,), (), float("inf"))
        return RunResult(st, status, 1, (delta,), (fe,), delta)

    @staticmethod
    def _state_is_finite(state: State) -> bool:
        """All numeric runtime leaves must remain finite (NEG is finite)."""
        return all(
            bool(np.isfinite(np.asarray(leaf)).all())
            for leaf in jax.tree_util.tree_leaves(state)
        )

    def _has_zero_support(self, state: State) -> bool:
        """Whether an edge or ordinary factor belief has empty exact support.

        Vectorized over each edge-cardinality bucket and each group's member
        stack (cff-d56d): the check runs on every exact-path run, so a
        per-member Python loop would put an O(T) interpreter cost back into
        the O(E) schedule."""
        messages = np.asarray(state.messages)
        for card, pairs in self._support_edge_groups:
            logq = (
                messages[self.arena.message_indices(pairs[:, 0], card)]
                + messages[self.arena.message_indices(pairs[:, 1], card)]
            )
            if not np.all(np.any(np.exp(logq) > 0, axis=-1)):
                return True

        for group in self.arena.groups:
            supports = [  # per role [M, card]: exact support of the incoming
                np.exp(messages[self.arena.message_indices(group.gather[:, r], card)]) > 0
                for r, card in enumerate(group.cards)
            ]
            n = len(group.members)
            if group.type == "equality":
                joint = np.logical_and.reduce(supports)  # [M, k]
                if not np.all(np.any(joint, axis=-1)):
                    return True
                continue
            factor_support = np.exp(np.asarray(state.params[group.name])) > 0
            for role, card in enumerate(group.cards):
                shape = [n] + [1] * len(group.cards)
                shape[1 + role] = card
                factor_support = factor_support & supports[role].reshape(shape)
            if not np.all(np.any(factor_support.reshape(n, -1), axis=-1)):
                return True

        for constraint in self.arena.constraint_groups:
            supports = [
                np.flatnonzero(
                    np.exp(messages[self.arena.message_indices(constraint.gather[i], card)]) > 0
                ).tolist()
                for i, card in enumerate(constraint.cards)
            ]
            if any(not support for support in supports):
                return True
            if constraint.type == "exactly_k":
                possible = {0}
                for support in supports:
                    possible = {total + value for total in possible for value in support}
                feasible = constraint.k in possible
            elif constraint.type == "at_most_k":
                feasible = sum(min(support) for support in supports) <= constraint.k
            elif constraint.type == "table":
                feasible = any(
                    all(int(row[i]) in supports[i] for i in range(len(supports)))
                    for row in np.asarray(constraint.table)
                )
            else:  # alldifferent, n/card <= 10 by compiler validation
                def match(i: int, used: set[int]) -> bool:
                    if i == len(supports):
                        return True
                    return any(
                        value not in used and match(i + 1, used | {value})
                        for value in supports[i]
                    )

                feasible = match(0, set())
            if not feasible:
                return True
        return False

    # -- readouts ---------------------------------------------------------

    def marginals(self, state: State) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        msgs = np.asarray(state.messages)
        for e in self.arena.edges.values():
            if e.dim or e.ef:
                continue  # gaussian_marginals / ef_marginals
            lq = (
                msgs[self.arena.message_indices(e.slot0, e.card)]
                + msgs[self.arena.message_indices(e.slot1, e.card)]
            )
            lq -= lq.max()
            q = np.exp(lq)
            out[e.name] = q / q.sum()
        return out

    def node_beliefs(self, state: State) -> dict[str, np.ndarray]:
        """Factor name -> the node belief q_a, the object conjugate updates
        integrate against (cff-cd1i): e.g. a transition_mixture's q_a(z', z, u)
        is the smoothed pairwise marginal whose slices are Dirichlet count
        increments for B-learning.

        BP groups: exact q_a ∝ f_a · Π incoming, shaped [*cards]. VMP groups:
        the mean-field belief (outer product of edge marginals) — the correct
        expectation measure under that bead. Equality nodes report the shared
        marginal [k] (their joint lives on the diagonal). GFE composites and
        Gaussian groups are not included: composites own their update path
        (agent A-learning), Gaussian sufficient stats land with cff-7c5o."""
        out: dict[str, np.ndarray] = {}
        msgs = np.asarray(state.messages)

        def norm_exp(lb: np.ndarray) -> np.ndarray:
            m = lb.shape[0]
            flat = lb.reshape(m, -1)
            flat = flat - flat.max(axis=1, keepdims=True)
            q = np.exp(flat)
            return (q / q.sum(axis=1, keepdims=True)).reshape(lb.shape)

        for g in self.arena.groups:
            inc = np.asarray(_disc_rows(self.arena, state.messages, g.gather, g.cards))
            if g.type == "equality":
                k = g.cards[0]
                q = norm_exp(inc[:, :, :k].sum(axis=1))
            elif g.mode == "vmp":
                both = np.asarray(_disc_both(self.arena, state.messages, g.gather_both, g.cards))
                q = None
                for r, k in enumerate(g.cards):
                    lq = both[:, r, 0, :k] + both[:, r, 1, :k]
                    edge_q = norm_exp(lq).reshape(len(g.members), *([1] * r), k)
                    q = edge_q if q is None else q[..., None] * edge_q
            else:
                lb = np.asarray(state.params[g.name]).copy()  # [M, *cards]
                for r, k in enumerate(g.cards):
                    shape = [len(g.members)] + [1] * len(g.cards)
                    shape[1 + r] = k
                    lb = lb + inc[:, r, :k].reshape(shape)
                q = norm_exp(lb)
            for i, name in enumerate(g.members):
                out[name] = q[i]
        return out

    def gaussian_marginals(self, state: State) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Edge name -> (mean, cov). Requires a proper combined precision —
        run to coherence (and bind data) first. Improper beliefs return NaN
        mean/cov arrays, an explicit readout that cannot resemble a valid
        negative or huge covariance."""
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        eta, lam = np.asarray(state.g_eta), np.asarray(state.g_lam)
        for e in self.arena.edges.values():
            if not e.dim:
                continue
            d = e.dim
            eta_q = (
                eta[self.arena.g_eta_indices(e.slot0, d)]
                + eta[self.arena.g_eta_indices(e.slot1, d)]
            )
            lam_q = (
                lam[self.arena.g_lam_indices(e.slot0, d)]
                + lam[self.arena.g_lam_indices(e.slot1, d)]
            )
            mean, cov = gauss.moments(eta_q, lam_q)
            out[e.name] = (np.asarray(mean), np.asarray(cov))
        return out

    def gaussian_node_stats(self, state: State) -> dict[str, dict[str, np.ndarray]]:
        """Factor name -> expected sufficient statistics of the node belief
        (cff-7c5o, D4): the continuous analog of node_beliefs, feeding the
        NIW/MNIW conjugate updates in cffg.conj.

        gaussian:            {"mean" E[x], "second" E[xxᵀ]}
        gaussian_linear:     {"Syy" E[yyᵀ], "Syx" E[yx̃ᵀ], "Sxx" E[x̃x̃ᵀ]}
                             with x̃ = [x; 1] (offset column absorbed)
        gaussian_linear_obs: same keys with y fixed at the bound observation,
                             plus "mask" (0.0 = missing: skip the occurrence).
        Every record includes ``proper``. Improper beliefs or mutated non-PD
        node precisions return ``proper=False`` with NaN sufficient statistics;
        equality and hybrid nodes are not learnable and are skipped."""
        out: dict[str, dict[str, np.ndarray]] = {}
        g_eta, g_lam = np.asarray(state.g_eta), np.asarray(state.g_lam)
        g_obs, g_mask = np.asarray(state.g_obs), np.asarray(state.g_mask)
        for g in self.arena.gauss_groups:
            if g.type == "equality":
                continue
            inc_e, inc_l = _gauss_rows(self.arena, state.g_eta, state.g_lam, g.gather, g.dims)
            inc_e, inc_l = np.asarray(inc_e), np.asarray(inc_l)
            for i, name in enumerate(g.members):
                w = np.asarray(state.params[g.name + ":W"][i])
                if g.type == "gaussian":
                    (d,) = g.dims
                    m_blob = np.asarray(state.params[g.name + ":m"][i])
                    lam_q = w + inc_l[i, 0, :d, :d]
                    proper = bool(gauss.is_proper_precision(w)) and bool(
                        gauss.is_proper_precision(lam_q)
                    )
                    m_q, sig_q = gauss.moments(
                        w @ m_blob + inc_e[i, 0, :d], lam_q
                    )
                    m_q, sig_q = np.asarray(m_q), np.asarray(sig_q)
                    if not proper:
                        m_q = np.full_like(m_q, np.nan)
                        sig_q = np.full_like(sig_q, np.nan)
                    out[name] = {
                        "mean": m_q,
                        "second": sig_q + np.outer(m_q, m_q),
                        "proper": np.asarray(proper),
                    }
                elif g.type == "gaussian_linear":
                    do, di = g.dims
                    a = np.asarray(state.params[g.name + ":A"][i])
                    b = np.asarray(state.params[g.name + ":b"][i])
                    e_o, l_o = inc_e[i, 0, :do], inc_l[i, 0, :do, :do]
                    e_i, l_i = inc_e[i, 1, :di], inc_l[i, 1, :di, :di]
                    lam_j = np.block([[w + l_o, -w @ a], [-a.T @ w, a.T @ w @ a + l_i]])
                    eta_j = np.concatenate([w @ b + e_o, -a.T @ w @ b + e_i])
                    proper = bool(gauss.is_proper_precision(w)) and bool(
                        gauss.is_proper_precision(lam_j)
                    )
                    m_j, sig_j = gauss.moments(eta_j, lam_j)
                    m_j, sig_j = np.asarray(m_j), np.asarray(sig_j)
                    if not proper:
                        m_j = np.full_like(m_j, np.nan)
                        sig_j = np.full_like(sig_j, np.nan)
                    m_y, m_x = m_j[:do], m_j[do:]
                    s_yy = sig_j[:do, :do] + np.outer(m_y, m_y)
                    s_yx = sig_j[:do, do:] + np.outer(m_y, m_x)
                    s_xx = sig_j[do:, do:] + np.outer(m_x, m_x)
                    syx_aug = np.concatenate([s_yx, m_y[:, None]], axis=1)
                    sxx_aug = np.block(
                        [[s_xx, m_x[:, None]], [m_x[None, :], np.ones((1, 1))]]
                    )
                    if not proper:
                        s_yy = np.full_like(s_yy, np.nan)
                        syx_aug = np.full_like(syx_aug, np.nan)
                        sxx_aug = np.full_like(sxx_aug, np.nan)
                    out[name] = {
                        "Syy": s_yy,
                        "Syx": syx_aug,
                        "Sxx": sxx_aug,
                        "proper": np.asarray(proper),
                    }
                else:  # gaussian_linear_obs: y is data; q(x) is conditioned
                    do, di = g.dims
                    a = np.asarray(state.params[g.name + ":A"][i])
                    b = np.asarray(state.params[g.name + ":b"][i])
                    e_i, l_i = inc_e[i, 1, :di], inc_l[i, 1, :di, :di]
                    row = int(g.obs_rows[i])
                    y, mk = g_obs[self.arena.g_obs_indices(row, do)], g_mask[row]
                    lam_q = mk * (a.T @ w @ a) + l_i
                    proper = bool(gauss.is_proper_precision(w)) and bool(
                        gauss.is_proper_precision(lam_q)
                    )
                    m_x, sig_q = gauss.moments(
                        mk * (a.T @ (w @ (y - b))) + e_i, lam_q
                    )
                    m_x, sig_q = np.asarray(m_x), np.asarray(sig_q)
                    if not proper:
                        m_x = np.full_like(m_x, np.nan)
                        sig_q = np.full_like(sig_q, np.nan)
                    s_xx = sig_q + np.outer(m_x, m_x)
                    syx_aug = np.concatenate(
                        [np.outer(y, m_x), y[:, None]], axis=1
                    )
                    sxx_aug = np.block(
                        [[s_xx, m_x[:, None]], [m_x[None, :], np.ones((1, 1))]]
                    )
                    if not proper:
                        syx_aug = np.full_like(syx_aug, np.nan)
                        sxx_aug = np.full_like(sxx_aug, np.nan)
                    out[name] = {
                        "Syy": (
                            np.outer(y, y)
                            if proper
                            else np.full((do, do), np.nan)
                        ),
                        "Syx": syx_aug,
                        "Sxx": sxx_aug,
                        "mask": np.asarray(mk),
                        "proper": np.asarray(proper),
                    }
        return out

    def ef_marginals(self, state: State) -> dict[str, "EfMarginal"]:
        """Edge name -> EF marginal in standard parameters. Observation-family
        edges (poisson/bernoulli DATA ports) carry no messages and are
        skipped. An improper combined belief (e.g. before any sweep) reports
        proper=False with NaN moments — check it before trusting `mean`."""
        out: dict[str, EfMarginal] = {}
        eta = np.asarray(state.ef_eta)
        for e in self.arena.edges.values():
            if not e.ef or e.ef not in ef.FAMILIES:
                continue
            fam = ef.FAMILIES[e.ef]
            eta_q = eta[e.slot0] + eta[e.slot1]
            proper = bool(fam.is_proper(eta_q))
            out[e.name] = EfMarginal(
                family=e.ef,
                params=tuple(float(p) for p in fam.from_natural(eta_q)),
                mean=float(fam.mean(eta_q)) if proper else float("nan"),
                proper=proper,
            )
        return out

    def kernel_readouts(self, state: State) -> dict[str, dict[str, np.ndarray]]:
        """Factor name -> the optional readout its FactorKernel declares
        (cff-ri8m): posterior summaries beyond the generic edge marginals,
        computed from the same (inc, params, obs, mask) contract as the
        message/energy callables. Kernels without a readout are skipped."""
        out: dict[str, dict[str, np.ndarray]] = {}
        for g in self.arena.ef_groups:
            kern = kernels.kernel_for(g.type)
            if kern is None or kern.readout is None:
                continue
            gp = {key: state.params[f"{g.name}:{key}"] for key in kern.param_keys}
            obs = state.ef_obs[g.obs_rows] if g.obs_rows is not None else None
            mask = state.ef_mask[g.obs_rows] if g.obs_rows is not None else None
            vals = kern.readout(state.ef_eta[np.asarray(g.gather)], gp, obs, mask)
            for i, member in enumerate(g.members):
                out[member] = {key: np.asarray(v)[i] for key, v in vals.items()}
        return out

    def bfe(self, state: State) -> float:
        if self.arena.semiring == "max":
            raise RuntimeError(
                "min-sum reports an objective value, not a free energy — call Engine.objective"
            )
        # NEG is a finite representation of log zero, so the local ledger can
        # otherwise turn log Z = -inf into a huge finite sentinel score.  Empty
        # exact edge support proves Z=0 and therefore F=+inf.
        if self._has_zero_support(state):
            return float("inf")
        return float(
            self._bfe(
                state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
                state.ef_eta, state.ef_obs, state.ef_mask, state.params,
            )
        )

    def decode(self, state: State) -> dict[str, int]:
        """MAP decode x-hat: the per-edge argmax of the (max-)marginals. Defined
        in both semirings (sum-mode marginals decode too); in max mode at a tree
        fixed point this is the exact Viterbi assignment (cff-e8zk)."""
        return {name: int(np.argmax(q)) for name, q in self.marginals(state).items()}

    def objective(self, state: State) -> float:
        """Exact -log f(x-hat) of the decoded assignment, summed over every
        factor from the compiled parameters + the decode (NOT from messages;
        messages only guide the decode). Equality and hard-constraint indicators
        contribute 0 when the decode is consistent and +inf when it is not — an
        inconsistent decode is reported loudly. On a tree at the max-product
        fixed point this equals the brute-force MAP energy (Viterbi identity)."""
        xhat = self.decode(state)
        slot2var = {}
        for e in self.arena.edges.values():
            if e.dim or e.ef:
                continue
            slot2var[e.slot0] = e.name
            slot2var[e.slot1] = e.name
        total = 0.0
        for g in self.arena.groups:
            vals = [
                [xhat[slot2var[int(g.scatter[i, r])]] for r in range(len(g.roles))]
                for i in range(len(g.members))
            ]
            if g.type == "equality":
                for v in vals:
                    total += 0.0 if v[0] == v[1] == v[2] else np.inf
                continue
            logf = np.asarray(state.params[g.name])  # [M, *cards], log potential
            for i, v in enumerate(vals):
                total += -float(logf[(i, *v)])
        for c in self.arena.constraint_groups:
            v = [xhat[slot2var[int(c.scatter[j])]] for j in range(len(c.cards))]
            total += 0.0 if self._constraint_ok(c, v) else np.inf
        return total

    @staticmethod
    def _constraint_ok(c, v: list[int]) -> bool:
        if c.type == "alldifferent":
            return len(set(v)) == len(v)
        if c.type == "exactly_k":
            return sum(v) == c.k
        if c.type == "at_most_k":
            return sum(v) <= c.k
        # table: the assignment must be one of the allowed tuples
        return any(np.array_equal(np.asarray(v), row) for row in np.asarray(c.table))


@dataclass(frozen=True)
class EfMarginal:
    """A scalar EF edge belief: standard parameters, e.g. (shape, rate) for
    gamma, (a, b) for beta."""

    family: str
    params: tuple[float, ...]
    mean: float
    proper: bool


def _mix_sweep(arena, messages, g_eta, g_lam, params):
    """One synchronous flooding sweep over the hybrid Gaussian-mixture groups
    (cff-pksr). Reads only the pre-sweep arrays; returns the writes for both
    arenas so the caller can fold them into each arena's damping phase:
    Each write carries its exact cardinality/dimension; no model-wide padded
    representation is constructed. Dims and K are static per group."""
    sel_writes = []
    gauss_writes = []
    for g in arena.mix_groups:
        mu_e = messages[arena.message_indices(g.s_gather, g.k_sel)]
        inc_e, inc_l = _gauss_rows(arena, g_eta, g_lam, g.g_gather, g.dims)
        w = params[g.name + ":W"]
        if g.type == "gaussian_mixture_linear":
            do, di = g.dims
            a, b = params[g.name + ":A"], params[g.name + ":b"]
            e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
            e_i, l_i = inc_e[:, 1, :di], inc_l[:, 1, :di, :di]
            g_outs = [
                mixg.mixlin_forward(e_i, l_i, mu_e, a, b, w),  # -> out
                mixg.mixlin_backward(e_o, l_o, mu_e, a, b, w),  # -> in
            ]
            to_sel = mixg.mixlin_to_selector(e_o, l_o, e_i, l_i, a, b, w)  # [M, K]
        else:  # gaussian_mixture
            (do,) = g.dims
            m = params[g.name + ":m"]
            e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
            g_outs = [mixg.mix_forward(mu_e, m, w)]  # -> out
            to_sel = mixg.mix_to_selector(e_o, l_o, m, w)  # [M, K]
        for r, (oe, ol) in enumerate(g_outs):
            gauss_writes.append((g.g_scatter[:, r], g.dims[r], oe, ol))
        sel_writes.append((g.s_scatter, g.k_sel, to_sel))
    return sel_writes, gauss_writes


def _prec_sweep(arena, g_eta, g_lam, ef_eta, params):
    """One synchronous flooding sweep over the gaussian_prec groups (cff-7c5o).
    Straddles the Gaussian and EF arenas; reads only the pre-sweep arrays and
    returns the writes for both so they fold into each arena's damping phase:
    Both VMP messages read marginals (both slots summed), the mean-field
    convention."""
    gauss_writes = []
    ef_writes = []
    for g in arena.prec_groups:
        tau_q = ef_eta[g.tau_gather_both].sum(axis=1)  # [M, 2] τ-edge marginal
        x_eta_q = g_eta[arena.g_eta_indices(g.x_gather_both, 1)].sum(axis=1)
        x_lam_q = g_lam[arena.g_lam_indices(g.x_gather_both, 1)].sum(axis=1)
        m = params[g.name + ":m"]  # [M, 1]
        oe, ol = gauss.prec_to_x(tau_q, m)  # (eta [M,1], lam [M,1,1])
        gauss_writes.append((g.x_scatter, 1, oe, ol))
        ef_writes.append((g.tau_scatter, gauss.prec_to_tau(x_eta_q, x_lam_q, m)))  # [M, 2]
    return gauss_writes, ef_writes


def _gauss_outs(g, inc_e, inc_l, par, y=None, mk=None):
    """Per-role outgoing (eta, Lambda) messages for one Gaussian group.

    Shared by the flooding sweep and the directed-schedule executors
    (cff-d56d): `par` holds the group's parameter stacks already subset to
    the members being fired, `inc_e`/`inc_l` the matching gathered rows."""
    if g.type == "gaussian":
        w, m = par["W"], par["m"]
        return [gauss.prior_message(jnp.einsum("mij,mj->mi", w, m), w)]
    if g.type == "equality":
        d = g.dims[0]
        to_e, to_l = gauss.equality_messages(inc_e[..., :d], inc_l[..., :d, :d])
        return [(to_e[:, r], to_l[:, r]) for r in range(len(g.roles))]
    do, di = g.dims  # gaussian_linear / gaussian_linear_obs; roles = (out, in)
    a, b, w = par["A"], par["b"], par["W"]
    e_in, l_in = inc_e[:, 1, :di], inc_l[:, 1, :di, :di]
    outs = [gauss.linear_forward(e_in, l_in, a, b, w)]
    if g.type == "gaussian_linear_obs":
        outs.append(gauss.linear_observed_backward(y, mk, a, b, w))
    else:
        e_out, l_out = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
        outs.append(gauss.linear_backward(e_out, l_out, a, b, w))
    return outs


def _gauss_group_params(g, params, rows=None):
    """The group's parameter stacks, keyed bare and optionally row-subset."""
    keys = {
        "gaussian": ("W", "m"),
        "equality": (),
        "gaussian_linear": ("A", "b", "W"),
        "gaussian_linear_obs": ("A", "b", "W"),
    }[g.type]
    return {
        k: params[f"{g.name}:{k}"] if rows is None else params[f"{g.name}:{k}"][rows]
        for k in keys
    }


def _gauss_sweep(arena, g_eta, g_lam, g_obs, g_mask, params):
    """One synchronous flooding sweep over the Gaussian groups; returns the
    fully rewritten flat (eta, Lambda) arrays. Dims are static per group, so
    every rule call sees a shape-local block."""
    new_eta, new_lam = g_eta, g_lam
    for g in arena.gauss_groups:
        inc_e, inc_l = _gauss_rows(arena, g_eta, g_lam, g.gather, g.dims)
        y = mk = None
        if g.type == "gaussian_linear_obs":
            y, mk = _obs_rows(arena, g_obs, g.obs_rows, g.dims[0]), g_mask[g.obs_rows]
        outs = _gauss_outs(g, inc_e, inc_l, _gauss_group_params(g, params), y, mk)
        for r, (oe, ol) in enumerate(outs):
            new_eta, new_lam = _set_gauss_block(
                arena, new_eta, new_lam, g.scatter[:, r], g.dims[r], oe, ol
            )
    return new_eta, new_lam


def _ef_group_out(g, inc, params, rows=None, obs=None, mask=None):
    """Outgoing EF messages [m, R, 2] for one group from gathered ``inc``.

    Shared by the flooding sweep and the directed-schedule executors
    (cff-d56d): ``rows`` optionally subsets the group's parameter stacks to
    the members being fired."""
    if g.type == "equality":
        return efrules.equality_messages(inc)
    kern = kernels.KERNELS[g.type]
    gp = {
        key: params[f"{g.name}:{key}"] if rows is None else params[f"{g.name}:{key}"][rows]
        for key in kern.param_keys
    }
    return kern.message(inc, gp, obs, mask)


def _ef_sweep(arena, ef_eta, ef_obs, ef_mask, params):
    """One synchronous flooding sweep over the EF groups — conjugate BP in
    its entirety (DERIVATIONS.md D2). Each non-equality group's update comes
    from its registered FactorKernel (cffg.kernels, cff-ri8m); equality is
    structural sum-minus-self. Groups are static, so kernel dispatch happens
    at trace time and the jitted step is branch-free."""
    new = ef_eta
    for g in arena.ef_groups:
        obs = ef_obs[g.obs_rows] if g.obs_rows is not None else None
        mask = ef_mask[g.obs_rows] if g.obs_rows is not None else None
        out = _ef_group_out(g, ef_eta[g.gather], params, None, obs, mask).astype(new.dtype)
        new = new.at[g.scatter.reshape(-1)].set(out.reshape(-1, out.shape[-1]))
    return new


def _constraint_writes(arena: Arena, messages: jax.Array, op):
    """Combinatorial messages for the hard-constraint factors (cff-e8zk). Reads
    the pre-sweep messages (synchronous flooding); returns exact segmented
    writes for the discrete arena, folded into damping by the caller."""
    writes = []
    for c in arena.constraint_groups:
        k_local = max(c.cards)
        inc = _disc_rows(arena, messages, c.gather, c.cards)
        fn = csp.CONSTRAINT_MESSAGES[c.type]
        if c.type in ("exactly_k", "at_most_k"):
            out = fn(inc, c.cards, k_local, op, k=c.k)
        elif c.type == "table":
            out = fn(inc, c.cards, k_local, op, table=jnp.asarray(c.table))
        else:
            out = fn(inc, c.cards, k_local, op)
        writes.append((c.scatter, c.cards, out))
    return writes


def _make_step(arena: Arena, skip_scheduled: bool = False):
    """The synchronous sweep. With ``skip_scheduled=True`` (the mixed-mode
    phase step, cff-d56d) the plain BP discrete groups, Gaussian groups, and
    EF groups are omitted — the directed island passes own them — while
    every iterative structure (vmp beads, GFE composites, constraints,
    hybrid mixtures, gaussian_prec) still fires and damps exactly as in
    flooding."""
    groups = (
        [g for g in arena.groups if g.mode != "bp"] if skip_scheduled else arena.groups
    )
    op = jnp.max if arena.semiring == "max" else logsumexp

    def step(
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        ef_eta: jax.Array,
        ef_obs: jax.Array,
        ef_mask: jax.Array,
        params: dict[str, jax.Array],
        damp: float = 0.0,
        harden: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        new = messages
        inner_residual = jnp.asarray(0.0, dtype=messages.dtype)
        inner_status = jnp.asarray(0)
        for g in groups:
            logf = params.get(g.name)
            k_local = max(g.cards)
            if g.mode == "vmp":
                inc2 = _disc_both(arena, messages, g.gather_both, g.cards)
                out = VMP_UPDATES[g.type](inc2, logf, g.cards, k_local)
            else:
                inc = _disc_rows(arena, messages, g.gather, g.cards)
                out = BP_UPDATES[g.type](inc, logf, g.cards, k_local, op)
            new = _set_disc_rows(arena, new, g.scatter, g.cards, out)
        for scatter, cards, out in _constraint_writes(arena, messages, op):
            new = _set_disc_rows(arena, new, scatter, cards, out)
        for g in arena.gfe_groups:
            ko, ki = g.cards
            k_local = max(g.cards)
            log_d = messages[arena.message_indices(g.gather_z, ki)]
            a_key = g.name + (":counts" if g.learned else ":A")
            mu2, log_qx, residuals, statuses = gfe_update(
                log_d, params[a_key], params[g.name + ":c"], k_local,
                learned=g.learned, return_diagnostics=True,
            )
            new = _set_disc_block(arena, new, g.scatter_z, ki, mu2)
            new = _set_disc_block(arena, new, g.scatter_x, ko, log_qx)
            inner_residual = jnp.maximum(inner_residual, jnp.max(residuals))
            inner_status = jnp.maximum(inner_status, jnp.max(statuses))
        for g in arena.gfe_tm_groups:
            ko, ki, ku = g.cards
            k_local = max(g.cards)
            md = messages[arena.message_indices(g.gather[:, 0], ko)]
            mi = messages[arena.message_indices(g.gather[:, 1], ki)]
            me = messages[arena.message_indices(g.gather[:, 2], ku)]
            b_key = g.name + (":counts" if g.learned else ":B")
            to_out, to_in, to_sel, residuals, statuses = gfe_tm_update(
                mi, me, md, params[b_key], k_local,
                learned=g.learned, return_diagnostics=True,
            )
            new = _set_disc_block(arena, new, g.scatter[:, 0], ko, to_out)
            new = _set_disc_block(arena, new, g.scatter[:, 1], ki, to_in)
            new = _set_disc_block(arena, new, g.scatter[:, 2], ku, to_sel)
            inner_residual = jnp.maximum(inner_residual, jnp.max(residuals))
            inner_status = jnp.maximum(inner_status, jnp.max(statuses))
        # Hybrid Gaussian-mixture sweep (cff-pksr): straddles both arenas.
        # Computed here from the pre-sweep arrays so its writes fold into each
        # arena's damping phase — Gaussian writes below, selector writes with
        # the discrete blend (synchronous flooding: gather from OLD arrays).
        mix_sel, mix_gauss = (
            _mix_sweep(arena, messages, g_eta, g_lam, params)
            if arena.mix_groups
            else ([], [])
        )
        # gaussian_prec sweep (cff-7c5o): also straddles two arenas — its out
        # edge writes the Gaussian arena, its prec edge the EF arena. Computed
        # here from the pre-sweep arrays so both writes fold into the matching
        # arena's damping phase (synchronous flooding: gather from OLD arrays).
        prec_gauss, prec_ef = (
            _prec_sweep(arena, g_eta, g_lam, ef_eta, params)
            if arena.prec_groups
            else ([], [])
        )
        # Gaussian sweep: information-form damping below is the log-space
        # damping of the Gaussian family (log of a Gaussian is its (eta,
        # Lambda) quadratic), so one damp knob covers both arenas.
        new_eta, new_lam = g_eta, g_lam
        if arena.gauss_groups and not skip_scheduled:
            new_eta, new_lam = _gauss_sweep(arena, g_eta, g_lam, g_obs, g_mask, params)
        for scatter, dim, oe, ol in [*mix_gauss, *prec_gauss]:
            new_eta, new_lam = _set_gauss_block(
                arena, new_eta, new_lam, scatter, dim, oe, ol
            )
        if arena.gauss_groups or arena.mix_groups or arena.prec_groups:
            new_eta = damp * g_eta + (1.0 - damp) * new_eta
            new_lam = damp * g_lam + (1.0 - damp) * new_lam
        # EF sweep: natural parameters are the log-space of their family, so
        # the same damp knob is log-space damping here too.
        new_ef = ef_eta
        if arena.ef_groups or arena.prec_groups:
            if arena.ef_groups and not skip_scheduled:
                new_ef = _ef_sweep(arena, ef_eta, ef_obs, ef_mask, params)
            for scatter, out in prec_ef:
                new_ef = new_ef.at[scatter].set(out)
            new_ef = damp * ef_eta + (1.0 - damp) * new_ef
        # Hybrid selector writes join the discrete arena before its damping.
        for scatter, card, rows in mix_sel:
            new = _set_disc_block(arena, new, scatter, card, rows)
        # Log-space damping: stabilizes the outer coordination between GFE
        # composites (their local solves are exact, but synchronous flooding
        # between them can limit-cycle — the outer face of Part II §4.5).
        new = damp * messages + (1.0 - damp) * new
        # Floor at NEG: exact log-zeros compound additively through chains
        # (NEG + NEG + ...), and unchecked they overflow to -inf and NaN once
        # δ-hardened controls make states truly impossible. exp(NEG) == 0
        # either way, so clamping changes nothing semantically.
        new = jnp.maximum(new, NEG)
        if harden and len(arena.delta_pairs):
            # δ form constraint: project each constrained edge onto a pointmass
            # at the argmax of its current marginal (both slots one-hot).
            for card, pairs in arena.delta_slot_groups:
                p0, p1 = pairs[:, 0], pairs[:, 1]
                i0 = arena.message_indices(p0, card)
                i1 = arena.message_indices(p1, card)
                lq = new[i0] + new[i1]
                hot = jnp.where(
                    jnp.arange(card)[None, :] == jnp.argmax(lq, axis=-1)[:, None],
                    0.0,
                    NEG,
                ).astype(new.dtype)
                new = new.at[i0].set(hot).at[i1].set(hot)
        return new, new_eta, new_lam, new_ef, inner_residual, inner_status

    return step


def _batch_stage_fn(arena: Arena, stage, op):
    """Executor for one BatchStage: fire a member subset of one group,
    reading the *current* arrays (sequential semantics — the whole point of
    a directed schedule) and scattering all of its outgoing roles."""
    if stage.kind == "disc":
        g = arena.groups[stage.group]
        rows = stage.rows
        gather, scatter = g.gather[rows], g.scatter[rows]
        k_local = max(g.cards)

        def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
            inc = _disc_rows(arena, messages, gather, g.cards)
            logf = params.get(g.name)
            if logf is not None:
                logf = logf[rows]
            out = BP_UPDATES[g.type](inc, logf, g.cards, k_local, op)
            out = jnp.maximum(out, NEG)
            messages = _set_disc_rows(arena, messages, scatter, g.cards, out)
            return messages, g_eta, g_lam, ef_eta

        return fn

    if stage.kind == "ef":
        g = arena.ef_groups[stage.group]
        rows = stage.rows
        gather, scatter = g.gather[rows], g.scatter[rows]
        obs_rows = g.obs_rows[rows] if g.obs_rows is not None else None

        def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
            obs = ef_obs[obs_rows] if obs_rows is not None else None
            mask = ef_mask[obs_rows] if obs_rows is not None else None
            out = _ef_group_out(g, ef_eta[gather], params, rows, obs, mask)
            out = out.astype(ef_eta.dtype)
            ef_eta = ef_eta.at[scatter.reshape(-1)].set(
                out.reshape(-1, out.shape[-1])
            )
            return messages, g_eta, g_lam, ef_eta

        return fn

    g = arena.gauss_groups[stage.group]
    rows = stage.rows
    gather, scatter = g.gather[rows], g.scatter[rows]
    obs_rows = g.obs_rows[rows] if g.obs_rows is not None else None

    def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
        inc_e, inc_l = _gauss_rows(arena, g_eta, g_lam, gather, g.dims)
        y = mk = None
        if g.type == "gaussian_linear_obs":
            y, mk = _obs_rows(arena, g_obs, obs_rows, g.dims[0]), g_mask[obs_rows]
        outs = _gauss_outs(g, inc_e, inc_l, _gauss_group_params(g, params, rows), y, mk)
        for r, (oe, ol) in enumerate(outs):
            g_eta, g_lam = _set_gauss_block(
                arena, g_eta, g_lam, scatter[:, r], g.dims[r], oe, ol
            )
        return messages, g_eta, g_lam, ef_eta

    return fn


def _scan_group_of(arena: Arena, elem):
    return {
        "disc": arena.groups,
        "gauss": arena.gauss_groups,
        "ef": arena.ef_groups,
    }[elem.kind][elem.group]


def _classify_scan(arena: Arena, stage):
    """Registerization analysis for a ScanStage (cff-d56d).

    Per (elem, role, member-column) gather, find where its slots come from
    *within* the stage, uniformly across all steps: written by elem ``we``
    role ``wr`` column ``wc`` in the previous step (d=1, a carried
    register), the same step (d=0: a register when the writer runs earlier,
    otherwise a pre-stage read), or the next step (d=-1: always a pre-stage
    read — the write comes after every read of that slot, and each directed
    slot is written at most once per pass). Columns whose slots the stage
    never writes are pre-stage reads too ("static").

    Returns per-elem per-role plans compressed into three shapes — whole
    role static ("table"), whole role one aligned register ("direct"), or
    per-column mixed ("mix") — or None when any column's dependency is
    non-uniform, in which case the flat-carry executor stays correct."""
    elems = stage.elems
    # Slot numberings are independent per arena (disc/gauss/ef), so reader
    # and writer tables only compare within the same kind.
    writers = []
    written_by_kind: dict[str, set[int]] = {}
    for ei, elem in enumerate(elems):
        sc = _scan_group_of(arena, elem).scatter[elem.rows]  # [S, m, R]
        for r in range(sc.shape[2]):
            writers.append((ei, r, sc[:, :, r]))
            written_by_kind.setdefault(elem.kind, set()).update(
                int(s) for s in sc[:, :, r].reshape(-1)
            )

    plans = []
    for ei, elem in enumerate(elems):
        ga = _scan_group_of(arena, elem).gather[elem.rows]  # [S, m, R]
        all_written = written_by_kind.get(elem.kind, set())
        role_plans = []
        for r in range(ga.shape[2]):
            cols = []
            for c in range(ga.shape[1]):
                tab = ga[:, c, r]  # [S]
                match = None
                for we, wr, wt in writers:
                    if elems[we].kind != elem.kind:
                        continue
                    for wc in range(wt.shape[1]):
                        col = wt[:, wc]
                        if stage.length > 1 and np.array_equal(col[:-1], tab[1:]):
                            match = ("r", we, wr, wc, 1)
                        elif np.array_equal(col, tab):
                            match = ("r", we, wr, wc, 0) if we < ei else ("s",)
                        elif stage.length > 1 and np.array_equal(col[1:], tab[:-1]):
                            # Written only at the *next* step — every read
                            # precedes the write, so the pre-stage value is
                            # correct, provided the final step's slot (outside
                            # the matched pattern) is not written at all.
                            if int(tab[-1]) not in all_written:
                                match = ("s",)
                        if match:
                            break
                    if match:
                        break
                if match is None:
                    if {int(s) for s in tab.reshape(-1)} & all_written:
                        return None  # non-uniform in-stage dependency
                    match = ("s",)
                cols.append(match)
            if all(p == ("s",) for p in cols):
                role_plans.append(("table",))
                continue
            direct = (
                all(p[0] == "r" for p in cols)
                and len({(p[1], p[2], p[4]) for p in cols}) == 1
                and [p[3] for p in cols] == list(range(len(cols)))
            )
            if direct:
                direct = elems[cols[0][1]].rows.shape[1] == len(cols)
            if direct:
                role_plans.append(("direct", cols[0][1], cols[0][2], cols[0][4]))
            else:
                role_plans.append(("mix", tuple(cols)))
        plans.append(role_plans)
    return plans


def _registerized_scan_fn(arena: Arena, stage, plans, op):
    """Registerized executor: the scan carries only the small message blocks
    that cross steps; pre-stage inputs are prefetched [S, ...] tensors,
    outputs are stacked by the scan and scattered into the flat arrays once
    at the end. This is the executor that makes a chain cost what a
    hand-written ``lax.scan`` smoother costs — no dynamic gather/scatter on
    the flat arena inside the loop."""
    elems = stage.elems
    metas = []
    for ei, elem in enumerate(elems):
        g = _scan_group_of(arena, elem)
        ga, sc = g.gather[elem.rows], g.scatter[elem.rows]  # [S, m, R]
        metas.append((elem, g, ga, sc))

    def _col_plans(plan, m):
        if plan[0] == "mix":
            return plan[1]
        if plan[0] == "table":
            return (("s",),) * m
        _, we, wr, d = plan
        return tuple(("r", we, wr, wc, d) for wc in range(m))

    # Carried registers: (we, wr) consumed with d=1 anywhere. Their step-0
    # value is the pre-stage message at each reader's first gather slot;
    # unread entries default to the writer's own step-0 slots (any finite
    # value — they are overwritten before every read).
    reg_init_slots: dict[tuple[int, int], np.ndarray] = {}
    for ei, (elem, g, ga, _) in enumerate(metas):
        for r, plan in enumerate(plans[ei]):
            for c, cp in enumerate(_col_plans(plan, ga.shape[1])):
                if cp[0] == "r" and cp[4] == 1:
                    key = (cp[1], cp[2])
                    if key not in reg_init_slots:
                        we_elem, _, _, we_sc = metas[cp[1]]
                        reg_init_slots[key] = we_sc[0, :, cp[2]].copy()
                    reg_init_slots[key][cp[3]] = ga[0, c, r]

    def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
        xs = []
        for ei, (elem, g, ga, _) in enumerate(metas):
            x = {}
            for r, plan in enumerate(plans[ei]):
                if plan[0] == "direct":
                    continue
                cols = _col_plans(plan, ga.shape[1])
                scols = [c for c, cp in enumerate(cols) if cp[0] == "s"]
                if not scols:
                    continue
                slots = ga[:, scols, r]  # [S, ns]
                if elem.kind == "disc":
                    idx = arena.message_indices(slots, g.cards[r])
                    x[f"s{r}"] = messages[jnp.asarray(idx)]
                elif elem.kind == "ef":
                    x[f"s{r}"] = ef_eta[jnp.asarray(slots)]
                else:
                    d = g.dims[r]
                    x[f"se{r}"] = g_eta[jnp.asarray(arena.g_eta_indices(slots, d))]
                    x[f"sl{r}"] = g_lam[jnp.asarray(arena.g_lam_indices(slots, d))]
            if elem.kind == "disc":
                logf = params.get(g.name)
                if logf is not None:
                    x["lf"] = logf[elem.rows]
            elif elem.kind == "ef":
                kern = kernels.kernel_for(g.type)
                if kern is not None:
                    for key in kern.param_keys:
                        x[f"p{key}"] = params[f"{g.name}:{key}"][elem.rows]
                if g.obs_rows is not None:
                    orows = g.obs_rows[elem.rows]
                    x["ov"] = ef_obs[jnp.asarray(orows)]
                    x["om"] = ef_mask[jnp.asarray(orows)]
            else:
                for k, v in _gauss_group_params(g, params).items():
                    x[f"p{k}"] = v[elem.rows]
                if g.type == "gaussian_linear_obs":
                    orows = g.obs_rows[elem.rows]
                    x["ov"] = g_obs[jnp.asarray(arena.g_obs_indices(orows, g.dims[0]))]
                    x["om"] = g_mask[jnp.asarray(orows)]
            xs.append(x)
        xs = tuple(xs)

        init = {}
        for (we, wr), slots in reg_init_slots.items():
            wg = metas[we][1]
            if elems[we].kind == "disc":
                init[f"{we}:{wr}"] = messages[
                    jnp.asarray(arena.message_indices(slots, wg.cards[wr]))
                ]
            elif elems[we].kind == "ef":
                init[f"{we}:{wr}"] = ef_eta[jnp.asarray(slots)]
            else:
                d = wg.dims[wr]
                init[f"{we}:{wr}:e"] = g_eta[jnp.asarray(arena.g_eta_indices(slots, d))]
                init[f"{we}:{wr}:l"] = g_lam[jnp.asarray(arena.g_lam_indices(slots, d))]

        def body(prev, x_t):
            cur: dict = {}
            ys = {}

            def reg(cp, pair):
                _, we, wr, wc, d = cp
                src = prev if d == 1 else cur
                if pair:
                    return (
                        src[f"{we}:{wr}:e"][wc],
                        src[f"{we}:{wr}:l"][wc],
                    )
                return src[f"{we}:{wr}"][wc]

            def read(ei, r, pair):
                plan = plans[ei][r]
                if plan[0] == "table":
                    if pair:
                        return x_t[ei][f"se{r}"], x_t[ei][f"sl{r}"]
                    return x_t[ei][f"s{r}"]
                if plan[0] == "direct":
                    _, we, wr, d = plan
                    src = prev if d == 1 else cur
                    if pair:
                        return src[f"{we}:{wr}:e"], src[f"{we}:{wr}:l"]
                    return src[f"{we}:{wr}"]
                cols = plan[1]
                si = 0
                pieces = []
                for cp in cols:
                    if cp[0] == "s":
                        if pair:
                            pieces.append((x_t[ei][f"se{r}"][si], x_t[ei][f"sl{r}"][si]))
                        else:
                            pieces.append(x_t[ei][f"s{r}"][si])
                        si += 1
                    else:
                        pieces.append(reg(cp, pair))
                if pair:
                    return (
                        jnp.stack([p[0] for p in pieces]),
                        jnp.stack([p[1] for p in pieces]),
                    )
                return jnp.stack(pieces)

            for ei, (elem, g, _, _) in enumerate(metas):
                if elem.kind == "disc":
                    k_local = max(g.cards)
                    blocks = [
                        jnp.pad(
                            read(ei, r, False),
                            ((0, 0), (0, k_local - card)),
                            constant_values=NEG,
                        )
                        for r, card in enumerate(g.cards)
                    ]
                    inc = jnp.stack(blocks, axis=-2)
                    out = BP_UPDATES[g.type](
                        inc, x_t[ei].get("lf"), g.cards, k_local, op
                    )
                    out = jnp.maximum(out, NEG)
                    for r, card in enumerate(g.cards):
                        block = out[:, r, :card]
                        cur[f"{ei}:{r}"] = block
                        ys[f"{ei}:{r}"] = block
                elif elem.kind == "ef":
                    inc = jnp.stack(
                        [read(ei, r, False) for r in range(len(g.roles))], axis=-2
                    )
                    if g.type == "equality":
                        out = efrules.equality_messages(inc)
                    else:
                        gp = {
                            k[1:]: v for k, v in x_t[ei].items() if k.startswith("p")
                        }
                        out = kernels.KERNELS[g.type].message(
                            inc, gp, x_t[ei].get("ov"), x_t[ei].get("om")
                        )
                    out = out.astype(inc.dtype)
                    for r in range(len(g.roles)):
                        block = out[:, r]
                        cur[f"{ei}:{r}"] = block
                        ys[f"{ei}:{r}"] = block
                else:
                    width = max(g.dims)
                    eb, lb = [], []
                    for r, d in enumerate(g.dims):
                        e, l = read(ei, r, True)
                        eb.append(jnp.pad(e, ((0, 0), (0, width - d))))
                        lb.append(jnp.pad(l, ((0, 0), (0, width - d), (0, width - d))))
                    inc_e, inc_l = jnp.stack(eb, axis=-2), jnp.stack(lb, axis=-3)
                    par = {k[1:]: v for k, v in x_t[ei].items() if k.startswith("p")}
                    y = mk = None
                    if g.type == "gaussian_linear_obs":
                        y, mk = x_t[ei]["ov"], x_t[ei]["om"]
                    outs = _gauss_outs(g, inc_e, inc_l, par, y, mk)
                    for r, (oe, ol) in enumerate(outs):
                        d = g.dims[r]
                        oe, ol = oe[..., :d], ol[..., :d, :d]
                        cur[f"{ei}:{r}:e"], cur[f"{ei}:{r}:l"] = oe, ol
                        ys[f"{ei}:{r}:e"], ys[f"{ei}:{r}:l"] = oe, ol
            new_carry = {k: cur[k] for k in prev}
            return new_carry, ys

        _, ys = jax.lax.scan(body, init, xs)

        for ei, (elem, g, _, sc) in enumerate(metas):
            if elem.kind == "disc":
                for r, card in enumerate(g.cards):
                    idx = jnp.asarray(arena.message_indices(sc[:, :, r], card))
                    messages = messages.at[idx].set(ys[f"{ei}:{r}"])
            elif elem.kind == "ef":
                for r in range(len(g.roles)):
                    ef_eta = ef_eta.at[jnp.asarray(sc[:, :, r])].set(ys[f"{ei}:{r}"])
            else:
                for r, d in enumerate(g.dims):
                    ei_idx = jnp.asarray(arena.g_eta_indices(sc[:, :, r], d))
                    li_idx = jnp.asarray(arena.g_lam_indices(sc[:, :, r], d))
                    g_eta = g_eta.at[ei_idx].set(ys[f"{ei}:{r}:e"])
                    g_lam = g_lam.at[li_idx].set(ys[f"{ei}:{r}:l"])
        return messages, g_eta, g_lam, ef_eta

    return fn


def _scan_stage_fn(arena: Arena, stage, op):
    """Pick the ScanStage executor: registerized when the dependency
    analysis succeeds (chains always do), flat-carry otherwise."""
    plans = _classify_scan(arena, stage)
    if plans is not None:
        fn = _registerized_scan_fn(arena, stage, plans, op)
        fn.registerized = True
        return fn
    fn = _flat_scan_fn(arena, stage, op)
    fn.registerized = False
    return fn


def _flat_scan_fn(arena: Arena, stage, op):
    """Fallback ScanStage executor: S structurally identical rounds as one
    ``lax.scan`` over the flat arrays. All flat value indices are
    precomputed [S, ...] tables fed through the scan inputs, so
    trace/compile cost is O(1) in S while the executed work stays the same
    as the unrolled rounds."""
    metas, xs = [], []
    for elem in stage.elems:
        if elem.kind == "disc":
            g = arena.groups[elem.group]
            slots_g, slots_s = g.gather[elem.rows], g.scatter[elem.rows]  # [S, m, R]
            x = {
                "rows": jnp.asarray(elem.rows),
                "gv": tuple(
                    jnp.asarray(arena.message_indices(slots_g[:, :, r], card))
                    for r, card in enumerate(g.cards)
                ),
                "sv": tuple(
                    jnp.asarray(arena.message_indices(slots_s[:, :, r], card))
                    for r, card in enumerate(g.cards)
                ),
            }
        elif elem.kind == "ef":
            g = arena.ef_groups[elem.group]
            x = {
                "rows": jnp.asarray(elem.rows),
                "gs": jnp.asarray(g.gather[elem.rows]),  # [S, m, R] ef slots
                "ss": jnp.asarray(g.scatter[elem.rows]),
            }
            if g.obs_rows is not None:
                x["or"] = jnp.asarray(g.obs_rows[elem.rows])
        else:
            g = arena.gauss_groups[elem.group]
            slots_g, slots_s = g.gather[elem.rows], g.scatter[elem.rows]
            x = {
                "rows": jnp.asarray(elem.rows),
                "ge": tuple(
                    jnp.asarray(arena.g_eta_indices(slots_g[:, :, r], d))
                    for r, d in enumerate(g.dims)
                ),
                "gl": tuple(
                    jnp.asarray(arena.g_lam_indices(slots_g[:, :, r], d))
                    for r, d in enumerate(g.dims)
                ),
                "se": tuple(
                    jnp.asarray(arena.g_eta_indices(slots_s[:, :, r], d))
                    for r, d in enumerate(g.dims)
                ),
                "sl": tuple(
                    jnp.asarray(arena.g_lam_indices(slots_s[:, :, r], d))
                    for r, d in enumerate(g.dims)
                ),
            }
            if g.type == "gaussian_linear_obs":
                orows = g.obs_rows[elem.rows]  # [S, m]
                x["ov"] = jnp.asarray(arena.g_obs_indices(orows, g.dims[0]))
                x["mr"] = jnp.asarray(orows)
        metas.append((elem.kind, g))
        xs.append(x)
    xs = tuple(xs)

    def fn(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
        def body(carry, x_t):
            messages, g_eta, g_lam, ef_eta = carry
            for (kind, g), x in zip(metas, x_t):
                if kind == "disc":
                    k_local = max(g.cards)
                    blocks = [
                        jnp.pad(
                            messages[x["gv"][r]], ((0, 0), (0, k_local - card)),
                            constant_values=NEG,
                        )
                        for r, card in enumerate(g.cards)
                    ]
                    inc = jnp.stack(blocks, axis=-2)
                    logf = params.get(g.name)
                    if logf is not None:
                        logf = logf[x["rows"]]
                    out = BP_UPDATES[g.type](inc, logf, g.cards, k_local, op)
                    out = jnp.maximum(out, NEG)
                    for r, card in enumerate(g.cards):
                        messages = messages.at[x["sv"][r]].set(out[:, r, :card])
                elif kind == "ef":
                    obs = ef_obs[x["or"]] if "or" in x else None
                    mask = ef_mask[x["or"]] if "or" in x else None
                    out = _ef_group_out(
                        g, ef_eta[x["gs"]], params, x["rows"], obs, mask
                    ).astype(ef_eta.dtype)
                    ef_eta = ef_eta.at[x["ss"]].set(out)
                else:
                    width = max(g.dims)
                    eb = [
                        jnp.pad(g_eta[x["ge"][r]], ((0, 0), (0, width - d)))
                        for r, d in enumerate(g.dims)
                    ]
                    lb = [
                        jnp.pad(
                            g_lam[x["gl"][r]],
                            ((0, 0), (0, width - d), (0, width - d)),
                        )
                        for r, d in enumerate(g.dims)
                    ]
                    inc_e, inc_l = jnp.stack(eb, axis=-2), jnp.stack(lb, axis=-3)
                    par = _gauss_group_params(g, params, x["rows"])
                    y = mk = None
                    if g.type == "gaussian_linear_obs":
                        y, mk = g_obs[x["ov"]], g_mask[x["mr"]]
                    outs = _gauss_outs(g, inc_e, inc_l, par, y, mk)
                    for r, (oe, ol) in enumerate(outs):
                        d = g.dims[r]
                        g_eta = g_eta.at[x["se"][r]].set(oe[..., :d])
                        g_lam = g_lam.at[x["sl"][r]].set(ol[..., :d, :d])
            return (messages, g_eta, g_lam, ef_eta), None

        (messages, g_eta, g_lam, ef_eta), _ = jax.lax.scan(
            body, (messages, g_eta, g_lam, ef_eta), xs
        )
        return messages, g_eta, g_lam, ef_eta

    return fn


def _make_support_check(arena: Arena):
    """Jitted mirror of ``Engine._has_zero_support`` for constraint-free
    models (every two-pass model): True iff an edge or factor belief has
    empty exact support. Semantics identical to the numpy check — exp of the
    NEG log-zero underflows to exactly 0 in either backend."""
    if arena.constraint_groups:  # combinatorial feasibility is host-side
        raise ValueError("jitted support check requires a constraint-free model")
    by_card: dict[int, list[list[int]]] = {}
    for e in arena.edges.values():
        if not e.dim and not e.ef:
            by_card.setdefault(e.card, []).append([e.slot0, e.slot1])
    edge_buckets = [
        (card, np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
        for card, pairs in sorted(by_card.items())
    ]

    def check(messages, params):
        bad = jnp.asarray(False)
        for card, pairs in edge_buckets:
            lq = (
                messages[arena.message_indices(pairs[:, 0], card)]
                + messages[arena.message_indices(pairs[:, 1], card)]
            )
            bad |= ~jnp.all(jnp.any(jnp.exp(lq) > 0, axis=-1))
        for group in arena.groups:
            supports = [
                jnp.exp(messages[arena.message_indices(group.gather[:, r], card)]) > 0
                for r, card in enumerate(group.cards)
            ]
            n = len(group.members)
            if group.type == "equality":
                joint = supports[0]
                for s in supports[1:]:
                    joint = joint & s
                bad |= ~jnp.all(jnp.any(joint, axis=-1))
                continue
            factor_support = jnp.exp(params[group.name]) > 0
            for role, card in enumerate(group.cards):
                shape = [n] + [1] * len(group.cards)
                shape[1 + role] = card
                factor_support = factor_support & supports[role].reshape(shape)
            bad |= ~jnp.all(jnp.any(factor_support.reshape(n, -1), axis=-1))
        return bad

    return check


def _make_exact_unit(arena: Arena, propagate):
    """Propagation + certification sweep + residual, support flags, and
    ledger in one XLA computation: the warm status-bearing run is a single
    dispatch (review continuation (c), 2026-07-27). ``propagate`` is an
    unjitted stage pipeline from :func:`_make_propagate`."""
    raw_step = _make_step(arena)
    raw_bfe = _make_bfe(arena)
    support_bad = _make_support_check(arena)
    is_max = arena.semiring == "max"

    def exact_unit(
        messages, g_eta, g_lam, g_obs, g_mask, ef_eta, ef_obs, ef_mask, params
    ):
        pre_bad = support_bad(messages, params)
        m, e, l, f = propagate(
            messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params
        )
        vm, ve, vl, vf, _, _ = raw_step(
            m, e, l, g_obs, g_mask, f, ef_obs, ef_mask, params, 0.0, False
        )
        delta = jnp.asarray(0.0, dtype=m.dtype)
        for a, b in ((vm, m), (ve, e), (vl, l), (vf, f)):
            if a.size:
                delta = jnp.maximum(delta, jnp.max(jnp.abs(a - b)))
        fe = (
            jnp.asarray(jnp.nan, dtype=m.dtype)
            if is_max
            else raw_bfe(m, e, l, g_obs, g_mask, f, ef_obs, ef_mask, params)
        )
        return m, e, l, f, delta, fe, pre_bad, support_bad(m, params)

    return exact_unit


def _make_propagate(arena: Arena, stages: tuple):
    """Compile a stage list into one jittable propagation function.

    Stages execute in order against the running arrays (asynchronous /
    directed semantics), unlike `_make_step`, which gathers everything from
    the pre-sweep arrays (synchronous flooding). Iterative structures (vmp,
    GFE, constraints, hybrid, prec) never appear as stages — they run in the
    restricted synchronous step of the mixed loop."""
    op = jnp.max if arena.semiring == "max" else logsumexp
    stage_fns = [
        _batch_stage_fn(arena, s, op) if isinstance(s, BatchStage)
        else _scan_stage_fn(arena, s, op)
        for s in stages
    ]

    def propagate(messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params):
        for fn in stage_fns:
            messages, g_eta, g_lam, ef_eta = fn(
                messages, g_eta, g_lam, ef_eta, g_obs, g_mask, ef_obs, ef_mask, params
            )
        return messages, g_eta, g_lam, ef_eta

    propagate.scan_registerized = tuple(
        fn.registerized for fn in stage_fns if hasattr(fn, "registerized")
    )
    return propagate


def _gauss_bfe(arena, g_eta, g_lam, g_obs, g_mask, params):
    """Gaussian node energies + interior-edge differential entropies. Exact
    closed forms (cffg.gauss docstring); with conditioned observations the
    tree identity BFE = −log p(y) holds to fixed-point/ridge precision."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for g in arena.gauss_groups:
        inc_e, inc_l = _gauss_rows(arena, g_eta, g_lam, g.gather, g.dims)
        if g.type == "gaussian":
            (d,) = g.dims
            total = total + gauss.prior_energy(
                inc_e[:, 0, :d], inc_l[:, 0, :d, :d],
                params[g.name + ":m"], params[g.name + ":W"],
            ).sum()
        elif g.type == "equality":
            d = g.dims[0]
            total = total + gauss.equality_energy(inc_e[..., :d], inc_l[..., :d, :d]).sum()
        else:
            do, di = g.dims
            a, b, w = (params[g.name + k] for k in (":A", ":b", ":W"))
            e_i, l_i = inc_e[:, 1, :di], inc_l[:, 1, :di, :di]
            if g.type == "gaussian_linear_obs":
                y, mk = _obs_rows(arena, g_obs, g.obs_rows, do), g_mask[g.obs_rows]
                total = total + gauss.linear_observed_energy(e_i, l_i, y, mk, a, b, w).sum()
            else:
                e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
                total = total + gauss.linear_energy(e_o, l_o, e_i, l_i, a, b, w).sum()
    for d, pairs in arena.gauss_interior:
        l0 = g_lam[arena.g_lam_indices(pairs[:, 0], d)]
        l1 = g_lam[arena.g_lam_indices(pairs[:, 1], d)]
        total = total + gauss.edge_entropy(l0, l1).sum()
    return total


def _mix_bfe(arena, messages, g_eta, g_lam, params):
    """Hybrid Gaussian-mixture node energies (cff-pksr): exact for the held
    labelled-mixture beliefs (mixg energies). The Gaussian and selector edge
    entropies need no code here — the out/in edges land in gauss_interior and
    the selector edge in interior_slot_pairs, both factor-agnostic tables."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for g in arena.mix_groups:
        mu_e = messages[arena.message_indices(g.s_gather, g.k_sel)]
        inc_e, inc_l = _gauss_rows(arena, g_eta, g_lam, g.g_gather, g.dims)
        w = params[g.name + ":W"]
        if g.type == "gaussian_mixture_linear":
            do, di = g.dims
            a, b = params[g.name + ":A"], params[g.name + ":b"]
            e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
            e_i, l_i = inc_e[:, 1, :di], inc_l[:, 1, :di, :di]
            total = total + mixg.mixlin_energy(e_o, l_o, e_i, l_i, mu_e, a, b, w).sum()
        else:
            (do,) = g.dims
            m = params[g.name + ":m"]
            e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
            total = total + mixg.mix_energy(e_o, l_o, mu_e, m, w).sum()
    return total


def _prec_bfe(arena, g_eta, g_lam, ef_eta, params):
    """gaussian_prec node energies (cff-7c5o): the mean-field VFE contribution
    −H[q_x] − H[q_τ] − E_q[log f]. gauss.prec_energy supplies −E_q[log f] (with
    the digamma correction that makes the mean-field VFE exact); the two edge
    neg-entropies are added here, exactly as the discrete VMP energies add one
    _neg_entropy per edge (cffg.rules.vmp_energy_matrix_cat). They net against
    the neighbours' −H and the interior +H (gauss_interior / ef_interior) to
    the single mean-field −H[q_i] per variable. Improper marginals ⇒ +inf."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for g in arena.prec_groups:
        tau_q = ef_eta[g.tau_gather_both].sum(axis=1)  # [M, 2]
        x_eta_q = g_eta[arena.g_eta_indices(g.x_gather_both, 1)].sum(axis=1)
        x_lam_q = g_lam[arena.g_lam_indices(g.x_gather_both, 1)].sum(axis=1)
        m = params[g.name + ":m"]
        e = gauss.prec_energy(tau_q, x_eta_q, x_lam_q, m)  # −E_q[log f], +inf if improper
        neg_h_x = gauss._neg_entropy(x_lam_q)  # −H[q_x]
        neg_h_tau = -efrules.GAMMA.entropy(tau_q)  # −H[q_τ]
        ok = (x_lam_q[..., 0, 0] > 0) & efrules.GAMMA.is_proper(tau_q)
        total = total + jnp.where(ok, neg_h_x + neg_h_tau + e, jnp.inf).sum()
    return total


def _ef_bfe(arena, ef_eta, ef_obs, ef_mask, params):
    """EF node energies + interior-edge entropies, all closed form
    (cffg.efrules). Improper beliefs anywhere make the total +inf — loud.
    With conditioned observations the tree identity BFE = −log p(data)
    holds exactly (no ridge: nothing here is iterative or solved)."""
    total = jnp.asarray(0.0, dtype=ef_eta.dtype)
    for g in arena.ef_groups:
        inc = ef_eta[g.gather]  # [M, R, 2]
        if g.type == "equality":
            total = total + efrules.equality_energy(
                efrules.JFAMILIES[g.family], inc
            ).sum()
        else:  # node energy comes from the registered kernel (cff-ri8m)
            kern = kernels.KERNELS[g.type]
            gp = {key: params[f"{g.name}:{key}"] for key in kern.param_keys}
            obs = ef_obs[g.obs_rows] if g.obs_rows is not None else None
            mask = ef_mask[g.obs_rows] if g.obs_rows is not None else None
            total = total + kern.energy(inc, gp, obs, mask).sum()
    for fam_name, pairs in arena.ef_interior:
        fam = efrules.JFAMILIES[fam_name]
        total = total + efrules.edge_entropy(
            fam, ef_eta[pairs[:, 0]], ef_eta[pairs[:, 1]]
        ).sum()
    return total


def _make_bfe(arena: Arena):
    groups = arena.groups

    def bfe(
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        ef_eta: jax.Array,
        ef_obs: jax.Array,
        ef_mask: jax.Array,
        params: dict[str, jax.Array],
    ) -> jax.Array:
        total = jnp.asarray(0.0, dtype=messages.dtype)
        # prec_groups included so the interior-edge entropy tables run even for
        # a prec-only model: _gauss_bfe owns gauss_interior (the out edges),
        # _ef_bfe owns ef_interior (the prec edges), both EdgeInfo-driven.
        if arena.gauss_groups or arena.prec_groups:
            total = total + _gauss_bfe(arena, g_eta, g_lam, g_obs, g_mask, params)
        if arena.mix_groups:
            total = total + _mix_bfe(arena, messages, g_eta, g_lam, params)
        if arena.ef_groups or arena.prec_groups:
            total = total + _ef_bfe(arena, ef_eta, ef_obs, ef_mask, params)
        if arena.prec_groups:
            total = total + _prec_bfe(arena, g_eta, g_lam, ef_eta, params)
        for g in groups:
            logf = params.get(g.name)
            if logf is None:
                logf = jnp.zeros(())  # equality handles its own zero log-f
            if g.mode == "vmp":
                inc = _disc_both(arena, messages, g.gather_both, g.cards)
                total = total + VMP_ENERGIES[g.type](inc, logf, g.cards).sum()
            else:
                inc = _disc_rows(arena, messages, g.gather, g.cards)
                total = total + NODE_ENERGIES[g.type](inc, logf, g.cards).sum()
        for c in arena.constraint_groups:
            inc = _disc_rows(arena, messages, c.gather, c.cards)
            fn = csp.CONSTRAINT_ENERGIES[c.type]
            if c.type in ("exactly_k", "at_most_k"):
                total = total + fn(inc, c.cards, k=c.k)
            elif c.type == "table":
                total = total + fn(inc, c.cards, table=jnp.asarray(c.table))
            else:
                total = total + fn(inc, c.cards)
        for g in arena.gfe_groups:
            _, ki = g.cards
            log_d = messages[arena.message_indices(g.gather_z, ki)]
            a_key = g.name + (":counts" if g.learned else ":A")
            total = total + gfe_energy(
                log_d, params[a_key], params[g.name + ":c"], learned=g.learned
            ).sum()
        for g in arena.gfe_tm_groups:
            ko, ki, ku = g.cards
            md = messages[arena.message_indices(g.gather[:, 0], ko)]
            mi = messages[arena.message_indices(g.gather[:, 1], ki)]
            me = messages[arena.message_indices(g.gather[:, 2], ku)]
            b_key = g.name + (":counts" if g.learned else ":B")
            total = total + gfe_tm_energy(
                mi, me, md, params[b_key], learned=g.learned
            ).sum()
        for card, pairs in arena.interior_slot_groups:
            lq = (
                messages[arena.message_indices(pairs[:, 0], card)]
                + messages[arena.message_indices(pairs[:, 1], card)]
            )
            lq = lq - logsumexp(lq, axis=-1, keepdims=True)
            q = jnp.exp(lq)
            entropy = -jnp.where(q > _TINY, q * lq, 0.0).sum()
            total = total + entropy
        return total

    return bfe
