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

Module map:
- ``State`` and status results define the runtime boundary.
- ``Engine`` owns state setup, data binding, host control flow, and readouts.
- ``execution.synchronous`` gathers, combines and damps local factor updates.
- ``schedule_program`` owns stage construction and exact certification units;
  ``execution.transport`` owns shared segmented gathers and scatters.
- ``_make_bfe`` owns the ledger supplied to the schedule program.

Paired paths:
- ``Engine.step`` calls the shared sweep; ``Engine.run`` drives flooding or mixed
  iterations, while ``_run_exact`` consumes the program's exact unit.
- Host ``_has_zero_support`` and the program's JIT support check share exact-support
  semantics; public ``Engine.bfe`` delegates to ``_make_bfe``.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cffg import csp, ef, efrules, gauss, hybrid_gfe, kernels, mixg, recurrence
from cffg.compiled import NEG, Arena
from cffg.execution import (
    parameters as _parameters,
)
from cffg.execution.owners import SCHEDULED_EXECUTORS
from cffg.execution.support import log_support as _jax_log_support
from cffg.execution.synchronous import make_sweep
from cffg.lifecycle import StateLifecycle
from cffg.mu3_lifecycle import Mu3Lifecycle, Mu3PlanView, Mu3StateView
from cffg.execution.transport import (
    _disc_both,
    _disc_rows,
    _gauss_rows,
    _obs_rows,
)
from cffg.goal_parameters import shared_goal_energy
from cffg.kernel_runtime import (
    discrete_energy as _disc_energy,
    discrete_potential as _disc_potential,
    discrete_readout as _disc_readout,
    gaussian_energy as _gauss_energy,
    gaussian_readout as _gauss_readout,
    parameters as _kernel_params,
)
from cffg.outer import (
    OuterArrays,
    OuterChart,
    OuterKernelPlan,
    OuterTerminationReason,
    _make_root_diagnostics,
    _make_trust_kernel,
    chart_arrays,
    chart_coordinates,
    outer_chart_signature,
    solve_outer,
)
from cffg.schedule import Schedule, eliminated_slot_sets
from cffg.schedule_program import build_schedule_program
from cffg.runtime_types import (
    InferenceError as InferenceError,  # noqa: PLC0414
    Mu3ScoreResult as Mu3ScoreResult,  # noqa: PLC0414
    RunResult as RunResult,  # noqa: PLC0414
    RunStatus as RunStatus,  # noqa: PLC0414
    State as State,  # noqa: PLC0414
    _replace as _replace,  # noqa: PLC0414
)
from cffg.rules import (
    dirichlet_vmp_energy,
    dirichlet_expected_log,
    gfe_energy,
    gfe_energy_stats,
    gfe_goal_structural_stats,
    gfe_parameter_stats,
    gfe_tm_energy,
)



def _np_log_support(values: np.ndarray) -> np.ndarray:
    """Exact support of log-space values under the reserved NEG sentinel.

    Discrete messages are clamped to NEG after every update, and compiled
    exact-zero factor entries are NEG. Every other finite value is supported,
    regardless of magnitude: exponentiation underflows and a numeric
    threshold would merely move that same false-zero boundary elsewhere.
    """
    values = np.asarray(values)
    return values != np.asarray(NEG, dtype=values.dtype)


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

    def __init__(
        self,
        arena: Arena,
        schedule: str | Schedule = "auto",
        *,
        mu3_samples: int = 256,
        mu3_min_ess_fraction: float = 0.1,
    ):
        if mu3_samples <= 0:
            raise ValueError("mu3_samples must be positive")
        if not 0.0 < mu3_min_ess_fraction <= 1.0:
            raise ValueError("mu3_min_ess_fraction must be in (0, 1]")
        self.arena = arena
        self._state_lifecycle = StateLifecycle(arena)
        for group in arena.groups:
            if group.kernel is not None and arena.semiring not in group.kernel.binding.semirings:
                raise ValueError(f"kernel {group.kernel.impl_id!r} does not support {arena.semiring!r}")
        self.mu3_samples = int(mu3_samples)
        self.mu3_min_ess_fraction = float(mu3_min_ess_fraction)
        self._mu3_lifecycle = None
        if arena.mu3_groups:
            groups = {g.name: g for g in arena.gfe_groups if g.mu3}
            self._mu3_lifecycle = Mu3Lifecycle(
                Mu3PlanView(
                    plans=arena.mu3_groups,
                    moment_groups=tuple(groups),
                    latent_indices={
                        plan.name: arena.message_indices(
                            groups[plan.gfe_name].gather_z[plan.rows], plan.cards[1]
                        )
                        for plan in arena.mu3_groups
                    },
                )
            )
        raw_bfe = _make_bfe(arena)
        self._bfe = eqx.filter_jit(raw_bfe)
        program = build_schedule_program(
            arena, schedule, SCHEDULED_EXECUTORS,
            make_sweep=make_sweep, free_energy=raw_bfe,
        )
        self._schedule_program = program
        self.schedule = program.schedule
        self._step = program.step
        self._iter_step = program.iter_step
        self._propagate = program.propagate
        self._prime = program.prime
        self._reconstruct = program.reconstruct
        self._exact_unit = program.exact_unit
        self.scan_registerized = program.scan_registerized
        self._outer_eliminated: dict[str, frozenset[int]] = (
            eliminated_slot_sets(arena) if self.schedule.kind == "mixed" else {}
        )
        self._outer_kernel_cache: dict[tuple, tuple[object, object]] = {}
        # Discrete edge slot pairs bucketed by cardinality: the zero-support
        # check runs per bucket instead of per edge (Python-loop-free).
        by_card: dict[int, list[list[int]]] = {}
        for e in arena.edges.values():
            if not e.dim and not e.ef and not e.param_shape:
                by_card.setdefault(e.card, []).append([e.slot0, e.slot1])
        self._support_edge_groups = [
            (card, np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
            for card, pairs in sorted(by_card.items())
        ]

    # -- state construction ---------------------------------------------

    def init_state(self) -> State:
        params = self._state_lifecycle.initial_parameters()
        if self._mu3_lifecycle is not None:
            params = self._mu3_lifecycle.initialize_parameters(
                params, samples=self.mu3_samples
            )
        return self._state_lifecycle.init_state(params)

    def _uniform_messages(self) -> jax.Array:
        return self._state_lifecycle.uniform_messages()

    def prepare_mu3(self, state: State, key: jax.Array) -> tuple[State, jax.Array]:
        """Draw one fixed particle set per functional parameter group.

        The supplied key is split exactly once per group and the returned key
        is the only key callers should use for a later refresh. A proposal is
        refused unless every Dirichlet concentration is finite and strictly
        positive, and tied slices must expose the same base posterior.
        """
        if self._mu3_lifecycle is None:
            return state, key
        prepared, next_key = self._mu3_lifecycle.prepare(
            Mu3StateView(state.messages, state.params), key, samples=self.mu3_samples
        )
        return _replace(state, params=prepared.params), next_key

    def _require_mu3_ready(self, state: State) -> None:
        if self._mu3_lifecycle is not None:
            self._mu3_lifecycle.require_ready(Mu3StateView(state.messages, state.params))

    def _update_mu3(self, state: State) -> tuple[State, float, int, float]:
        if self._mu3_lifecycle is None:
            return state, 0.0, 0, 0.0
        updated, delta, code, residual = self._mu3_lifecycle.update(
            Mu3StateView(state.messages, state.params),
            samples=self.mu3_samples,
            min_ess_fraction=self.mu3_min_ess_fraction,
        )
        return _replace(state, params=updated.params), delta, code, residual

    def hybrid_gfe_diagnostics(
        self, state: State
    ) -> dict[str, dict[str, np.ndarray | float | str]]:
        """D5 objective channels and exact-mixture entropy bounds by factor."""
        output: dict[str, dict[str, np.ndarray | float | str]] = {}
        for group in self.arena.hybrid_gfe_groups:
            incoming = state.messages[
                self.arena.message_indices(group.s_gather, group.k_sel)
            ]
            inc_eta, inc_lam = _gauss_rows(
                self.arena,
                state.g_eta,
                state.g_lam,
                group.g_gather,
                group.dims,
            )
            common = (
                state.params[group.name + ":W"],
                state.params[group.name + ":goal_m"],
                state.params[group.name + ":goal_W"],
            )
            if group.type == "gaussian_mixture_linear":
                _, di = group.dims
                diagnostics = hybrid_gfe.linear_diagnostics(
                    incoming,
                    inc_eta[:, 1, :di],
                    inc_lam[:, 1, :di, :di],
                    state.params[group.name + ":A"],
                    state.params[group.name + ":b"],
                    *common,
                )
            else:
                diagnostics = hybrid_gfe.prior_diagnostics(
                    incoming,
                    state.params[group.name + ":m"],
                    *common,
                )
            for row, member in enumerate(group.members):
                code = int(np.asarray(diagnostics.status[row]))
                output[member] = {
                    "goal_factor": group.goal_members[row],
                    "rho": np.asarray(diagnostics.rho[row]),
                    "gfe_lower_bound": float(diagnostics.total[row]),
                    "mi_regime_surrogate": float(
                        diagnostics.mi_regime_surrogate[row]
                    ),
                    "mi_regime_exact_lower": float(
                        diagnostics.mi_regime_lower[row]
                    ),
                    "mi_regime_exact_upper": float(
                        diagnostics.mi_regime_upper[row]
                    ),
                    "mi_state": float(diagnostics.mi_state[row]),
                    "h_regime": float(diagnostics.h_regime[row]),
                    "h_state": float(diagnostics.h_state[row]),
                    "goal_risk": float(diagnostics.goal_risk[row]),
                    "local_residual": float(diagnostics.residual[row]),
                    "status": (
                        RunStatus.CONVERGED.value
                        if code == 0
                        else RunStatus.BUDGET_EXHAUSTED.value
                        if code == 1
                        else RunStatus.DIVERGED.value
                    ),
                    "bound_direction": "reported_gfe_lower_bounds_exact",
                }
        return output

    def mu3_diagnostics(self, state: State) -> dict[str, dict[str, np.ndarray | float | str]]:
        """Per-group weighted moments and stochastic diagnostics."""
        if self._mu3_lifecycle is None:
            return {}
        return self._mu3_lifecycle.diagnostics(Mu3StateView(state.messages, state.params))

    def _mu3_result_diagnostics(
        self, state: State
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[RunStatus, ...]]:
        if self._mu3_lifecycle is None:
            return (), (), ()
        return self._mu3_lifecycle.result_diagnostics(
            Mu3StateView(state.messages, state.params)
        )

    def score_mu3_epochs(
        self,
        state: State,
        *,
        key: jax.Array,
        epochs: int,
        burn_in: int,
        n_iters: int,
        tol: float,
        damp: float = 0.0,
    ) -> Mu3ScoreResult:
        """Average complete FE over independently refreshed particle epochs.

        Every retained epoch must itself converge. Burn-in only chooses which
        converged epoch scores enter the mean; it never certifies convergence.
        """
        if self._mu3_lifecycle is None:
            raise ValueError("score_mu3_epochs requires a compiled mu3 evaluator")
        return self._mu3_lifecycle.score_epochs(
            state,
            prepare=self.prepare_mu3,
            run=self.run,
            key=key,
            epochs=epochs,
            burn_in=burn_in,
            n_iters=n_iters,
            tol=tol,
            damp=damp,
        )

    def bind_data(self, state: State, observations: dict) -> State:
        """Bind DATA ports: discrete ports take an int (None = missing =
        uniform); Gaussian ports take a length-dim vector, EF ports a scalar
        count/flip (None = missing = the likelihood node drops out)."""
        return self._state_lifecycle.bind_data(state, observations)

    # -- execution --------------------------------------------------------

    def step(self, state: State, damp: float = 0.0, harden: bool = True) -> State:
        state, _, mu3_code, _ = self._update_mu3(state)
        if mu3_code:
            outcome = "budget exhausted" if mu3_code == 1 else "diverged"
            raise RuntimeError(f"mu3 importance update {outcome}; use Engine.run for status")
        msgs, g_eta, g_lam, ef_eta, _, inner_code = self._step(
            state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
            state.ef_eta, state.ef_obs, state.ef_mask, state.params, damp, harden,
        )
        if int(inner_code):
            raise RuntimeError(
                "factor-local inference diverged; use Engine.run for status"
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
        outer_solver: str = "auto",
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

        Mixed-GFE plans use the proof-backed trust-region residual solver by
        default. ``damp`` remains a Picard/flooding parameter and is rejected
        for that path; select ``outer_solver="picard"`` explicitly to retain
        the legacy fixed-damping iteration as a diagnostic.

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
        the elementwise dtype-scaled mixed absolute/relative certificate
        (also with ``tol=None`` — the schedule proves the fixed point, so no
        sweep budget applies)."""
        if n_iters < 0:
            raise ValueError("n_iters must be non-negative")
        if outer_solver not in ("auto", "trust_region", "picard"):
            raise ValueError(
                "outer_solver must be 'auto', 'trust_region', or 'picard'"
            )
        self._require_mu3_ready(state)
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
        has_gfe = bool(
            self.arena.gfe_groups
            or self.arena.gfe_tm_groups
            or self.arena.hybrid_gfe_groups
        )
        use_trust_region = has_gfe and outer_solver != "picard"
        if outer_solver == "trust_region" and not has_gfe:
            raise ValueError("outer_solver='trust_region' requires a GFE plan")
        if use_trust_region:
            if damp != 0.0:
                raise ValueError(
                    "damp does not apply to mixed-GFE trust-region inference; "
                    "omit it or select outer_solver='picard' explicitly"
                )
            return self._run_outer_trust_region(
                state, n_iters, tol=tol, harden_last=harden_last
            )

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
            ess, log_normalizers, mu3_statuses = self._mu3_result_diagnostics(state)
            return RunResult(
                state, RunStatus.DIVERGED, 0, (), (), float("inf"),
                RunStatus.DIVERGED, float("inf"),
                mu3_ess=ess,
                mu3_log_normalizers=log_normalizers,
                mu3_statuses=mu3_statuses,
            )
        if self._has_zero_support(state):
            ess, log_normalizers, mu3_statuses = self._mu3_result_diagnostics(state)
            return RunResult(
                state, RunStatus.INFEASIBLE, 0, (), (float("inf"),), float("inf"),
                mu3_ess=ess,
                mu3_log_normalizers=log_normalizers,
                mu3_statuses=mu3_statuses,
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
                st, mu3_delta, mu3_code, mu3_residual = self._update_mu3(st)
                if mu3_code:
                    inner_status = (
                        RunStatus.BUDGET_EXHAUSTED
                        if mu3_code == 1 else RunStatus.DIVERGED
                    )
                    inner_residual = mu3_residual
                    status = inner_status
                    residual = mu3_residual
                    return st, True
                msgs, g_eta, g_lam, ef_eta, inner_r, inner_code = self._iter_step(
                    st.messages, st.g_eta, st.g_lam, st.g_obs, st.g_mask,
                    st.ef_eta, st.ef_obs, st.ef_mask, st.params, damp, harden,
                )
                nxt = _replace(
                    st, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta
                )
                delta = max(
                    [
                        mu3_delta,
                        *(
                        float(jnp.max(jnp.abs(getattr(nxt, f) - getattr(st, f))))
                        for f in ("messages", "g_eta", "g_lam", "ef_eta")
                        if getattr(st, f).size
                        ),
                    ]
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
                    self.arena.gauss_groups
                    or self.arena.ef_groups
                    or self.arena.prec_groups
                    or self.arena.recurrence_groups
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
            and (
                self.arena.gauss_groups
                or self.arena.ef_groups
                or self.arena.prec_groups
                or self.arena.recurrence_groups
            )
        ):
            # A continuous/EF ledger that never became defined is not a
            # usable budget result; it is a numerically failed inference.
            status = RunStatus.DIVERGED
            residual = float("inf")
        ess, log_normalizers, mu3_statuses = self._mu3_result_diagnostics(state)
        return RunResult(
            state=state,
            status=status,
            iterations=len(deltas),
            message_deltas=tuple(deltas),
            free_energies=tuple(energies),
            residual=residual,
            inner_status=inner_status,
            inner_residual=inner_residual,
            mu3_ess=ess,
            mu3_log_normalizers=log_normalizers,
            mu3_statuses=mu3_statuses,
            termination_reason=(
                OuterTerminationReason.LEGACY_PICARD
                if has_gfe else OuterTerminationReason.NOT_RUN
            ),
        )

    def _outer_kernel_plan(
        self,
        chart: OuterChart,
        state: State,
        hard_blocks: list[tuple[np.ndarray, np.ndarray, jax.Array]],
    ) -> OuterKernelPlan:
        """Reuse derivative kernels while keeping every runtime value dynamic."""
        hard_indices = tuple(
            (
                np.asarray(i0).dtype.str,
                np.asarray(i0).shape,
                np.asarray(i0).tobytes(),
                np.asarray(i1).dtype.str,
                np.asarray(i1).shape,
                np.asarray(i1).tobytes(),
            )
            for i0, i1, _ in hard_blocks
        )
        key = outer_chart_signature(chart), hard_indices
        compiled = key not in self._outer_kernel_cache
        if compiled:
            static_indices = tuple((i0, i1) for i0, i1, _ in hard_blocks)

            def residual(value, context):
                (
                    base,
                    g_obs,
                    g_mask,
                    ef_obs,
                    ef_mask,
                    params,
                    hard_values,
                ) = context
                arrays = chart_arrays(chart, value, base)
                if self._reconstruct is not None:
                    messages, g_eta, g_lam, ef_eta = self._reconstruct(
                        arrays.messages,
                        arrays.g_eta,
                        arrays.g_lam,
                        arrays.ef_eta,
                        g_obs,
                        g_mask,
                        ef_obs,
                        ef_mask,
                        params,
                    )
                    arrays = OuterArrays(messages, g_eta, g_lam, ef_eta)
                messages, g_eta, g_lam, ef_eta, _, _ = self._iter_step(
                    arrays.messages,
                    arrays.g_eta,
                    arrays.g_lam,
                    g_obs,
                    g_mask,
                    arrays.ef_eta,
                    ef_obs,
                    ef_mask,
                    params,
                    0.0,
                    False,
                )
                for (i0, i1), hot in zip(static_indices, hard_values):
                    messages = messages.at[i0].set(hot).at[i1].set(hot)
                mapped = OuterArrays(messages, g_eta, g_lam, ef_eta)
                return chart_coordinates(chart, mapped) - value

            self._outer_kernel_cache[key] = (
                _make_trust_kernel(residual, chart.size),
                _make_root_diagnostics(residual),
            )
        trust, root_diagnostics = self._outer_kernel_cache[key]
        context = (
            chart.base,
            state.g_obs,
            state.g_mask,
            state.ef_obs,
            state.ef_mask,
            state.params,
            tuple(hot for _, _, hot in hard_blocks),
        )
        return OuterKernelPlan(trust, root_diagnostics, context, compiled)

    def _run_outer_trust_region(
        self,
        state: State,
        n_iters: int,
        *,
        tol: float | None,
        harden_last: int,
    ) -> RunResult:
        """Solve each fixed hard-constraint phase with the D7 outer method."""
        dtype = np.asarray(state.messages).dtype
        solver_tol = max(
            64.0 * np.finfo(dtype).eps,
            0.0 if tol is None else float(tol),
        )
        if not self._state_is_finite(state):
            ess, log_normalizers, mu3_statuses = self._mu3_result_diagnostics(state)
            return RunResult(
                state, RunStatus.DIVERGED, 0, (), (), float("inf"),
                RunStatus.DIVERGED, float("inf"),
                mu3_ess=ess,
                mu3_log_normalizers=log_normalizers,
                mu3_statuses=mu3_statuses,
                termination_reason=OuterTerminationReason.NONFINITE_EVALUATION,
            )
        if self._has_zero_support(state):
            ess, log_normalizers, mu3_statuses = self._mu3_result_diagnostics(state)
            return RunResult(
                state, RunStatus.INFEASIBLE, 0, (), (float("inf"),),
                float("inf"),
                mu3_ess=ess,
                mu3_log_normalizers=log_normalizers,
                mu3_statuses=mu3_statuses,
                termination_reason=OuterTerminationReason.INFEASIBLE,
            )

        if self._prime is not None and n_iters >= 1:
            msgs, g_eta, g_lam, ef_eta = self._prime(
                state.messages, state.g_eta, state.g_lam, state.ef_eta,
                state.g_obs, state.g_mask, state.ef_obs, state.ef_mask,
                state.params,
            )
            state = _replace(
                state, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta
            )

        if self.arena.mu3_groups:
            state, _, mu3_code, mu3_residual = self._update_mu3(state)
            if mu3_code:
                status = (
                    RunStatus.BUDGET_EXHAUSTED
                    if mu3_code == 1 else RunStatus.DIVERGED
                )
                ess, log_normalizers, mu3_statuses = (
                    self._mu3_result_diagnostics(state)
                )
                return RunResult(
                    state, status, 0, (), (), mu3_residual,
                    status, mu3_residual,
                    mu3_ess=ess,
                    mu3_log_normalizers=log_normalizers,
                    mu3_statuses=mu3_statuses,
                    termination_reason=OuterTerminationReason.INNER_FAILURE,
                )

        all_deltas: list[float] = []
        all_energies: list[float] = []
        total_iterations = accepted = rejected = 0
        outer_map_evaluations = trial_evaluations = 0
        linear_iterations = solver_compilations = 0
        mu3_pending = False
        mu3_failure_status: RunStatus | None = None
        mu3_failure_residual = 0.0
        phases: list[tuple[int, list[tuple[int, np.ndarray]]]] = [
            (n_iters - harden_last, [])
        ]
        hard_rows = [
            (card, pairs[row : row + 1])
            for card, pairs in self.arena.delta_slot_groups
            for row in range(len(pairs))
        ]
        if harden_last:
            if hard_rows:
                phase_count = min(len(hard_rows), harden_last)
                quotient, remainder = divmod(harden_last, phase_count)
                row_groups = np.array_split(
                    np.arange(len(hard_rows)), phase_count
                )
                phases.extend(
                    (
                        quotient + (phase < remainder),
                        [hard_rows[int(row)] for row in rows],
                    )
                    for phase, rows in enumerate(row_groups)
                )
            else:
                phases.append((harden_last, []))
        solved = None
        hard_blocks: list[tuple[np.ndarray, np.ndarray, jax.Array]] = []
        for budget, new_hard_rows in phases:
            if budget == 0:
                continue
            if new_hard_rows:
                projected = state.messages
                for card, pairs in new_hard_rows:
                    i0 = self.arena.message_indices(pairs[:, 0], card)
                    i1 = self.arena.message_indices(pairs[:, 1], card)
                    choice = np.argmax(
                        np.asarray(projected[i0] + projected[i1]), axis=-1
                    )
                    hot = jnp.where(
                        jnp.arange(card)[None, :] == choice[:, None],
                        0.0,
                        NEG,
                    ).astype(projected.dtype)
                    projected = projected.at[i0].set(hot).at[i1].set(hot)
                    hard_blocks.append((i0, i1, hot))
                state = _replace(state, messages=projected)
            remaining = budget
            while remaining > 0:
                params = state.params

                def reconstruct(arrays: OuterArrays) -> OuterArrays:
                    if self._reconstruct is None:
                        return arrays
                    messages, g_eta, g_lam, ef_eta = self._reconstruct(
                        arrays.messages, arrays.g_eta, arrays.g_lam,
                        arrays.ef_eta, state.g_obs, state.g_mask,
                        state.ef_obs, state.ef_mask, params,
                    )
                    return OuterArrays(messages, g_eta, g_lam, ef_eta)

                def outer_map(arrays: OuterArrays):
                    arrays = reconstruct(arrays)
                    messages, g_eta, g_lam, ef_eta, inner_r, inner_code = (
                        self._iter_step(
                            arrays.messages, arrays.g_eta, arrays.g_lam,
                            state.g_obs, state.g_mask, arrays.ef_eta,
                            state.ef_obs, state.ef_mask, params, 0.0, False,
                        )
                    )
                    for i0, i1, hot in hard_blocks:
                        messages = (
                            messages.at[i0].set(hot).at[i1].set(hot)
                        )
                    return (
                        OuterArrays(messages, g_eta, g_lam, ef_eta),
                        inner_r,
                        inner_code,
                    )

                def as_state(arrays: OuterArrays) -> State:
                    return _replace(
                        state,
                        messages=arrays.messages,
                        g_eta=arrays.g_eta,
                        g_lam=arrays.g_lam,
                        ef_eta=arrays.ef_eta,
                    )

                solved = solve_outer(
                    OuterArrays(
                        state.messages, state.g_eta, state.g_lam, state.ef_eta
                    ),
                    self.arena,
                    outer_map,
                    lambda arrays: self.bfe(as_state(reconstruct(arrays))),
                    lambda arrays: self._has_zero_support(
                        as_state(reconstruct(arrays))
                    ),
                    n_iters=remaining,
                    tol=solver_tol,
                    eliminated=self._outer_eliminated,
                    kernel_factory=lambda chart: self._outer_kernel_plan(
                        chart, state, hard_blocks
                    ),
                    candidate_limit=(
                        0 if self.arena.mu3_groups else 128
                    ),
                )
                state = as_state(reconstruct(solved.arrays))
                consumed = max(solved.iterations, 1)
                remaining -= consumed
                total_iterations += consumed
                accepted += solved.accepted_steps
                rejected += solved.rejected_steps
                outer_map_evaluations += solved.outer_map_evaluations
                trial_evaluations += solved.trial_evaluations
                linear_iterations += solved.linear_iterations
                solver_compilations += solved.solver_compilations
                all_deltas.extend(solved.message_deltas)
                all_energies.extend(solved.free_energies)
                if (
                    solved.reason in (
                        OuterTerminationReason.STATIONARY_NONROOT,
                        OuterTerminationReason.STAGNATION,
                        OuterTerminationReason.TRUST_REGION_COLLAPSED,
                    )
                    and remaining > 0
                ):
                    # D remains frozen within each trust solve. A truthful
                    # non-root termination may start a new solve from its
                    # accepted state with a newly frozen local scale.
                    continue
                if (
                    not self.arena.mu3_groups
                    or solved.reason is not OuterTerminationReason.ROOT_CONVERGED
                ):
                    break
                state, mu3_delta, mu3_code, mu3_residual = self._update_mu3(
                    state
                )
                if mu3_code:
                    mu3_failure_status = (
                        RunStatus.BUDGET_EXHAUSTED
                        if mu3_code == 1 else RunStatus.DIVERGED
                    )
                    mu3_failure_residual = mu3_residual
                    break
                mu3_pending = mu3_delta > solver_tol
                if mu3_delta <= solver_tol:
                    break
            if (
                mu3_failure_status is not None
                or solved.reason in (
                    OuterTerminationReason.NONFINITE_EVALUATION,
                    OuterTerminationReason.INFEASIBLE,
                    OuterTerminationReason.INNER_FAILURE,
                )
            ):
                break

        if solved is None:
            ess, log_normalizers, mu3_statuses = self._mu3_result_diagnostics(state)
            return RunResult(
                state, RunStatus.BUDGET_EXHAUSTED, 0, (), (), float("inf"),
                mu3_ess=ess,
                mu3_log_normalizers=log_normalizers,
                mu3_statuses=mu3_statuses,
                termination_reason=OuterTerminationReason.ITERATION_BUDGET,
            )
        if mu3_failure_status is not None:
            status = mu3_failure_status
        elif (
            solved.reason is OuterTerminationReason.ROOT_CONVERGED
            and not mu3_pending
        ):
            status = (
                RunStatus.CONVERGED
                if tol is not None else RunStatus.BUDGET_EXHAUSTED
            )
        elif solved.reason is OuterTerminationReason.INFEASIBLE:
            status = RunStatus.INFEASIBLE
        elif (
            solved.reason is OuterTerminationReason.NONFINITE_EVALUATION
            or (
                solved.reason is OuterTerminationReason.INNER_FAILURE
                and solved.inner_code == 2
            )
            or (
                solved.reason
                is OuterTerminationReason.SUPPORT_RESTART_EXHAUSTED
                and not all_energies
            )
        ):
            status = RunStatus.DIVERGED
        else:
            status = RunStatus.BUDGET_EXHAUSTED
        if mu3_failure_status is not None:
            inner_status = mu3_failure_status
            inner_residual = mu3_failure_residual
            residual = mu3_failure_residual
            termination_reason = OuterTerminationReason.INNER_FAILURE
        else:
            inner_status = (
                RunStatus.CONVERGED
                if solved.inner_code == 0 else
                RunStatus.BUDGET_EXHAUSTED
                if solved.inner_code == 1 else
                RunStatus.DIVERGED
            )
            inner_residual = solved.inner_residual
            residual = (
                abs(all_energies[-1] - all_energies[-2])
                if len(all_energies) >= 2 else solved.outer_residual
            )
            termination_reason = solved.reason
        ess, log_normalizers, mu3_statuses = self._mu3_result_diagnostics(state)
        return RunResult(
            state=state,
            status=status,
            iterations=total_iterations,
            message_deltas=tuple(all_deltas),
            free_energies=tuple(all_energies),
            residual=residual,
            inner_status=inner_status,
            inner_residual=inner_residual,
            mu3_ess=ess,
            mu3_log_normalizers=log_normalizers,
            mu3_statuses=mu3_statuses,
            outer_residual=solved.outer_residual,
            outer_merit=solved.merit,
            outer_gradient_norm=solved.gradient_norm,
            trust_radius=solved.trust_radius,
            accepted_steps=accepted,
            rejected_steps=rejected,
            termination_reason=termination_reason,
            outer_map_evaluations=outer_map_evaluations,
            trial_evaluations=trial_evaluations,
            linear_iterations=linear_iterations,
            solver_compilations=solver_compilations,
        )

    def _run_exact(self, state: State, tol: float | None) -> RunResult:
        """Execute the two-pass directed schedule once and certify it.

        On an acyclic BP graph one inward+outward execution reaches the BP
        fixed point, so there is no sweep loop: the run costs O(E). A
        certification sweep (one synchronous flooding step) then re-derives
        every directed message from the propagated state; its elementwise
        mixed absolute/relative change is the fixed-point residual, so a
        scheduling defect — any stale message — surfaces as a non-converged
        status rather than as silently wrong marginals. The certification
        sweep recomputes each message once from already-final inputs, so a
        genuine fixed point sits at a few ulps of each entry's own
        magnitude, independent of graph diameter and of any other block's
        scale."""

        if not self._state_is_finite(state):
            return RunResult(
                state, RunStatus.DIVERGED, 0, (), (), float("inf"),
                RunStatus.DIVERGED, float("inf"),
            )

        # One dispatch: pre/post support flags, propagation, certification
        # sweep, residual, and ledger execute as a single XLA computation
        # (review item (c)).
        msgs, g_eta, g_lam, ef_eta, delta_dev, rel_dev, fe_dev, pre_bad, post_bad = (
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
        delta, rel = float(delta_dev), float(rel_dev)
        if (
            not np.isfinite(delta)
            or not np.isfinite(rel)
            or not self._state_is_finite(st)
        ):
            return RunResult(st, RunStatus.DIVERGED, 1, (delta,), (), float("inf"))
        if bool(post_bad):
            # The fused ledger value is discarded: Z = 0 means F = +inf, and
            # a finite NEG-sentinel score must never leak (cff-j4mu).
            return RunResult(
                st, RunStatus.INFEASIBLE, 1, (), (float("inf"),), float("inf")
            )

        # Structural certificate: the elementwise mixed absolute/relative
        # residual must sit at recomputation-rounding level. Every entry is
        # compared against its own magnitude, so no block's scale can widen
        # the certificate for another block. A caller-supplied tolerance is
        # additionally honored against the raw residual, mirroring the
        # flooding loop's message-delta criterion.
        eps = float(np.finfo(np.asarray(st.messages).dtype).eps)
        structural = rel <= 64.0 * eps
        by_tol = tol is not None and delta <= max(float(tol), 64.0 * eps)
        status = (
            RunStatus.CONVERGED if structural or by_tol else RunStatus.BUDGET_EXHAUSTED
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

        Support is tested in log space by exact comparison with the reserved
        NEG sentinel, never by exponentiating or applying a magnitude
        threshold. exp() underflows for legitimate small potentials, while a
        threshold only creates another false-zero boundary. For an edge, the
        two facing messages must each support a state; summing first would
        lose which operand (if either) carried NEG.

        Vectorized over each edge-cardinality bucket and each group's member
        stack (cff-d56d): the check runs on every exact-path run, so a
        per-member Python loop would put an O(T) interpreter cost back into
        the O(E) schedule."""
        messages = np.asarray(state.messages)
        for card, pairs in self._support_edge_groups:
            support = (
                _np_log_support(
                    messages[self.arena.message_indices(pairs[:, 0], card)]
                )
                & _np_log_support(
                    messages[self.arena.message_indices(pairs[:, 1], card)]
                )
            )
            if not np.all(np.any(support, axis=-1)):
                return True

        for group in self.arena.groups:
            supports = [  # per role [M, card]: exact support of the incoming
                _np_log_support(
                    messages[self.arena.message_indices(group.gather[:, r], card)]
                )
                for r, card in enumerate(group.cards)
            ]
            n = len(group.members)
            if group.type == "equality":
                joint = np.logical_and.reduce(supports)  # [M, k]
                if not np.all(np.any(joint, axis=-1)):
                    return True
                continue
            factor_support = _np_log_support(_disc_potential(group, _kernel_params(group, state.params)))
            for role, card in enumerate(group.cards):
                shape = [n] + [1] * len(group.cards)
                shape[1 + role] = card
                factor_support = factor_support & supports[role].reshape(shape)
            if not np.all(np.any(factor_support.reshape(n, -1), axis=-1)):
                return True

        for constraint in self.arena.constraint_groups:
            parameters = _parameters.constraint_parameters(constraint, state.params)
            supports = [
                np.flatnonzero(
                    _np_log_support(
                        messages[
                            self.arena.message_indices(constraint.gather[i], card)
                        ]
                    )
                ).tolist()
                for i, card in enumerate(constraint.cards)
            ]
            if any(not support for support in supports):
                return True
            if constraint.type == "exactly_k":
                possible = {0}
                for support in supports:
                    possible = {total + value for total in possible for value in support}
                feasible = int(np.asarray(parameters.threshold)) in possible
            elif constraint.type == "at_most_k":
                feasible = sum(min(support) for support in supports) <= int(
                    np.asarray(parameters.threshold)
                )
            elif constraint.type == "table":
                feasible = any(
                    all(int(row[i]) in supports[i] for i in range(len(supports)))
                    for row in np.asarray(parameters.table)
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
            if e.dim or e.ef or e.param_shape:
                continue  # gaussian_marginals / ef_marginals
            lq = ef.CATEGORICAL.product(
                msgs[self.arena.message_indices(e.slot0, e.card)],
                msgs[self.arena.message_indices(e.slot1, e.card)],
            )
            out[e.name] = ef.CATEGORICAL.mean(lq)
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
            return ef.CATEGORICAL.mean(flat, axis=1).reshape(lb.shape)

        for g in self.arena.groups:
            inc = np.asarray(_disc_rows(self.arena, state.messages, g.gather, g.cards))
            if g.type == "equality":
                k = g.cards[0]
                q = norm_exp(
                    ef.CATEGORICAL.product_axis(inc[:, :, :k], axis=1)
                )
            elif g.mode == "vmp":
                both = np.asarray(_disc_both(self.arena, state.messages, g.gather_both, g.cards))
                q = None
                for r, k in enumerate(g.cards):
                    lq = ef.CATEGORICAL.product(
                        both[:, r, 0, :k], both[:, r, 1, :k]
                    )
                    edge_q = norm_exp(lq).reshape(len(g.members), *([1] * r), k)
                    q = edge_q if q is None else q[..., None] * edge_q
            else:
                lb = np.asarray(_disc_potential(g, _kernel_params(g, state.params))).copy()
                for r, k in enumerate(g.cards):
                    shape = [len(g.members)] + [1] * len(g.cards)
                    shape[1 + r] = k
                    lb = lb + inc[:, r, :k].reshape(shape)
                q = norm_exp(lb)
            for i, name in enumerate(g.members):
                out[name] = q[i]
        for g in self.arena.dirichlet_groups:
            both = np.asarray(
                _disc_both(self.arena, state.messages, g.gather_both, g.cards)
            )
            q = None
            for r, k in enumerate(g.cards):
                lq = both[:, r, 0, :k] + both[:, r, 1, :k]
                edge_q = norm_exp(lq).reshape(
                    len(g.members), *([1] * r), k
                )
                q = edge_q if q is None else q[..., None] * edge_q
            for i, name in enumerate(g.members):
                out[name] = q[i]
        return out

    def dirichlet_marginals(
        self, state: State
    ) -> dict[str, "DirichletMarginal"]:
        """Parameter-edge name -> proper Dirichlet posterior readout.

        Concentrations are reconstructed from the two packed edge messages:
        the prior's positive ``alpha0`` plus the likelihood's nonnegative
        fractional counts.  Improper pre-sweep state is reported with
        ``proper=False`` and NaN moments rather than resembling a posterior.
        """
        out: dict[str, DirichletMarginal] = {}
        messages = np.asarray(state.messages)

        for e in self.arena.edges.values():
            if not e.param_shape:
                continue
            packed = ef.DIRICHLET.posterior(
                messages[self.arena.message_indices(e.slot0, e.card)],
                messages[self.arena.message_indices(e.slot1, e.card)],
            )
            alpha = packed.reshape(e.param_shape)
            proper = bool(
                ef.DIRICHLET.is_proper_concentration(
                    alpha, axis=tuple(range(alpha.ndim))
                )
            )
            if proper:
                mean = ef.DIRICHLET.mean_from_concentration(
                    alpha, axis=e.param_axis
                )
                expected_log = np.asarray(
                    efrules.DIRICHLET.mean_stats_from_concentration(
                        alpha, axis=e.param_axis
                    )
                )
            else:
                mean = np.full_like(alpha, np.nan)
                expected_log = np.full_like(alpha, np.nan)
            out[e.name] = DirichletMarginal(
                alpha=alpha.copy(),
                mean=mean,
                expected_log=expected_log,
                proper=proper,
            )
        return out

    def goal_marginals(
        self, state: State
    ) -> dict[str, "DirichletMarginal"]:
        """Learned goal-factor name -> its Dirichlet posterior."""
        parameter = self.dirichlet_marginals(state)
        out: dict[str, DirichletMarginal] = {}
        for group in self.arena.gfe_groups:
            for goal, edge in zip(group.goal_members, group.goal_param_edges):
                out[goal] = parameter[edge]
        return out

    def goal_prior_updates(self, state: State) -> dict[str, np.ndarray]:
        """Return converged goal posteriors keyed by serialized prior blob.

        This dictionary is the explicit trial-rollover/checkpoint boundary.
        Reading it does not mutate this state or a future trial's prior.
        """
        parameter = self.dirichlet_marginals(state)
        out: dict[str, np.ndarray] = {}
        for plan in self.arena.goal_parameter_plans:
            blob = plan.prior_blob
            alpha = parameter[plan.parameter_edges[0]].alpha
            for edge in plan.parameter_edges:
                marginal = parameter[edge]
                if not marginal.proper:
                    raise ValueError(
                        f"goal prior {blob!r}: posterior is improper; "
                        "rollover requires a completed proper coordinate"
                    )
                if not np.array_equal(alpha, marginal.alpha):
                    raise ValueError(
                        f"goal prior blob {blob!r} has inconsistent posteriors "
                        "across compiled goal parameter edges"
                    )
            out[blob] = alpha.copy()
        return out

    def with_goal_priors(
        self, state: State, priors: dict[str, np.ndarray]
    ) -> State:
        """Install trial priors on a fresh state without accumulating sweeps.

        Counts already present on a goal parameter edge prove that the state
        has participated in inference. Callers must start the next trial from
        ``init_state`` rather than add a posterior to an old count message.
        """
        known = {
            blob
            for group in self.arena.gfe_groups
            for blob in group.goal_prior_blobs
        }
        unknown = sorted(set(priors) - known)
        if unknown:
            raise KeyError(
                f"unknown learned-goal prior blobs {unknown}; have {sorted(known)}"
            )
        params = dict(state.params)
        for group in self.arena.gfe_groups:
            if not group.goal_learned:
                continue
            existing = np.asarray(
                state.messages[
                    self.arena.message_indices(
                        group.goal_count_scatter, group.cards[0]
                    )
                ]
            )
            if np.any(existing != 0):
                raise ValueError(
                    "goal priors can only be installed on a fresh inference "
                    "state; within-trial count messages are replaced per sweep"
                )
            alpha0 = params[group.name + ":c_alpha0"]
            for row, blob in enumerate(group.goal_prior_blobs):
                if blob not in priors:
                    continue
                value = np.asarray(priors[blob])
                if (
                    value.shape != (group.cards[0],)
                    or not np.all(np.isfinite(value))
                    or not np.all(value > 0)
                ):
                    raise ValueError(
                        f"goal prior {blob!r}: expected finite positive shape "
                        f"{(group.cards[0],)}, got {value.shape}"
                    )
                alpha0 = alpha0.at[row].set(
                    jnp.asarray(value, dtype=alpha0.dtype)
                )
            params[group.name + ":c_alpha0"] = alpha0
        return _replace(state, params=params)

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
            eta_q, lam_q = ef.GAUSSIAN.product(
                (
                    eta[self.arena.g_eta_indices(e.slot0, d)],
                    lam[self.arena.g_lam_indices(e.slot0, d)],
                ),
                (
                    eta[self.arena.g_eta_indices(e.slot1, d)],
                    lam[self.arena.g_lam_indices(e.slot1, d)],
                ),
            )
            mean, cov = gauss.moments(eta_q, lam_q)
            out[e.name] = (np.asarray(mean), np.asarray(cov))
        return out

    def recurrence_pair_beliefs(
        self, state: State
    ) -> dict[str, dict[str, np.ndarray]]:
        """Factor name -> normalized structured recurrence pair belief.

        Pair rows are source regimes (the ``in`` role) and columns are
        destination regimes (the ``out`` role). ``pair_support`` records
        literal categorical support independently of floating-point
        probability underflow. The readout also exposes the direct Gaussian
        drive moments and D9.12 masses shared by inference and learning.
        Improper inputs preserve support but return shape-correct NaNs and
        ``proper=False`` for numeric fields.
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        for group in self.arena.recurrence_groups:
            card = group.cards[0]
            destination = state.messages[
                self.arena.message_indices(
                    group.d_gather_both[:, 0, 1], card
                )
            ]
            source = state.messages[
                self.arena.message_indices(
                    group.d_gather_both[:, 1, 1], card
                )
            ]
            destination = jnp.where(
                _jax_log_support(destination), destination, -jnp.inf
            )
            source = jnp.where(
                _jax_log_support(source), source, -jnp.inf
            )
            drive_eta = state.g_eta[
                self.arena.g_eta_indices(
                    group.drive_gather_both, group.dim
                )
            ].sum(axis=1)
            drive_precision = state.g_lam[
                self.arena.g_lam_indices(
                    group.drive_gather_both, group.dim
                )
            ].sum(axis=1)
            beliefs = recurrence.structured_readout(
                destination,
                source,
                drive_eta,
                drive_precision,
                state.params[group.name + ":weights"],
                state.params[group.name + ":offsets"],
            )
            for index, name in enumerate(group.members):
                out[name] = {
                    "pair_probability": np.asarray(
                        beliefs.pair_probability[index]
                    ),
                    "source_probability": np.asarray(
                        beliefs.source_probability[index]
                    ),
                    "destination_probability": np.asarray(
                        beliefs.destination_probability[index]
                    ),
                    "pair_support": np.asarray(beliefs.pair_support[index]),
                    "source_support": np.asarray(
                        beliefs.source_support[index]
                    ),
                    "destination_support": np.asarray(
                        beliefs.destination_support[index]
                    ),
                    "drive_mean": np.asarray(beliefs.drive_mean[index]),
                    "drive_covariance": np.asarray(
                        beliefs.drive_covariance[index]
                    ),
                    "drive_second": np.asarray(
                        beliefs.drive_second[index]
                    ),
                    "omega": np.asarray(beliefs.omega[index]),
                    "exposure": np.asarray(beliefs.exposure[index]),
                    "signed_success": np.asarray(
                        beliefs.signed_success[index]
                    ),
                    "proper": np.asarray(beliefs.proper[index]),
                }
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
        gaussian_mixture_linear: soft ``responsibility`` plus responsibility-
                             weighted ``Syy``, ``Syx``, and ``Sxx``.
        gaussian_mixture: soft ``responsibility``, ``weighted_mean``, and
                             ``weighted_second``.
        recurrence_link: normalized source/destination pair, its marginals and
                             literal support, plus shared D9 masses and direct
                             continuous drive moments.
        Every record includes ``proper``. Improper beliefs or mutated non-PD
        node precisions return ``proper=False`` with NaN sufficient statistics;
        equality nodes are skipped."""
        out: dict[str, dict[str, np.ndarray]] = {}
        for g in self.arena.gauss_groups:
            if g.kernel is None or g.kernel.binding.statistics is None:
                continue
            inc_e, inc_l = _gauss_rows(
                self.arena, state.g_eta, state.g_lam, g.gather, g.dims
            )
            y = mk = None
            if g.obs_rows is not None:
                y = _obs_rows(self.arena, state.g_obs, g.obs_rows, g.observed_dim)
                mk = state.g_mask[jnp.asarray(g.obs_rows)]
            values = _gauss_readout(
                g, inc_e, inc_l, _kernel_params(g, state.params), y, mk,
                statistics=True,
            )
            for i, name in enumerate(g.members):
                out[name] = {key: np.asarray(value)[i] for key, value in values.items()}

        for g in self.arena.mix_groups:
            mu_e = state.messages[
                self.arena.message_indices(g.s_gather, g.k_sel)
            ]
            inc_e, inc_l = _gauss_rows(
                self.arena,
                state.g_eta,
                state.g_lam,
                g.g_gather,
                g.dims,
            )
            w = state.params[g.name + ":W"]
            if g.type == "gaussian_mixture_linear":
                do, di = g.dims
                stats = mixg.mixlin_stats(
                    inc_e[:, 0, :do],
                    inc_l[:, 0, :do, :do],
                    inc_e[:, 1, :di],
                    inc_l[:, 1, :di, :di],
                    mu_e,
                    state.params[g.name + ":A"],
                    state.params[g.name + ":b"],
                    w,
                )
                for i, name in enumerate(g.members):
                    out[name] = {
                        "responsibility": np.asarray(stats.responsibility[i]),
                        "Syy": np.asarray(stats.syy[i]),
                        "Syx": np.asarray(stats.syx[i]),
                        "Sxx": np.asarray(stats.sxx[i]),
                        "proper": np.asarray(stats.proper[i]),
                    }
            else:
                (do,) = g.dims
                stats = mixg.mix_stats(
                    inc_e[:, 0, :do],
                    inc_l[:, 0, :do, :do],
                    mu_e,
                    state.params[g.name + ":m"],
                    w,
                )
                for i, name in enumerate(g.members):
                    out[name] = {
                        "responsibility": np.asarray(stats.responsibility[i]),
                        "weighted_mean": np.asarray(stats.weighted_mean[i]),
                        "weighted_second": np.asarray(
                            stats.weighted_second[i]
                        ),
                        "proper": np.asarray(stats.proper[i]),
                    }

        for name, pair in self.recurrence_pair_beliefs(state).items():
            out[name] = {
                **pair,
                "out_probability": pair["destination_probability"],
                "in_probability": pair["source_probability"],
                "out_support": pair["destination_support"],
                "in_support": pair["source_support"],
            }
        return out

    def recurrence_weight_stats(
        self, state: State
    ) -> dict[str, dict[str, np.ndarray]]:
        """Factor-local PG statistics for between-trial recurrence learning.

        The result is keyed by ``recurrence_link`` factor name. ``linear``
        and ``quadratic`` are pair-derived natural-parameter increments for
        independent Gaussian posteriors over each augmented coefficient
        ``[R_il; r_il]``. Scalar predicate and tolerance fields explain an
        improper row.
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        beliefs = self.recurrence_pair_beliefs(state)
        for group in self.arena.recurrence_groups:
            rows = [beliefs[name] for name in group.members]
            stats = recurrence.weight_stats(
                jnp.stack(
                    [jnp.asarray(row["pair_probability"]) for row in rows]
                ),
                jnp.stack([jnp.asarray(row["drive_mean"]) for row in rows]),
                jnp.stack(
                    [jnp.asarray(row["drive_covariance"]) for row in rows]
                ),
                state.params[group.name + ":weights"],
                state.params[group.name + ":offsets"],
            )
            for index, name in enumerate(group.members):
                out[name] = {
                    "omega": np.asarray(stats.omega[index]),
                    "exposure": np.asarray(stats.exposure[index]),
                    "signed_success": np.asarray(
                        stats.signed_success[index]
                    ),
                    "linear": np.asarray(stats.linear[index]),
                    "quadratic": np.asarray(stats.quadratic[index]),
                    "proper": np.asarray(stats.proper[index]),
                    "probability_ok": np.asarray(
                        stats.probability_ok[index]
                    ),
                    "drive_ok": np.asarray(stats.drive_ok[index]),
                    "parameter_ok": np.asarray(stats.parameter_ok[index]),
                    "second_ok": np.asarray(stats.second_ok[index]),
                    "finite_output_ok": np.asarray(
                        stats.finite_output_ok[index]
                    ),
                    "probability_minimum": np.asarray(
                        stats.probability_minimum[index]
                    ),
                    "probability_sum_error": np.asarray(
                        stats.probability_sum_error[index]
                    ),
                    "probability_tolerance": np.asarray(
                        stats.probability_tolerance[index]
                    ),
                    "drive_min_eigenvalue": np.asarray(
                        stats.drive_min_eigenvalue[index]
                    ),
                    "drive_symmetry_error": np.asarray(
                        stats.drive_symmetry_error[index]
                    ),
                    "drive_tolerance": np.asarray(
                        stats.drive_tolerance[index]
                    ),
                    "logit_second_minimum": np.asarray(
                        stats.logit_second_minimum[index]
                    ),
                    "logit_second_tolerance": np.asarray(
                        stats.logit_second_tolerance[index]
                    ),
                }
        return out

    def gaussian_mixture_predictives(
        self, state: State
    ) -> dict[str, dict[str, np.ndarray]]:
        """Factor-to-output cavity predictives for Gaussian-mixture nodes.

        Each record exposes selector weights, component moments, collapsed
        moments, and ``proper``. The readout deliberately excludes the
        output-facing message and therefore remains suitable for one-step
        prediction after an observation has updated that edge.
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        for g in self.arena.mix_groups:
            mu_e = state.messages[
                self.arena.message_indices(g.s_gather, g.k_sel)
            ]
            w = state.params[g.name + ":W"]
            if g.type == "gaussian_mixture_linear":
                _, di = g.dims
                inc_e, inc_l = _gauss_rows(
                    self.arena,
                    state.g_eta,
                    state.g_lam,
                    g.g_gather,
                    g.dims,
                )
                predictive = mixg.mixlin_predictive(
                    inc_e[:, 1, :di],
                    inc_l[:, 1, :di, :di],
                    mu_e,
                    state.params[g.name + ":A"],
                    state.params[g.name + ":b"],
                    w,
                )
            else:
                predictive = mixg.mix_predictive(
                    mu_e,
                    state.params[g.name + ":m"],
                    w,
                )
            for i, name in enumerate(g.members):
                out[name] = {
                    "weight": np.asarray(predictive.weight[i]),
                    "component_mean": np.asarray(
                        predictive.component_mean[i]
                    ),
                    "component_covariance": np.asarray(
                        predictive.component_covariance[i]
                    ),
                    "mean": np.asarray(predictive.mean[i]),
                    "covariance": np.asarray(predictive.covariance[i]),
                    "proper": np.asarray(predictive.proper[i]),
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
            eta_q = fam.product(eta[e.slot0], eta[e.slot1])
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
        for g in self.arena.groups:
            if g.kernel is None:
                continue
            mode = g.kernel.binding.vmp if g.mode == "vmp" else g.kernel.binding.bp
            if mode.readout is None:
                continue
            inc = (_disc_both(self.arena, state.messages, g.gather_both, g.cards)
                   if g.mode == "vmp" else _disc_rows(self.arena, state.messages, g.gather, g.cards))
            vals = _disc_readout(g, inc, _kernel_params(g, state.params))
            for i, member in enumerate(g.members):
                out[member] = {key: np.asarray(v)[i] for key, v in vals.items()}
        for g in self.arena.gauss_groups:
            if g.kernel is None or g.kernel.binding.readout is None:
                continue
            inc_e, inc_l = _gauss_rows(self.arena, state.g_eta, state.g_lam, g.gather, g.dims)
            y = mk = None
            if g.obs_rows is not None:
                y = _obs_rows(self.arena, state.g_obs, g.obs_rows, g.observed_dim)
                mk = state.g_mask[jnp.asarray(g.obs_rows)]
            vals = _gauss_readout(g, inc_e, inc_l, _kernel_params(g, state.params), y, mk)
            for i, member in enumerate(g.members):
                out[member] = {key: np.asarray(v)[i] for key, v in vals.items()}
        for g in self.arena.ef_groups:
            kern = kernels.kernel_for(g.type)
            if kern is None or kern.readout is None:
                continue
            gp = {key: state.params[f"{g.name}:{key}"] for key in kern.param_keys}
            obs = state.ef_obs[jnp.asarray(g.obs_rows)] if g.obs_rows is not None else None
            mask = state.ef_mask[jnp.asarray(g.obs_rows)] if g.obs_rows is not None else None
            vals = kern.readout(state.ef_eta[np.asarray(g.gather)], gp, obs, mask)
            for i, member in enumerate(g.members):
                out[member] = {key: np.asarray(v)[i] for key, v in vals.items()}
        return out

    def bfe(self, state: State) -> float:
        if self.arena.semiring == "max":
            raise RuntimeError(
                "min-sum reports an objective value, not a free energy — call Engine.objective"
            )
        self._require_mu3_ready(state)
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
            if e.dim or e.ef or e.param_shape:
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
            logf = np.asarray(_disc_potential(g, _kernel_params(g, state.params)))
            for i, v in enumerate(vals):
                total += -float(logf[(i, *v)])
        for c in self.arena.constraint_groups:
            v = [xhat[slot2var[int(c.scatter[j])]] for j in range(len(c.cards))]
            parameters = _parameters.constraint_parameters(c, state.params)
            total += 0.0 if self._constraint_ok(c, v, parameters) else np.inf
        return total

    @staticmethod
    def _constraint_ok(c, v: list[int], parameters: _parameters.ConstraintParameters) -> bool:
        if c.type == "alldifferent":
            return len(set(v)) == len(v)
        if c.type == "exactly_k":
            return sum(v) == int(np.asarray(parameters.threshold))
        if c.type == "at_most_k":
            return sum(v) <= int(np.asarray(parameters.threshold))
        # table: the assignment must be one of the allowed tuples
        return any(np.array_equal(np.asarray(v), row) for row in np.asarray(parameters.table))


@dataclass(frozen=True)
class EfMarginal:
    """A scalar EF edge belief: standard parameters, e.g. (shape, rate) for
    gamma, (a, b) for beta."""

    family: str
    params: tuple[float, ...]
    mean: float
    proper: bool


@dataclass(frozen=True)
class DirichletMarginal:
    """One in-graph table posterior in canonical serialized layout."""

    alpha: np.ndarray
    mean: np.ndarray
    expected_log: np.ndarray
    proper: bool






def _gauss_bfe(arena, g_eta, g_lam, g_obs, g_mask, params):
    """Gaussian node energies + interior-edge differential entropies. Exact
    closed forms (cffg.gauss docstring); with conditioned observations the
    tree identity BFE = −log p(y) holds to fixed-point/ridge precision."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for g in arena.gauss_groups:
        inc_e, inc_l = _gauss_rows(arena, g_eta, g_lam, g.gather, g.dims)
        y = mk = None
        if g.obs_rows is not None:
            y = _obs_rows(arena, g_obs, g.obs_rows, g.observed_dim)
            mk = g_mask[jnp.asarray(g.obs_rows)]
        total = total + _gauss_energy(g, inc_e, inc_l, _kernel_params(g, params), y, mk).sum()
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


def _recurrence_bfe(arena, messages, g_eta, g_lam, params):
    """D9 pair-region energies, upper-bounding exact transition terms."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for group in arena.recurrence_groups:
        card = group.cards[0]
        out_cavity = messages[
            arena.message_indices(group.d_gather_both[:, 0, 1], card)
        ]
        in_cavity = messages[
            arena.message_indices(group.d_gather_both[:, 1, 1], card)
        ]
        out_cavity = jnp.where(
            _jax_log_support(out_cavity), out_cavity, -jnp.inf
        )
        in_cavity = jnp.where(
            _jax_log_support(in_cavity), in_cavity, -jnp.inf
        )
        drive_eta = g_eta[
            arena.g_eta_indices(group.drive_gather_both, group.dim)
        ].sum(axis=1)
        drive_precision = g_lam[
            arena.g_lam_indices(group.drive_gather_both, group.dim)
        ].sum(axis=1)
        total = total + recurrence.structured_recurrence_energy(
            out_cavity,
            in_cavity,
            drive_eta,
            drive_precision,
            params[group.name + ":weights"],
            params[group.name + ":offsets"],
        ).sum()
    return total


def _hybrid_gfe_bfe(arena, messages, g_eta, g_lam, params):
    """Direct D5 G-hat terms; absorbed goals and output entropies stay out."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for group in arena.hybrid_gfe_groups:
        incoming = messages[
            arena.message_indices(group.s_gather, group.k_sel)
        ]
        inc_eta, inc_lam = _gauss_rows(
            arena, g_eta, g_lam, group.g_gather, group.dims
        )
        common = (
            params[group.name + ":W"],
            params[group.name + ":goal_m"],
            params[group.name + ":goal_W"],
        )
        if group.type == "gaussian_mixture_linear":
            _, di = group.dims
            values = hybrid_gfe.linear_energy(
                incoming,
                inc_eta[:, 1, :di],
                inc_lam[:, 1, :di, :di],
                params[group.name + ":A"],
                params[group.name + ":b"],
                *common,
            )
        else:
            values = hybrid_gfe.prior_energy(
                incoming,
                params[group.name + ":m"],
                *common,
            )
        total = total + values.sum()
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
        tau_q = ef_eta[jnp.asarray(g.tau_gather_both)].sum(axis=1)  # [M, 2]
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
        inc = ef_eta[jnp.asarray(g.gather)]  # [M, R, 2]
        if g.type == "equality":
            total = total + efrules.equality_energy(
                efrules.JFAMILIES[g.family], inc
            ).sum()
        else:  # node energy comes from the registered kernel (cff-ri8m)
            kern = kernels.KERNELS[g.type]
            gp = {key: params[f"{g.name}:{key}"] for key in kern.param_keys}
            obs = ef_obs[jnp.asarray(g.obs_rows)] if g.obs_rows is not None else None
            mask = ef_mask[jnp.asarray(g.obs_rows)] if g.obs_rows is not None else None
            total = total + kern.energy(inc, gp, obs, mask).sum()
    for fam_name, pairs in arena.ef_interior:
        fam = efrules.JFAMILIES[fam_name]
        total = total + efrules.edge_entropy(
            fam, ef_eta[jnp.asarray(pairs[:, 0])], ef_eta[jnp.asarray(pairs[:, 1])]
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
        if arena.gauss_groups or arena.prec_groups or arena.recurrence_groups:
            total = total + _gauss_bfe(arena, g_eta, g_lam, g_obs, g_mask, params)
        if arena.mix_groups:
            total = total + _mix_bfe(arena, messages, g_eta, g_lam, params)
        if arena.recurrence_groups:
            total = total + _recurrence_bfe(
                arena, messages, g_eta, g_lam, params
            )
        if arena.hybrid_gfe_groups:
            total = total + _hybrid_gfe_bfe(
                arena, messages, g_eta, g_lam, params
            )
        if arena.ef_groups or arena.prec_groups:
            total = total + _ef_bfe(arena, ef_eta, ef_obs, ef_mask, params)
        if arena.prec_groups:
            total = total + _prec_bfe(arena, g_eta, g_lam, ef_eta, params)
        for g in arena.dirichlet_groups:
            inc = _disc_both(arena, messages, g.gather_both, g.cards)
            packed_alpha = (
                messages[arena.message_indices(g.prior_scatter, g.param_size)]
                + messages[arena.message_indices(g.count_scatter, g.param_size)]
            )
            alpha = packed_alpha.reshape((len(g.members), *g.param_shape))
            total = total + dirichlet_vmp_energy(
                g.type,
                inc,
                alpha,
                params[f"{g.name}:alpha0"],
                g.cards,
            ).sum()
        for g in groups:
            if g.mode == "vmp":
                inc = _disc_both(arena, messages, g.gather_both, g.cards)
            else:
                inc = _disc_rows(arena, messages, g.gather, g.cards)
            total = total + _disc_energy(g, inc, _kernel_params(g, params)).sum()
        for c in arena.constraint_groups:
            inc = _disc_rows(arena, messages, c.gather, c.cards)
            fn = csp.CONSTRAINT_ENERGIES[c.type]
            parameters = _parameters.constraint_parameters(c, params)
            if c.type in ("exactly_k", "at_most_k"):
                total = total + fn(inc, c.cards, k=parameters.threshold)
            elif c.type == "table":
                total = total + fn(inc, c.cards, table=parameters.table)
            else:
                total = total + fn(inc, c.cards)
        goal_predictions = {}
        for g in arena.gfe_groups:
            _, ki = g.cards
            ko, _ = g.cards
            log_d = messages[arena.message_indices(g.gather_z, ki)]
            a_key = g.name + (":counts" if g.learned else ":A")
            if g.goal_learned:
                alpha0 = params[g.name + ":c_alpha0"]
                current_counts = messages[
                    arena.message_indices(g.goal_count_scatter, ko)
                ]
                alpha = efrules.DIRICHLET.posterior(
                    alpha0, current_counts
                )
                expected_log_c = dirichlet_expected_log(alpha)
                abar, hbar = gfe_parameter_stats(
                    params[a_key], learned=g.learned
                )
                structural, predicted = gfe_goal_structural_stats(
                    log_d, abar, hbar, expected_log_c
                )
                total = total + structural.sum()
                goal_predictions[g.name] = predicted
            elif g.mu3:
                total = total + gfe_energy_stats(
                    log_d,
                    params[g.name + ":Abar"],
                    params[g.name + ":hbar"],
                    params[g.name + ":c"],
                ).sum()
            else:
                total = total + gfe_energy(
                    log_d, params[a_key], params[g.name + ":c"], learned=g.learned
                ).sum()
        total = total + shared_goal_energy(arena, messages, params, goal_predictions)
        # The structural terms above contain -E_q[Σ phi_t]. At the coordinate
        # optimum q*(A) ∝ q0(A) exp(Σ phi_t), the shared parameter ledger is
        # KL(q*||q0) = E_q[Σ phi_t] - log Z. Add it exactly once per tied
        # particle group, leaving the complete parameter contribution -log Z.
        for plan in arena.mu3_groups:
            total = total + params[plan.name + ":kl"]
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
            lq = efrules.CATEGORICAL.product(
                messages[arena.message_indices(pairs[:, 0], card)],
                messages[arena.message_indices(pairs[:, 1], card)],
            )
            total = total + efrules.CATEGORICAL.entropy_total(lq)
        return total

    return bfe
