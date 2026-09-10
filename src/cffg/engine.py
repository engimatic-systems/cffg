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
- ``runtime_types`` defines public State and status values.
- ``lifecycle`` and ``mu3_lifecycle`` own state/data and functional preparation.
- ``iteration``, ``exact_strategy`` and ``outer_strategy`` own run control;
  ``run_results`` constructs their public results.
- ``outer_kernels`` owns lazy derivative plans and dynamic runtime context.
- ``ledger`` owns belief and assignment objectives; ``ledger_terms`` owns local terms.
- ``Engine`` composes these owners and retains readouts.
- ``execution.synchronous`` gathers, combines and damps local factor updates.
- ``schedule_program`` owns stage construction and exact certification units;
  ``execution.transport`` owns shared segmented gathers and scatters.
- ``_make_bfe`` is a compatibility alias for the raw ledger factory.

Paired paths:
- ``Engine.step`` calls the shared sweep; ``Engine.run`` delegates ordinary
  iterations, while ``_run_exact`` consumes the program's exact unit.
- Host ``StateChecks`` and the program's JIT support check share exact-support
  semantics; public ``Engine.bfe`` delegates to ``ObjectiveLedger``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cffg import (
    ef,
    efrules,
    gauss,
    hybrid_gfe,
    kernels,
    mixg,
    recurrence,
    run_results,
)
from cffg.compiled import NEG as NEG  # noqa: PLC0414
from cffg.compiled import Arena
from cffg.exact_strategy import ExactRunStrategy
from cffg.execution import (
    parameters as _parameters,
)
from cffg.execution.owners import SCHEDULED_EXECUTORS
from cffg.execution.support import log_support as _jax_log_support
from cffg.execution.synchronous import make_sweep
from cffg.execution.transport import (
    _disc_both,
    _disc_rows,
    _gauss_rows,
    _obs_rows,
)
from cffg.iteration import IterationPolicy, IterationServices, run_iterations
from cffg.kernel_runtime import (
    discrete_potential as _disc_potential,
)
from cffg.kernel_runtime import (
    discrete_readout as _disc_readout,
)
from cffg.kernel_runtime import (
    gaussian_readout as _gauss_readout,
)
from cffg.kernel_runtime import (
    parameters as _kernel_params,
)
from cffg.ledger import ObjectiveLedger, constraint_satisfied
from cffg.ledger import make_belief_energy as _make_bfe
from cffg.lifecycle import StateLifecycle
from cffg.mu3_lifecycle import Mu3Lifecycle, Mu3PlanView, Mu3StateView
from cffg.outer import (
    OuterChart,
    OuterKernelPlan,
)
from cffg.outer import OuterTerminationReason as OuterTerminationReason  # noqa: PLC0414
from cffg.outer_kernels import OuterKernelProvider
from cffg.outer_strategy import OuterServices, OuterTrustRegionStrategy
from cffg.outer_structure import OuterStructure
from cffg.runtime_checks import StateChecks, state_is_finite
from cffg.runtime_types import (
    InferenceError as InferenceError,  # noqa: PLC0414
)
from cffg.runtime_types import (
    Mu3ScoreResult as Mu3ScoreResult,  # noqa: PLC0414
)
from cffg.runtime_types import (
    RunResult as RunResult,  # noqa: PLC0414
)
from cffg.runtime_types import (
    RunStatus as RunStatus,  # noqa: PLC0414
)
from cffg.runtime_types import (
    State as State,  # noqa: PLC0414
)
from cffg.runtime_types import (
    _replace as _replace,  # noqa: PLC0414
)
from cffg.schedule import Schedule, eliminated_slot_sets
from cffg.schedule_program import ScheduleProgram, build_schedule_program


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
        self._ledger = ObjectiveLedger(arena, raw_bfe)
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
        self._outer_kernels = OuterKernelProvider()
        self._outer_kernel_cache = self._outer_kernels.cache
        self._checks = StateChecks(arena)

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

        outcome = run_iterations(
            state,
            self._iteration_program(),
            services=self._iteration_services(),
            policy=self._iteration_policy(state, tol),
            n_iters=n_iters,
            damp=damp,
            harden_last=harden_last,
        )
        ess, log_normalizers, mu3_statuses = self._mu3_result_diagnostics(outcome.state)
        return run_results.finalize_iteration(
            outcome,
            legacy_picard=has_gfe,
            mu3_ess=ess,
            mu3_log_normalizers=log_normalizers,
            mu3_statuses=mu3_statuses,
        )

    def _iteration_services(self) -> IterationServices:
        return IterationServices(
            self._state_is_finite,
            self._has_zero_support,
            self.bfe,
            self.objective,
            self._update_mu3,
        )

    def _iteration_policy(self, state: State, tol: float | None) -> IterationPolicy:
        return IterationPolicy.from_dtype(
            state.messages.dtype,
            semiring=self.arena.semiring,
            allow_initial_undefined_energy=bool(
                self.arena.gauss_groups
                or self.arena.ef_groups
                or self.arena.prec_groups
                or self.arena.recurrence_groups
            ),
            tol=tol,
        )

    def _iteration_program(self) -> ScheduleProgram:
        return replace(
            self._schedule_program, iter_step=self._iter_step, prime=self._prime
        )

    def _outer_kernel_plan(
        self,
        chart: OuterChart,
        state: State,
        hard_blocks: list[tuple[np.ndarray, np.ndarray, jax.Array]],
    ) -> OuterKernelPlan:
        return self._outer_kernels.plan(
            chart, state, hard_blocks,
            iter_step=self._iter_step, reconstruct=self._reconstruct,
        )

    def _run_outer_trust_region(
        self,
        state: State,
        n_iters: int,
        *,
        tol: float | None,
        harden_last: int,
    ) -> RunResult:
        outcome = OuterTrustRegionStrategy(
            structure=OuterStructure.from_arena(self.arena),
            hardening=tuple(self.arena.delta_slot_groups),
            eliminated=self._outer_eliminated,
            kernel_provider=self._outer_kernel_plan,
            iter_step=self._iter_step,
            prime=self._prime,
            reconstruct=self._reconstruct,
            services=OuterServices(
                self._state_is_finite,
                self._has_zero_support,
                self.bfe,
                self._update_mu3 if self.arena.mu3_groups else None,
            ),
        ).run(state, n_iters, tol=tol, harden_last=harden_last)
        ess, log_normalizers, mu3_statuses = self._mu3_result_diagnostics(outcome.state)
        return run_results.finalize_outer(
            outcome,
            mu3_ess=ess,
            mu3_log_normalizers=log_normalizers,
            mu3_statuses=mu3_statuses,
        )

    def _run_exact(self, state: State, tol: float | None) -> RunResult:
        outcome = ExactRunStrategy(
            self._exact_unit, self.arena.semiring, self._state_is_finite, self.objective
        ).run(state, tol)
        return run_results.finalize_exact(outcome)
    @staticmethod
    def _state_is_finite(state: State) -> bool:
        return state_is_finite(state)

    def _has_zero_support(self, state: State) -> bool:
        return self._checks.has_zero_support(state)

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
        return self._ledger.free_energy(
            state,
            evaluate=self._bfe,
            zero_support=self._has_zero_support,
            require_ready=self._require_mu3_ready,
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
        return self._ledger.assignment_energy(self.decode(state), state.params)

    @staticmethod
    def _constraint_ok(c, v: list[int], parameters: _parameters.ConstraintParameters) -> bool:
        return constraint_satisfied(c, v, parameters)


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
