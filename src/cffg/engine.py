"""Public inference facade over immutable plans and focused runtime owners.

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
- ``readouts`` owns representation and plan-specific posterior/statistic adapters.
- ``Engine`` composes these owners behind the existing public API.
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

from dataclasses import replace

import equinox as eqx
import jax
import numpy as np

from cffg import (
    run_results,
)
from cffg.compiled import NEG as NEG  # noqa: PLC0414
from cffg.compiled import Arena
from cffg.exact_strategy import ExactRunStrategy
from cffg.execution import (
    parameters as _parameters,
)
from cffg.execution.owners import SCHEDULED_EXECUTORS
from cffg.execution.synchronous import make_sweep
from cffg.execution.transport import _disc_both as _disc_both  # noqa: PLC0414
from cffg.execution.transport import _disc_rows as _disc_rows  # noqa: PLC0414
from cffg.execution.transport import _gauss_rows as _gauss_rows  # noqa: PLC0414
from cffg.execution.transport import _obs_rows as _obs_rows  # noqa: PLC0414
from cffg.iteration import IterationPolicy, IterationServices, run_iterations
from cffg.execution.failures import FailureIndex
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
from cffg.readouts import discrete as _discrete_readouts
from cffg.readouts import exponential_family as _exponential_family_readouts
from cffg.readouts import gaussian as _gaussian_readouts
from cffg.readouts import hybrid as _hybrid_readouts
from cffg.readouts import parameters as _parameters_readouts
from cffg.readouts import recurrence as _recurrence_readouts
from cffg.readouts import registered as _registered_readouts
from cffg.readouts.types import DirichletMarginal as DirichletMarginal  # noqa: PLC0414
from cffg.readouts.types import EfMarginal as EfMarginal  # noqa: PLC0414
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
        self._failure_names = FailureIndex(arena).names
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

    @property
    def has_adaptive_gaussian(self) -> bool:
        return any(
            g.kernel is not None and g.kernel.binding.iterative
            for g in self.arena.gauss_groups
        )

    @property
    def objective_metadata(self):
        """Per-Gaussian-factor energy conventions, including adaptive surrogates."""
        return self._ledger.metadata

    @property
    def objective_contract(self):
        """Global energy label, including the presence of adaptive Gaussian terms."""
        return self._ledger.contract

    def hybrid_gfe_diagnostics(
        self, state: State
    ) -> dict[str, dict[str, np.ndarray | float | str]]:
        """D5 objective channels and exact-mixture entropy bounds by factor."""
        return _hybrid_readouts.hybrid_gfe_diagnostics(
            self.arena, state,
        )

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
        msgs, g_eta, g_lam, ef_eta, _, inner_code, *_ = self._step(
            state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
            state.ef_eta, state.ef_obs, state.ef_mask, state.params, damp, harden,
        )
        if int(inner_code):
            raise RuntimeError(
                "factor-local inference diverged; use Engine.run for status"
            )
        return _replace(state, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta)

    def _can_run_exact(self, n_iters: int, damp: float, harden_last: int) -> bool:
        return (
            self._propagate is not None
            and n_iters >= 1
            and damp == 0.0
            and (harden_last == 0 or not len(self.arena.delta_pairs))
        )

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
        k sweeps so decisions commit sequentially. Without delta constraints,
        a valid ``harden_last`` request has no effect.

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
        if not len(self.arena.delta_pairs):
            harden_last = 0
        if self._can_run_exact(n_iters, damp, harden_last):
            return self._run_exact(state, tol)
        has_gfe = bool(
            self.arena.gfe_groups
            or self.arena.gfe_tm_groups
            or self.arena.hybrid_gfe_groups
        )
        adaptive = self.has_adaptive_gaussian
        if adaptive and outer_solver == "picard":
            raise ValueError("adaptive Gaussian inference requires residual globalization")
        use_trust_region = (has_gfe or adaptive) and outer_solver != "picard"
        if outer_solver == "trust_region" and not (has_gfe or adaptive):
            raise ValueError("outer_solver='trust_region' requires a GFE or adaptive Gaussian plan")
        if use_trust_region:
            if damp != 0.0:
                raise ValueError(
                    "damp does not apply to trust-region inference; omit it"
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
            factor_names=self._failure_names,
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
            adaptive_gaussian=self.has_adaptive_gaussian,
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
            factor_names=self._failure_names,
            mu3_ess=ess,
            mu3_log_normalizers=log_normalizers,
            mu3_statuses=mu3_statuses,
        )

    def _run_exact(self, state: State, tol: float | None) -> RunResult:
        outcome = ExactRunStrategy(
            self._exact_unit, self.arena.semiring, self._state_is_finite, self.objective
        ).run(state, tol)
        return run_results.finalize_exact(outcome, factor_names=self._failure_names)
    @staticmethod
    def _state_is_finite(state: State) -> bool:
        return state_is_finite(state)

    def _has_zero_support(self, state: State) -> bool:
        return self._checks.has_zero_support(state)

    # -- readouts ---------------------------------------------------------

    def marginals(self, state: State) -> dict[str, np.ndarray]:
        return _discrete_readouts.marginals(
            self.arena, state,
        )

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
        return _discrete_readouts.node_beliefs(
            self.arena, state,
        )

    def dirichlet_marginals(
        self, state: State
    ) -> dict[str, "DirichletMarginal"]:
        """Parameter-edge name -> proper Dirichlet posterior readout.

        Concentrations are reconstructed from the two packed edge messages:
        the prior's positive ``alpha0`` plus the likelihood's nonnegative
        fractional counts.  Improper pre-sweep state is reported with
        ``proper=False`` and NaN moments rather than resembling a posterior.
        """
        return _parameters_readouts.dirichlet_marginals(
            self.arena, state,
        )

    def goal_marginals(
        self, state: State
    ) -> dict[str, "DirichletMarginal"]:
        """Learned goal-factor name -> its Dirichlet posterior."""
        return _parameters_readouts.goal_marginals(
            self.arena, state, parameter_beliefs=self.dirichlet_marginals,
        )

    def goal_prior_updates(self, state: State) -> dict[str, np.ndarray]:
        """Return converged goal posteriors keyed by serialized prior blob.

        This dictionary is the explicit trial-rollover/checkpoint boundary.
        Reading it does not mutate this state or a future trial's prior.
        """
        return _parameters_readouts.goal_prior_updates(
            self.arena, state, parameter_beliefs=self.dirichlet_marginals,
        )

    def with_goal_priors(
        self, state: State, priors: dict[str, np.ndarray]
    ) -> State:
        """Install trial priors on a fresh state without accumulating sweeps.

        Counts already present on a goal parameter edge prove that the state
        has participated in inference. Callers must start the next trial from
        ``init_state`` rather than add a posterior to an old count message.
        """
        return self._state_lifecycle.with_goal_priors(state, priors)

    def gaussian_marginals(self, state: State) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Edge name -> (mean, cov). Requires a proper combined precision —
        run to coherence (and bind data) first. Improper beliefs return NaN
        mean/cov arrays, an explicit readout that cannot resemble a valid
        negative or huge covariance."""
        return _gaussian_readouts.gaussian_marginals(
            self.arena, state,
        )

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
        return _recurrence_readouts.recurrence_pair_beliefs(
            self.arena, state,
        )

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
        return _gaussian_readouts.gaussian_node_stats(
            self.arena, state, pair_beliefs=self.recurrence_pair_beliefs,
        )

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
        return _recurrence_readouts.recurrence_weight_stats(
            self.arena, state, pair_beliefs=self.recurrence_pair_beliefs,
        )

    def gaussian_mixture_predictives(
        self, state: State
    ) -> dict[str, dict[str, np.ndarray]]:
        """Factor-to-output cavity predictives for Gaussian-mixture nodes.

        Each record exposes selector weights, component moments, collapsed
        moments, and ``proper``. The readout deliberately excludes the
        output-facing message and therefore remains suitable for one-step
        prediction after an observation has updated that edge.
        """
        return _gaussian_readouts.gaussian_mixture_predictives(
            self.arena, state,
        )

    def ef_marginals(self, state: State) -> dict[str, "EfMarginal"]:
        """Edge name -> EF marginal in standard parameters. Observation-family
        edges (poisson/bernoulli DATA ports) carry no messages and are
        skipped. An improper combined belief (e.g. before any sweep) reports
        proper=False with NaN moments — check it before trusting `mean`."""
        return _exponential_family_readouts.ef_marginals(
            self.arena, state,
        )

    def kernel_readouts(self, state: State) -> dict[str, dict[str, np.ndarray]]:
        """Factor name -> the optional readout its FactorKernel declares
        (cff-ri8m): posterior summaries beyond the generic edge marginals,
        computed from the same (inc, params, obs, mask) contract as the
        message/energy callables. Kernels without a readout are skipped."""
        return _registered_readouts.kernel_readouts(
            self.arena, state,
        )

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
        return _discrete_readouts.decode(
            state, marginals=self.marginals,
        )

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
