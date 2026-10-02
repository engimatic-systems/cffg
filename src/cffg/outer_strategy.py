"""Trust-region phase and functional-coordinate control, separate from the solver."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from cffg._immutable import FrozenArray
from cffg.compiled import NEG
from cffg.failure_facts import (
    FailurePhase, empty_failure, empty_failure_at, first_failure, in_phase,
)
from cffg.outer import OuterArrays, solve_outer
from cffg.outer_evidence import offset_failure, terminal_failure
from cffg.outer_kernels import KernelPlanProvider
from cffg.outer_outcome import OuterOutcome
from cffg.outer_structure import OuterStructure
from cffg.outer_startup import run_startup
from cffg.execution.startup import StartupProgram
from cffg.adaptive_runtime import certify_adaptive_proposal, try_adaptive_proposals
from cffg.runtime_types import OuterTerminationReason, State, _replace
from cffg.schedule_contracts import Propagation, Sweep


@dataclass(frozen=True)
class OuterServices:
    finite: Callable[[State], bool]
    zero_support: Callable[[State], bool]
    free_energy: Callable[[State], float]
    update_functional: Callable[[State], tuple[State, float, int, float]] | None


@dataclass(frozen=True)
class OuterTrustRegionStrategy:
    """Run fixed hard-constraint phases with the existing D7 numerical solver."""

    structure: OuterStructure
    hardening: tuple[tuple[int, FrozenArray], ...]
    eliminated: Mapping[str, frozenset[int]]
    kernel_provider: KernelPlanProvider
    iter_step: Sweep
    prime: Propagation | None
    reconstruct: Propagation | None
    services: OuterServices
    adaptive_gaussian: bool = False
    startup: StartupProgram | None = None

    def run(
        self,
        state: State,
        n_iters: int,
        *,
        tol: float | None,
        harden_last: int,
    ) -> OuterOutcome:
        """Solve each fixed hard-constraint phase with the D7 outer method."""
        dtype = np.asarray(state.messages).dtype
        solver_tol = max(
            64.0 * np.finfo(dtype).eps,
            0.0 if tol is None else float(tol),
        )
        if not self.services.finite(state):
            fact = empty_failure()
            if self.startup is not None and n_iters > 0:
                eligible = self.startup.eligibility(state.g_eta, state.g_lam)
                admitted = tuple(jnp.zeros_like(row) for row in eligible)
                check = self.startup.inspect(state.g_eta, state.g_lam, state.g_obs, state.g_mask,
                                             state.params, eligible, admitted, nonfinite_only=True)
                fact = in_phase(check.failure, FailurePhase.INITIALIZATION)
            return OuterOutcome(
                state,
                OuterTerminationReason.NONFINITE_EVALUATION,
                inner_code=2,
                inner_residual=float("inf"),
                initial_rejection=int(fact.reason) == 0,
                failure=terminal_failure(fact, 0),
            )
        if self.services.zero_support(state):
            return OuterOutcome(
                state,
                OuterTerminationReason.INFEASIBLE,
                free_energies=(float("inf"),),
                initial_rejection=True,
            )

        priming_fact = empty_failure()
        startup_iterations = 0
        startup_deltas = ()
        if self.startup is not None and n_iters >= 1:
            initialized = run_startup(self.startup, self.structure, self.eliminated,
                                      self.services, state, n_iters)
            state = initialized.state
            startup_iterations = initialized.iterations
            startup_deltas = initialized.deltas
            if initialized.terminal is not None:
                return initialized.terminal
        if self.prime is not None and self.startup is None and n_iters >= 1:
            msgs, g_eta, g_lam, ef_eta, *auxiliary = self.prime(
                state.messages,
                state.g_eta,
                state.g_lam,
                state.ef_eta,
                state.g_obs,
                state.g_mask,
                state.ef_obs,
                state.ef_mask,
                state.params,
            )
            state = _replace(
                state, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta
            )
            if auxiliary:
                priming_fact = in_phase(auxiliary[0], FailurePhase.PRIMING)

        if self.services.update_functional is not None:
            state, _, mu3_code, mu3_residual = self.services.update_functional(state)
            if mu3_code:
                return OuterOutcome(
                    state,
                    OuterTerminationReason.INNER_FAILURE,
                    iterations=startup_iterations,
                    message_deltas=startup_deltas,
                    residual=mu3_residual,
                    inner_code=mu3_code,
                    inner_residual=mu3_residual,
                    functional_failure=True,
                    failure=terminal_failure(priming_fact, startup_iterations),
                )

        all_deltas: list[float] = list(startup_deltas)
        all_energies: list[float] = []
        total_iterations = startup_iterations
        accepted = rejected = 0
        outer_map_evaluations = trial_evaluations = 0
        linear_iterations = solver_compilations = 0
        mu3_pending = False
        mu3_failure_code = 0
        mu3_failure_residual = 0.0
        available = n_iters - startup_iterations
        hard_budget = min(harden_last, available)
        phases: list[tuple[int, list[tuple[int, np.ndarray]]]] = [(available - hard_budget, [])]
        hard_rows = [
            (card, pairs[row : row + 1])
            for card, pairs in self.hardening
            for row in range(len(pairs))
        ]
        if hard_budget:
            if hard_rows:
                phase_count = min(len(hard_rows), hard_budget)
                quotient, remainder = divmod(hard_budget, phase_count)
                row_groups = np.array_split(np.arange(len(hard_rows)), phase_count)
                phases.extend(
                    (
                        quotient + (phase < remainder),
                        [hard_rows[int(row)] for row in rows],
                    )
                    for phase, rows in enumerate(row_groups)
                )
            else:
                phases.append((hard_budget, []))
        solved = None
        failure = empty_failure_at()
        hard_blocks: list[tuple[np.ndarray, np.ndarray, jax.Array]] = []
        for budget, new_hard_rows in phases:
            if budget == 0:
                continue
            if new_hard_rows:
                projected = state.messages
                for card, pairs in new_hard_rows:
                    i0 = self.structure.message_indices(pairs[:, 0], card)
                    i1 = self.structure.message_indices(pairs[:, 1], card)
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
                phase = FailurePhase.HARD if hard_blocks else FailurePhase.SOFT

                def reconstruct_with_fact(
                    arrays: OuterArrays, *, state=state, params=params
                ):
                    if self.reconstruct is None:
                        return arrays, empty_failure()
                    messages, g_eta, g_lam, ef_eta, *auxiliary = self.reconstruct(
                        arrays.messages,
                        arrays.g_eta,
                        arrays.g_lam,
                        arrays.ef_eta,
                        state.g_obs,
                        state.g_mask,
                        state.ef_obs,
                        state.ef_mask,
                        params,
                    )
                    fact = auxiliary[0] if auxiliary else empty_failure()
                    return (
                        OuterArrays(messages, g_eta, g_lam, ef_eta),
                        in_phase(fact, FailurePhase.RECONSTRUCTION),
                    )

                def reconstruct(arrays, *, reconstruct_with_fact=reconstruct_with_fact):
                    return reconstruct_with_fact(arrays)[0]

                def outer_map(
                    arrays: OuterArrays,
                    *,
                    reconstruct_with_fact=reconstruct_with_fact,
                    state=state,
                    params=params,
                    phase=phase,
                ):
                    arrays, reconstruction_fact = reconstruct_with_fact(arrays)
                    messages, g_eta, g_lam, ef_eta, inner_r, inner_code, *auxiliary = (
                        self.iter_step(
                            arrays.messages,
                            arrays.g_eta,
                            arrays.g_lam,
                            state.g_obs,
                            state.g_mask,
                            arrays.ef_eta,
                            state.ef_obs,
                            state.ef_mask,
                            params,
                            0.0,
                            False,
                        )
                    )
                    for i0, i1, hot in hard_blocks:
                        messages = messages.at[i0].set(hot).at[i1].set(hot)
                    return (
                        OuterArrays(messages, g_eta, g_lam, ef_eta),
                        inner_r,
                        inner_code,
                        first_failure(
                            reconstruction_fact,
                            in_phase(auxiliary[0] if auxiliary else empty_failure(), phase),
                        ),
                    )

                def as_state(arrays: OuterArrays, *, state=state) -> State:
                    return _replace(
                        state,
                        messages=arrays.messages,
                        g_eta=arrays.g_eta,
                        g_lam=arrays.g_lam,
                        ef_eta=arrays.ef_eta,
                    )

                def energy(arrays):
                    return self.services.free_energy(as_state(reconstruct(arrays)))

                def infeasible(arrays):
                    return self.services.zero_support(as_state(reconstruct(arrays)))

                proposal = None
                if self.adaptive_gaussian and self.services.update_functional is None:
                    proposal = try_adaptive_proposals(
                        self.structure, self.eliminated, self.kernel_provider,
                        state, hard_blocks, remaining, solver_tol,
                    )
                    if proposal is not None:
                        proposal = certify_adaptive_proposal(
                            proposal, outer_map, energy, infeasible, solver_tol,
                        )
                if proposal is not None and proposal.result is None:
                    proposal_arrays, _ = reconstruct_with_fact(proposal.arrays)
                    state = as_state(proposal_arrays)
                solved = proposal.result if proposal is not None else None
                if solved is None:
                    solved = solve_outer(
                        OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta),
                        self.structure,
                        outer_map,
                        energy,
                        infeasible,
                        n_iters=remaining - (proposal.rounds if proposal is not None else 0),
                        tol=solver_tol,
                        eliminated=self.eliminated,
                        kernel_factory=lambda chart, state=state: self.kernel_provider(
                            chart, state, hard_blocks
                        ),
                        candidate_limit=(
                            0 if self.adaptive_gaussian or self.services.update_functional is not None else 128
                        ),
                        adaptive_gaussian=self.adaptive_gaussian,
                        initial_failure=(
                            terminal_failure(priming_fact, 0)
                            if total_iterations == 0 else None
                        ),
                    )
                    if proposal is not None:
                        solved = replace(
                            solved, iterations=solved.iterations + proposal.rounds,
                            message_deltas=(*proposal.message_deltas, *solved.message_deltas),
                            free_energies=(*proposal.free_energies, *solved.free_energies),
                            accepted_steps=solved.accepted_steps + proposal.accepted,
                            rejected_steps=solved.rejected_steps + proposal.rejected,
                            outer_map_evaluations=solved.outer_map_evaluations + proposal.maps,
                            trial_evaluations=solved.trial_evaluations + max(proposal.rounds - 1, 0),
                            solver_compilations=solved.solver_compilations + proposal.compilations,
                            failure=offset_failure(solved.failure, proposal.rounds),
                        )
                final_arrays, reconstruction_fact = reconstruct_with_fact(solved.arrays)
                state = as_state(final_arrays)
                failure = offset_failure(solved.failure, total_iterations)
                if int(failure.fact.reason) == 0:
                    failure = terminal_failure(
                        reconstruction_fact, total_iterations + solved.iterations + 1,
                    )
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
                    solved.reason
                    in (
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
                    self.services.update_functional is None
                    or solved.reason is not OuterTerminationReason.ROOT_CONVERGED
                ):
                    break
                state, mu3_delta, mu3_code, mu3_residual = (
                    self.services.update_functional(state)
                )
                if mu3_code:
                    mu3_failure_code = mu3_code
                    mu3_failure_residual = mu3_residual
                    break
                mu3_pending = mu3_delta > solver_tol
                if mu3_delta <= solver_tol:
                    break
            if mu3_failure_code != 0 or solved.reason in (
                OuterTerminationReason.NONFINITE_EVALUATION,
                OuterTerminationReason.INFEASIBLE,
                OuterTerminationReason.INNER_FAILURE,
            ):
                break

        if solved is None:
            return OuterOutcome(state, OuterTerminationReason.ITERATION_BUDGET,
                                iterations=total_iterations, message_deltas=all_deltas)
        if mu3_failure_code:
            inner_code = mu3_failure_code
            inner_residual = mu3_failure_residual
            residual = mu3_failure_residual
            termination_reason = OuterTerminationReason.INNER_FAILURE
        else:
            inner_code = solved.inner_code
            inner_residual = solved.inner_residual
            residual = (
                abs(all_energies[-1] - all_energies[-2])
                if len(all_energies) >= 2
                else solved.outer_residual
            )
            termination_reason = solved.reason
        return OuterOutcome(
            state=state,
            reason=termination_reason,
            iterations=total_iterations,
            message_deltas=all_deltas,
            free_energies=all_energies,
            residual=residual,
            inner_code=inner_code,
            inner_residual=inner_residual,
            tolerance_requested=tol is not None,
            functional_pending=mu3_pending,
            functional_failure=mu3_failure_code != 0,
            last_solve=solved,
            accepted_steps=accepted,
            rejected_steps=rejected,
            outer_map_evaluations=outer_map_evaluations,
            trial_evaluations=trial_evaluations,
            linear_iterations=linear_iterations,
            solver_compilations=solver_compilations,
            failure=failure,
        )
