"""Trust-region phase and functional-coordinate control, separate from the solver."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from cffg._immutable import FrozenArray
from cffg.compiled import NEG
from cffg.outer import OuterArrays, solve_outer
from cffg.outer_kernels import KernelPlanProvider
from cffg.outer_outcome import OuterOutcome
from cffg.outer_structure import OuterStructure
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
            return OuterOutcome(
                state,
                OuterTerminationReason.NONFINITE_EVALUATION,
                inner_code=2,
                inner_residual=float("inf"),
            )
        if self.services.zero_support(state):
            return OuterOutcome(
                state,
                OuterTerminationReason.INFEASIBLE,
                free_energies=(float("inf"),),
            )

        if self.prime is not None and n_iters >= 1:
            msgs, g_eta, g_lam, ef_eta, *_ = self.prime(
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

        if self.services.update_functional is not None:
            state, _, mu3_code, mu3_residual = self.services.update_functional(state)
            if mu3_code:
                return OuterOutcome(
                    state,
                    OuterTerminationReason.INNER_FAILURE,
                    residual=mu3_residual,
                    inner_code=mu3_code,
                    inner_residual=mu3_residual,
                    functional_failure=True,
                )

        all_deltas: list[float] = []
        all_energies: list[float] = []
        total_iterations = accepted = rejected = 0
        outer_map_evaluations = trial_evaluations = 0
        linear_iterations = solver_compilations = 0
        mu3_pending = False
        mu3_failure_code = 0
        mu3_failure_residual = 0.0
        phases: list[tuple[int, list[tuple[int, np.ndarray]]]] = [
            (n_iters - harden_last, [])
        ]
        hard_rows = [
            (card, pairs[row : row + 1])
            for card, pairs in self.hardening
            for row in range(len(pairs))
        ]
        if harden_last:
            if hard_rows:
                phase_count = min(len(hard_rows), harden_last)
                quotient, remainder = divmod(harden_last, phase_count)
                row_groups = np.array_split(np.arange(len(hard_rows)), phase_count)
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

                def reconstruct(
                    arrays: OuterArrays, *, state=state, params=params
                ) -> OuterArrays:
                    if self.reconstruct is None:
                        return arrays
                    messages, g_eta, g_lam, ef_eta, *_ = self.reconstruct(
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
                    return OuterArrays(messages, g_eta, g_lam, ef_eta)

                def outer_map(
                    arrays: OuterArrays,
                    *,
                    reconstruct=reconstruct,
                    state=state,
                    params=params,
                ):
                    arrays = reconstruct(arrays)
                    messages, g_eta, g_lam, ef_eta, inner_r, inner_code, *_ = (
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
                    )

                def as_state(arrays: OuterArrays, *, state=state) -> State:
                    return _replace(
                        state,
                        messages=arrays.messages,
                        g_eta=arrays.g_eta,
                        g_lam=arrays.g_lam,
                        ef_eta=arrays.ef_eta,
                    )

                solved = solve_outer(
                    OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta),
                    self.structure,
                    outer_map,
                    lambda arrays, as_state=as_state, reconstruct=reconstruct: (
                        self.services.free_energy(as_state(reconstruct(arrays)))
                    ),
                    lambda arrays, as_state=as_state, reconstruct=reconstruct: (
                        self.services.zero_support(as_state(reconstruct(arrays)))
                    ),
                    n_iters=remaining,
                    tol=solver_tol,
                    eliminated=self.eliminated,
                    kernel_factory=lambda chart, state=state: self.kernel_provider(
                        chart, state, hard_blocks
                    ),
                    candidate_limit=(
                        0 if self.adaptive_gaussian or self.services.update_functional is not None else 128
                    ),
                    adaptive_gaussian=self.adaptive_gaussian,
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
            return OuterOutcome(state, OuterTerminationReason.ITERATION_BUDGET)
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
        )
