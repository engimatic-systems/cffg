"""Functional parameter preparation, updates, diagnostics and epoch coordination.

Numerical factor calculations stay in execution.functional. Prepared state
views expose only categorical messages and parameters. Epoch coordination
carries full public states opaquely through the supplied prepare/run callbacks.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from cffg import efrules
from cffg._immutable import FrozenArray, FrozenMap, ImmutableRecord
from cffg.compiled import Mu3Plan
from cffg.execution import functional as _functional
from cffg.runtime_types import Mu3ScoreResult, RunResult, RunStatus, State


@dataclass(frozen=True)
class Mu3PlanView(ImmutableRecord):
    """Only functional plans, moment addresses and their latent-message indices."""

    plans: tuple[Mu3Plan, ...]
    moment_groups: tuple[str, ...]
    latent_indices: FrozenMap[str, FrozenArray]


class Mu3StateView(NamedTuple):
    messages: jax.Array
    params: Mapping[str, jax.Array]


class RunEpoch(Protocol):
    def __call__(
        self, state: State, *, n_iters: int, tol: float, harden_last: int
    ) -> RunResult: ...


@dataclass(frozen=True)
class Mu3Lifecycle:
    """Functional host lifecycle over a fixed plan view and current settings."""

    view: Mu3PlanView

    def initialize_parameters(
        self, values: Mapping[str, jax.Array], *, samples: int
    ) -> dict[str, jax.Array]:
        params = dict(values)
        # Moment and particle arrays are derived runtime state. Their keys and
        # shapes are deterministic from the compiled evaluator plan plus the
        # Engine's runtime particle budget, keeping the template as source of
        # truth while preserving a stable JAX pytree for every sweep.
        for name in self.view.moment_groups:
            counts = params[name + ":counts"]
            abar = efrules.DIRICHLET.mean_from_concentration(counts, axis=1)
            e_alog = efrules.DIRICHLET.expected_probability_log_probability(
                counts, axis=1
            )
            params[name + ":Abar"] = abar
            params[name + ":hbar"] = -e_alog.sum(axis=1)
        for plan in self.view.plans:
            ko, ki = plan.cards
            dtype = params[plan.count_key].dtype
            prefix = plan.name
            alpha = params[plan.count_key][int(plan.rows[0])]
            params[prefix + ":samples"] = jnp.zeros((samples, ko, ki), dtype=dtype)
            params[prefix + ":weights"] = jnp.full(
                (samples,), 1.0 / samples, dtype=dtype
            )
            params[prefix + ":base_alpha"] = alpha
            params[prefix + ":log_normalizer"] = jnp.asarray(0.0, dtype=dtype)
            params[prefix + ":expected_phi"] = jnp.asarray(0.0, dtype=dtype)
            params[prefix + ":kl"] = jnp.asarray(0.0, dtype=dtype)
            params[prefix + ":ess"] = jnp.asarray(samples, dtype=dtype)
            params[prefix + ":status"] = jnp.asarray(0)
            params[prefix + ":ready"] = jnp.asarray(False)
        return params

    def prepare(
        self, state: Mu3StateView, key: jax.Array, *, samples: int
    ) -> tuple[Mu3StateView, jax.Array]:
        """Draw one fixed particle set per functional parameter group.

        The supplied key is split exactly once per group and the returned key
        is the only key callers should use for a later refresh.  A proposal is
        refused unless every Dirichlet concentration is finite and strictly
        positive, and tied slices must expose the same base posterior.
        """
        if not self.view.plans:
            return state, key
        params = dict(state.params)
        next_key = key
        for plan in self.view.plans:
            counts = np.asarray(params[plan.count_key])[plan.rows]
            if not np.isfinite(counts).all() or np.any(counts <= 0):
                raise ValueError(
                    f"{plan.name}: mu3 proposal must be a proper Dirichlet "
                    "with finite, strictly positive concentrations"
                )
            dtype = counts.dtype
            tolerance = 64.0 * np.finfo(dtype).eps
            if not np.allclose(counts, counts[0], rtol=tolerance, atol=0.0):
                raise ValueError(
                    f"{plan.name}: slices tied to blob {plan.param_blob!r} "
                    "must share one Dirichlet base posterior"
                )
            alpha = jnp.asarray(counts[0])
            next_key, sample_key = jax.random.split(next_key)
            column_keys = jax.random.split(sample_key, plan.cards[1])
            by_column = jax.vmap(
                lambda column_key, concentration: jax.random.dirichlet(
                    column_key, concentration, shape=(samples,)
                )
            )(column_keys, alpha.T)
            particles = jnp.transpose(by_column, (1, 2, 0))
            sample_sums = np.asarray(particles).sum(axis=1)
            if (
                not np.isfinite(np.asarray(particles)).all()
                or np.any(np.asarray(particles) < 0)
                or not np.allclose(
                    sample_sums,
                    1.0,
                    rtol=64.0 * np.finfo(dtype).eps,
                    atol=64.0 * np.finfo(dtype).eps,
                )
            ):
                raise FloatingPointError(
                    f"{plan.name}: Dirichlet particle generation diverged "
                    f"under dtype {dtype}"
                )

            prefix = plan.name
            params[prefix + ":samples"] = particles
            params[prefix + ":weights"] = jnp.full(
                (samples,), 1.0 / samples, dtype=alpha.dtype
            )
            params[prefix + ":base_alpha"] = alpha
            params[prefix + ":log_normalizer"] = jnp.asarray(0.0, dtype=alpha.dtype)
            params[prefix + ":expected_phi"] = jnp.asarray(0.0, dtype=alpha.dtype)
            params[prefix + ":kl"] = jnp.asarray(0.0, dtype=alpha.dtype)
            params[prefix + ":ess"] = jnp.asarray(samples, dtype=alpha.dtype)
            params[prefix + ":status"] = jnp.asarray(0)
            params[prefix + ":ready"] = jnp.asarray(True)

            group_name = plan.gfe_name
            mean = efrules.DIRICHLET.mean_from_concentration(alpha, axis=0)
            e_alog = efrules.DIRICHLET.expected_probability_log_probability(
                alpha, axis=0
            )
            entropy = -e_alog.sum(axis=0)
            rows = jnp.asarray(plan.rows)
            params[group_name + ":Abar"] = (
                params[group_name + ":Abar"].at[rows].set(mean)
            )
            params[group_name + ":hbar"] = (
                params[group_name + ":hbar"].at[rows].set(entropy)
            )
        return state._replace(params=params), next_key

    def require_ready(self, state: Mu3StateView) -> None:
        missing = [
            plan.name
            for plan in self.view.plans
            if not bool(np.asarray(state.params[plan.name + ":ready"]))
        ]
        if missing:
            raise ValueError(
                "mu3 inference requires a fixed particle set; call "
                f"Engine.prepare_mu3(state, key) first (unprepared: {missing})"
            )

    def update(
        self, state: Mu3StateView, *, samples: int, min_ess_fraction: float
    ) -> tuple[Mu3StateView, float, int, float]:
        """One deterministic coordinate update on the current fixed particles."""
        if not self.view.plans:
            return state, 0.0, 0, 0.0
        self.require_ready(state)
        params = dict(state.params)
        max_delta = 0.0
        overall_code = 0
        overall_residual = 0.0

        for plan in self.view.plans:
            prefix = plan.name
            counts = np.asarray(params[plan.count_key])[plan.rows]
            base = np.asarray(params[prefix + ":base_alpha"])
            dtype = counts.dtype
            tolerance = 64.0 * np.finfo(dtype).eps
            if not np.allclose(
                counts, counts[0], rtol=tolerance, atol=0.0
            ) or not np.allclose(counts[0], base, rtol=tolerance, atol=0.0):
                raise ValueError(
                    f"{plan.name}: the Dirichlet base posterior changed after "
                    "sampling; call prepare_mu3 with a fresh explicit key"
                )

            group_name = plan.gfe_name
            rows = jnp.asarray(plan.rows)
            log_d = state.messages[np.asarray(self.view.latent_indices[plan.name])]
            old_abar = params[group_name + ":Abar"][rows]
            old_hbar = params[group_name + ":hbar"][rows]
            logc = params[group_name + ":c"][rows]
            stationary = _functional.execute_functional_stationary(
                plan, log_d, old_abar, old_hbar, logc
            )
            zbars = stationary.outgoing
            solve_code = int(np.max(np.asarray(stationary.status)))
            solve_residual = float(np.max(np.asarray(stationary.residual)))
            if solve_code:
                params[prefix + ":status"] = jnp.asarray(solve_code)
                overall_code = max(overall_code, solve_code)
                overall_residual = max(overall_residual, solve_residual)
                continue

            particles = params[prefix + ":samples"]
            estimate = _functional.execute_functional_importance(
                plan, particles, zbars, logc, old_abar[0]
            )
            (
                new_abar,
                new_hbar,
                weights,
                log_z,
                expected_phi,
                kl,
                ess,
            ) = estimate.outgoing
            code = int(np.asarray(estimate.status))
            ess_value = float(np.asarray(ess))
            if code == 0 and ess_value < min_ess_fraction * samples:
                code = 1
            params[prefix + ":status"] = jnp.asarray(code)
            params[prefix + ":ess"] = ess
            if code == 2:
                overall_code = 2
                overall_residual = float("inf")
                continue

            old_weights = params[prefix + ":weights"]
            repeated_abar = jnp.broadcast_to(new_abar, old_abar.shape)
            repeated_hbar = jnp.broadcast_to(new_hbar, old_hbar.shape)
            max_delta = max(
                max_delta,
                float(jnp.max(jnp.abs(weights - old_weights))),
                float(jnp.max(jnp.abs(repeated_abar - old_abar))),
                float(jnp.max(jnp.abs(repeated_hbar - old_hbar))),
            )
            params[prefix + ":weights"] = weights
            params[prefix + ":log_normalizer"] = log_z
            params[prefix + ":expected_phi"] = expected_phi
            params[prefix + ":kl"] = kl
            params[group_name + ":Abar"] = (
                params[group_name + ":Abar"].at[rows].set(repeated_abar)
            )
            params[group_name + ":hbar"] = (
                params[group_name + ":hbar"].at[rows].set(repeated_hbar)
            )
            overall_code = max(overall_code, code)
            if code == 1:
                threshold = min_ess_fraction * samples
                overall_residual = max(
                    overall_residual, (threshold - ess_value) / threshold
                )

        return state._replace(params=params), max_delta, overall_code, overall_residual

    def diagnostics(
        self, state: Mu3StateView
    ) -> dict[str, dict[str, np.ndarray | float | str]]:
        """Per-group weighted moments and stochastic diagnostics."""
        out: dict[str, dict[str, np.ndarray | float | str]] = {}
        for plan in self.view.plans:
            prefix = plan.name
            group_name = plan.gfe_name
            row = int(plan.rows[0])
            code = int(np.asarray(state.params[prefix + ":status"]))
            out[plan.name] = {
                "evaluator": plan.evaluator,
                "members": np.asarray(plan.members),
                "Abar": np.asarray(state.params[group_name + ":Abar"][row]),
                "hbar": np.asarray(state.params[group_name + ":hbar"][row]),
                "weights": np.asarray(state.params[prefix + ":weights"]),
                "log_normalizer": float(state.params[prefix + ":log_normalizer"]),
                "expected_phi": float(state.params[prefix + ":expected_phi"]),
                "kl": float(state.params[prefix + ":kl"]),
                "ess": float(state.params[prefix + ":ess"]),
                "status": (
                    RunStatus.CONVERGED.value
                    if code == 0
                    else RunStatus.BUDGET_EXHAUSTED.value
                    if code == 1
                    else RunStatus.DIVERGED.value
                ),
            }
        return out

    def result_diagnostics(
        self, state: Mu3StateView
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[RunStatus, ...]]:
        ess: list[float] = []
        log_normalizers: list[float] = []
        statuses: list[RunStatus] = []
        for plan in self.view.plans:
            prefix = plan.name
            code = int(np.asarray(state.params[prefix + ":status"]))
            ess.append(float(np.asarray(state.params[prefix + ":ess"])))
            log_normalizers.append(
                float(np.asarray(state.params[prefix + ":log_normalizer"]))
            )
            statuses.append(
                RunStatus.CONVERGED
                if code == 0
                else RunStatus.BUDGET_EXHAUSTED
                if code == 1
                else RunStatus.DIVERGED
            )
        return tuple(ess), tuple(log_normalizers), tuple(statuses)

    def score_epochs(
        self,
        state: State,
        *,
        prepare: Callable[[State, jax.Array], tuple[State, jax.Array]],
        run: RunEpoch,
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
        if not self.view.plans:
            raise ValueError("score_mu3_epochs requires a compiled mu3 evaluator")
        if damp != 0.0:
            raise ValueError("damp does not apply to mixed-GFE trust-region inference")
        if epochs <= 0 or not 0 <= burn_in < epochs:
            raise ValueError("require epochs > 0 and 0 <= burn_in < epochs")
        next_key = key
        current = state
        scores: list[float] = []
        runs: list[RunResult] = []
        for _ in range(epochs):
            current, next_key = prepare(current, next_key)
            result = run(
                current,
                n_iters=n_iters,
                tol=tol,
                harden_last=0,
            )
            runs.append(result)
            current = result.state
            if result.status is not RunStatus.CONVERGED or not result.free_energies:
                return Mu3ScoreResult(
                    current,
                    result.status,
                    tuple(scores),
                    float("nan"),
                    next_key,
                    tuple(runs),
                )
            scores.append(result.free_energies[-1])
        retained = scores[burn_in:]
        return Mu3ScoreResult(
            current,
            RunStatus.CONVERGED,
            tuple(scores),
            float(np.mean(retained)),
            next_key,
            tuple(runs),
        )
