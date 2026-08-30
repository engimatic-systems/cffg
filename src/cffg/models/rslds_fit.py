"""Observed-only variational fitting for recurrent switching state-space models.

The fitter is a host-side coordinate loop around the public :class:`Engine`.
Each coordinate starts from immutable conjugate priors, pools one sequence's
soft factor statistics, and writes point parameters into a fresh fixed-shape
state. The inner Bethe free energy and the outer fixed-point residual remain
separate signals; this module does not invent a scalar objective over parameter
posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import numpy as np

from cffg import conj
from cffg.compiler import compile_template
from cffg.engine import Engine, RunResult, RunStatus, State, _replace
from cffg.learn import Learnables
from cffg.models.coordinate_fit import CoordinateResidual, coordinate_residual
from cffg.models.rslds import (
    DYNAMICS_LEARNABLE,
    EMISSION_LEARNABLE,
    INITIAL_REGIME_LEARNABLE,
    RECURRENCE_LEARNABLE,
    RSLDSLearning,
    rslds,
)
from cffg.models.rslds_coordinate import (
    initial_runtime_blocks,
    legacy_fixed_point_residual,
    legacy_snapshot,
    learned_runtime_blocks,
    observation_probe_states,
    transition_residual,
)
from cffg.models.rslds_init import (
    PCAARHMMInitializer,
    RSLDSInitialization,
    RSLDSInitializer,
    _validate_initialization,
    _validate_observations,
)


@dataclass(frozen=True)
class RSLDSFitPriors:
    """Immutable hyperparameters used for every one-sequence coordinate."""

    initial_alpha: np.ndarray
    dynamics: tuple[conj.MNIW, ...]
    recurrence: conj.PGWeights
    emission: conj.MNIW

    def model_prior(self) -> RSLDSLearning:
        return RSLDSLearning(self.initial_alpha, self.dynamics, self.recurrence)

    def learning_priors(self) -> dict[str, object]:
        return {
            INITIAL_REGIME_LEARNABLE: self.initial_alpha,
            DYNAMICS_LEARNABLE: self.dynamics,
            RECURRENCE_LEARNABLE: self.recurrence,
            EMISSION_LEARNABLE: self.emission,
        }


def vague_rslds_fit_priors(
    regimes: int,
    latent_dimension: int,
    observation_dimension: int,
    *,
    initial_strength: float = 1.0,
    dynamics_scale: float = 1.0,
    recurrence_strength: float = 1e-3,
    emission_scale: float = 1.0,
) -> RSLDSFitPriors:
    """Construct proper weak priors for all fitted parameter families."""
    if not np.isfinite(initial_strength) or initial_strength <= 0:
        raise ValueError("initial_strength must be finite and positive")
    return RSLDSFitPriors(
        initial_alpha=np.full(regimes, initial_strength, dtype=np.float64),
        dynamics=tuple(
            conj.vague_mniw(latent_dimension, latent_dimension, scale=dynamics_scale)
            for _ in range(regimes)
        ),
        recurrence=conj.vague_pg_weights(
            regimes, latent_dimension, strength=recurrence_strength
        ),
        emission=conj.vague_mniw(
            observation_dimension, latent_dimension, scale=emission_scale
        ),
    )


@dataclass(frozen=True)
class RSLDSInferenceAttempt:
    """Bounded evidence from one damping attempt at fixed parameters."""

    damping: float
    status: RunStatus
    recurrence_status: RunStatus
    iterations: int
    residual: float
    message_delta_tail: tuple[float, ...]
    free_energy_tail: tuple[float, ...]
    gaussian_beliefs_proper: bool
    state_finite: bool


@dataclass(frozen=True)
class RSLDSFitPhase:
    """One hidden-state solve and its optional parameter replacement."""

    iteration: int
    coordinate_replaced: bool
    attempts: tuple[RSLDSInferenceAttempt, ...]
    parameter_scales_before: dict[str, float]
    parameter_scales_after: dict[str, float]
    fixed_point_residual: float | None
    runtime_residual: CoordinateResidual | None = None
    functional_transition_residual: CoordinateResidual | None = None


@dataclass(frozen=True)
class RSLDSFitResult:
    """Terminal fitting status, fitted parameters, and inner-solve evidence.

    When every requested coordinate completes, ``state`` has been inferred
    once after the final parameter write. ``last_inference`` is that terminal
    solve, and ``phases`` distinguishes it from parameter-replacement phases.
    ``model_posterior`` carries the current D9
    structured-pair statistic identity and may be checkpointed for resume;
    point ``parameters`` alone are a model initialization, not a resumable
    posterior.
    """

    engine: Engine
    state: State
    status: RunStatus
    inner_status: RunStatus
    iterations: int
    residual: float
    residuals: tuple[float, ...]
    free_energies: tuple[float, ...]
    parameters: dict[str, np.ndarray]
    initialization: RSLDSInitialization
    model_posterior: RSLDSLearning
    emission_posterior: dict[str, np.ndarray]
    last_inference: RunResult | None
    phases: tuple[RSLDSFitPhase, ...]
    timings: dict[str, float]
    error: str | None = None

    def require_converged(self) -> "RSLDSFitResult":
        if self.status is not RunStatus.CONVERGED:
            raise RuntimeError(
                f"rSLDS fitting {self.status.value} after {self.iterations} outer "
                f"iterations (residual={self.residual!r})"
            )
        return self


def _validate_priors(
    priors: RSLDSFitPriors,
    *,
    regimes: int,
    latent_dimension: int,
    observation_dimension: int,
) -> RSLDSLearning:
    model = priors.model_prior()
    if model.regimes != regimes or model.dimension != latent_dimension:
        raise ValueError("model priors do not match the requested regimes and dimension")
    if priors.emission.m.shape != (
        observation_dimension,
        latent_dimension + 1,
    ):
        raise ValueError("emission prior does not match observation and latent dimensions")
    return model


def _point_parameters(learning: Learnables) -> dict[str, np.ndarray]:
    return {
        "initial_regime_probability": np.asarray(
            learning.expected(INITIAL_REGIME_LEARNABLE)
        ).copy(),
        "dynamics_matrix": np.asarray(
            learning.rollover(DYNAMICS_LEARNABLE, "matrix")
        ).copy(),
        "dynamics_offset": np.asarray(
            learning.rollover(DYNAMICS_LEARNABLE, "offset")
        ).copy(),
        "dynamics_covariance": np.asarray(
            learning.rollover(DYNAMICS_LEARNABLE, "cov")
        ).copy(),
        "recurrence_weights": np.asarray(
            learning.rollover(RECURRENCE_LEARNABLE, "weights")
        ).copy(),
        "recurrence_offsets": np.asarray(
            learning.rollover(RECURRENCE_LEARNABLE, "offsets")
        ).copy(),
        "emission_matrix": np.asarray(
            learning.rollover(EMISSION_LEARNABLE, "matrix")
        ).copy(),
        "emission_offset": np.asarray(
            learning.rollover(EMISSION_LEARNABLE, "offset")
        ).copy(),
        "emission_covariance": np.asarray(
            learning.rollover(EMISSION_LEARNABLE, "cov")
        ).copy(),
    }


def _data(observations: np.ndarray) -> dict[str, np.ndarray]:
    return {f"y{step}": row for step, row in enumerate(observations)}


def _cast_state(state: State, dtype: np.dtype) -> State:
    return jax.tree_util.tree_map(
        lambda value: (
            value.astype(dtype)
            if np.issubdtype(value.dtype, np.inexact)
            else value
        ),
        state,
    )


def _install(
    engine: Engine,
    observations: np.ndarray,
    learning: Learnables,
) -> State:
    state = _cast_state(engine.init_state(), observations.dtype)
    params = learning.write(engine, state.params)
    state = _replace(state, params=params)
    return engine.bind_data(state, _data(observations))


def _parameter_scales(points: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        name: float(np.max(np.abs(np.asarray(value))))
        for name, value in points.items()
    }


def _attempt_evidence(
    engine: Engine,
    result: RunResult,
    damping: float,
) -> RSLDSInferenceAttempt:
    try:
        statistics = engine.gaussian_node_stats(result.state)
        gaussian_beliefs_proper = bool(statistics) and all(
            bool(row["proper"]) for row in statistics.values()
        )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        gaussian_beliefs_proper = False
    state_finite = all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(result.state)
    )
    return RSLDSInferenceAttempt(
        damping=float(damping),
        status=result.status,
        recurrence_status=result.inner_status,
        iterations=result.iterations,
        residual=float(result.residual),
        message_delta_tail=tuple(float(value) for value in result.message_deltas[-8:]),
        free_energy_tail=tuple(float(value) for value in result.free_energies[-8:]),
        gaussian_beliefs_proper=gaussian_beliefs_proper,
        state_finite=state_finite,
    )


def _run_inner(
    engine: Engine,
    state: State,
    *,
    max_iterations: int,
    tolerance: float,
    damping: float,
    max_retries: int,
) -> tuple[RunResult, tuple[RSLDSInferenceAttempt, ...]]:
    """Retry a finite exhausted coordinate with progressively larger steps.

    ``damping`` is the retained fraction of the old natural parameters. Each
    retry doubles the update fraction ``1 - damping`` while it remains a
    convex combination. The fixed points are unchanged, and every attempt
    restarts from the same parameter-bound state.
    """
    if engine.arena.mu3_groups:
        raise ValueError(
            "deterministic fitting retries do not support stochastic mu3 updates"
        )
    attempts: list[RSLDSInferenceAttempt] = []
    current_damping = damping
    result: RunResult | None = None
    for retry in range(max_retries + 1):
        result = engine.run(
            state,
            max_iterations,
            tol=tolerance,
            damp=current_damping,
        )
        attempts.append(_attempt_evidence(engine, result, current_damping))
        if result.status is not RunStatus.BUDGET_EXHAUSTED or retry == max_retries:
            break
        expanded_step_damping = 2.0 * current_damping - 1.0
        if expanded_step_damping <= 0.0:
            break
        current_damping = expanded_step_damping
    assert result is not None
    return result, tuple(attempts)


def _result(
    *,
    engine: Engine,
    state: State,
    status: RunStatus,
    inner_status: RunStatus,
    iterations: int,
    residuals: list[float],
    free_energies: list[float],
    points: dict[str, np.ndarray],
    initialization: RSLDSInitialization,
    learning: Learnables,
    last_inference: RunResult | None,
    phases: list[RSLDSFitPhase],
    timings: dict[str, float],
    error: str | None = None,
) -> RSLDSFitResult:
    return RSLDSFitResult(
        engine=engine,
        state=state,
        status=status,
        inner_status=inner_status,
        iterations=iterations,
        residual=float(residuals[-1]) if residuals else float("inf"),
        residuals=tuple(residuals),
        free_energies=tuple(free_energies),
        parameters={name: np.asarray(value).copy() for name, value in points.items()},
        initialization=initialization,
        model_posterior=RSLDSLearning._from_runtime(learning),
        emission_posterior={
            key: np.asarray(value).copy()
            for key, value in learning.params[EMISSION_LEARNABLE].items()
        },
        last_inference=last_inference,
        phases=tuple(phases),
        timings=dict(timings),
        error=error,
    )


def fit_rslds(
    observations: np.ndarray,
    *,
    regimes: int,
    latent_dimension: int,
    initializer: RSLDSInitializer,
    priors: RSLDSFitPriors,
    max_outer_iterations: int,
    key: jax.Array,
    max_inner_iterations: int = 200,
    inner_tolerance: float = 1e-6,
    outer_tolerance: float | None = None,
    schedule: str = "auto",
    damp: float = 0.0,
    max_inner_retries: int = 2,
) -> RSLDSFitResult:
    """Fit one rSLDS sequence without receiving generating variables.

    Every outer coordinate reruns hidden-state inference with one compiled
    engine, replaces all conjugate posteriors from ``priors``, and installs the
    resulting point parameters into a fresh state. Two consecutive
    fixed-point residuals at or below ``outer_tolerance`` certify convergence.
    A finite exhausted inner solve may restart from the same state with a
    larger convex update step, up to ``max_inner_retries`` times. The terminal
    state is inferred once after the last parameter replacement.
    """
    observations = _validate_observations(observations)
    if max_outer_iterations <= 0 or max_inner_iterations <= 0:
        raise ValueError("outer and inner iteration budgets must be positive")
    if not np.isfinite(inner_tolerance) or inner_tolerance <= 0:
        raise ValueError("inner_tolerance must be finite and positive")
    if not np.isfinite(damp) or not 0.0 <= damp < 1.0:
        raise ValueError("damp must be finite and in [0, 1)")
    if not isinstance(max_inner_retries, int) or max_inner_retries < 0:
        raise ValueError("max_inner_retries must be a nonnegative integer")
    if outer_tolerance is None:
        outer_tolerance = 8.0 * np.sqrt(np.finfo(observations.dtype).eps)
    if not np.isfinite(outer_tolerance) or outer_tolerance <= 0:
        raise ValueError("outer_tolerance must be finite and positive")

    _validate_priors(
        priors,
        regimes=regimes,
        latent_dimension=latent_dimension,
        observation_dimension=observations.shape[1],
    )
    started = time.perf_counter()
    initialization = _validate_initialization(
        initializer(
            observations.copy(),
            regimes=regimes,
            latent_dimension=latent_dimension,
            key=key,
        ),
        regimes=regimes,
        latent_dimension=latent_dimension,
        observation_dimension=observations.shape[1],
    )
    initialization_seconds = time.perf_counter() - started
    arrays = initialization.arrays()
    template, model_arrays = rslds(
        n_steps=observations.shape[0],
        dynamics_matrix=arrays["dynamics_matrix"],
        dynamics_offset=arrays["dynamics_offset"],
        dynamics_covariance=arrays["dynamics_covariance"],
        recurrence_weights=arrays["recurrence_weights"],
        recurrence_offsets=arrays["recurrence_offsets"],
        emission_matrix=arrays["emission_matrix"],
        emission_covariance=arrays["emission_covariance"],
        initial_state_mean=arrays["initial_state_mean"],
        initial_state_covariance=arrays["initial_state_covariance"],
        initial_regime_probability=arrays["initial_regime_probability"],
        emission_offset=arrays["emission_offset"],
    )
    started = time.perf_counter()
    engine = Engine(compile_template(template, arrays=model_arrays), schedule=schedule)
    compile_seconds = time.perf_counter() - started
    current_learning = Learnables(engine, priors=priors.learning_priors())
    current_points = {
        name: arrays[name]
        for name in (
            "initial_regime_probability",
            "dynamics_matrix",
            "dynamics_offset",
            "dynamics_covariance",
            "recurrence_weights",
            "recurrence_offsets",
            "emission_matrix",
            "emission_offset",
            "emission_covariance",
        )
    }
    current_snapshot = legacy_snapshot(current_learning, current_points)
    current_runtime_blocks = initial_runtime_blocks(current_points)
    probe_states = observation_probe_states(observations, initialization)
    current_state = engine.bind_data(
        _cast_state(engine.init_state(), observations.dtype), _data(observations)
    )
    residuals: list[float] = []
    free_energies: list[float] = []
    phases: list[RSLDSFitPhase] = []
    consecutive = 0
    timings = {
        "initialization": initialization_seconds,
        "compile": compile_seconds,
        "inference": 0.0,
        "coordinate": 0.0,
    }

    fit_status = RunStatus.BUDGET_EXHAUSTED
    completed_iterations = 0
    for iteration in range(1, max_outer_iterations + 1):
        completed_iterations = iteration
        scales_before = _parameter_scales(current_points)
        started = time.perf_counter()
        inner, attempts = _run_inner(
            engine,
            current_state,
            max_iterations=max_inner_iterations,
            tolerance=inner_tolerance,
            damping=damp,
            max_retries=max_inner_retries,
        )
        timings["inference"] += time.perf_counter() - started
        if inner.status is not RunStatus.CONVERGED:
            phases.append(
                RSLDSFitPhase(
                    iteration=iteration,
                    coordinate_replaced=False,
                    attempts=attempts,
                    parameter_scales_before=scales_before,
                    parameter_scales_after=scales_before,
                    fixed_point_residual=None,
                )
            )
            outer_status = (
                RunStatus.BUDGET_EXHAUSTED
                if inner.status is RunStatus.BUDGET_EXHAUSTED
                else RunStatus.DIVERGED
            )
            return _result(
                engine=engine,
                state=inner.state,
                status=outer_status,
                inner_status=inner.status,
                iterations=iteration,
                residuals=residuals,
                free_energies=free_energies,
                points=current_points,
                initialization=initialization,
                learning=current_learning,
                last_inference=inner,
                phases=phases,
                timings=timings,
                error=f"inner inference {inner.status.value}",
            )
        if not inner.free_energies or not np.all(np.isfinite(inner.free_energies)):
            phases.append(
                RSLDSFitPhase(
                    iteration=iteration,
                    coordinate_replaced=False,
                    attempts=attempts,
                    parameter_scales_before=scales_before,
                    parameter_scales_after=scales_before,
                    fixed_point_residual=None,
                )
            )
            return _result(
                engine=engine,
                state=inner.state,
                status=RunStatus.DIVERGED,
                inner_status=RunStatus.DIVERGED,
                iterations=iteration,
                residuals=residuals,
                free_energies=free_energies,
                points=current_points,
                initialization=initialization,
                learning=current_learning,
                last_inference=inner,
                phases=phases,
                timings=timings,
                error="inner Bethe free energy is missing or non-finite",
            )
        free_energies.append(float(inner.free_energies[-1]))

        started = time.perf_counter()
        try:
            candidate_learning = Learnables(
                engine, priors=priors.learning_priors()
            )
            candidate_learning.replace(engine, inner)
            candidate_points = _point_parameters(candidate_learning)
            candidate_snapshot = legacy_snapshot(candidate_learning, candidate_points)
            residual = legacy_fixed_point_residual(
                current_snapshot, candidate_snapshot
            )
            if not np.isfinite(residual):
                raise ValueError("outer fixed-point residual is non-finite")
            candidate_runtime_blocks = learned_runtime_blocks(
                candidate_learning, candidate_points
            )
            runtime_residual = coordinate_residual(
                current_runtime_blocks, candidate_runtime_blocks
            )
            functional_residual = transition_residual(
                current_points, candidate_points, probe_states
            )
            candidate_state = _install(engine, observations, candidate_learning)
        except (FloatingPointError, np.linalg.LinAlgError, TypeError, ValueError) as error:
            timings["coordinate"] += time.perf_counter() - started
            phases.append(
                RSLDSFitPhase(
                    iteration=iteration,
                    coordinate_replaced=False,
                    attempts=attempts,
                    parameter_scales_before=scales_before,
                    parameter_scales_after=scales_before,
                    fixed_point_residual=None,
                )
            )
            return _result(
                engine=engine,
                state=inner.state,
                status=RunStatus.DIVERGED,
                inner_status=inner.status,
                iterations=iteration,
                residuals=residuals,
                free_energies=free_energies,
                points=current_points,
                initialization=initialization,
                learning=current_learning,
                last_inference=inner,
                phases=phases,
                timings=timings,
                error=str(error),
            )
        timings["coordinate"] += time.perf_counter() - started

        residuals.append(residual)
        phases.append(
            RSLDSFitPhase(
                iteration=iteration,
                coordinate_replaced=True,
                attempts=attempts,
                parameter_scales_before=scales_before,
                parameter_scales_after=_parameter_scales(candidate_points),
                fixed_point_residual=residual,
                runtime_residual=runtime_residual,
                functional_transition_residual=functional_residual,
            )
        )
        consecutive = consecutive + 1 if residual <= outer_tolerance else 0
        current_learning = candidate_learning
        current_points = candidate_points
        current_snapshot = candidate_snapshot
        current_runtime_blocks = candidate_runtime_blocks
        current_state = candidate_state
        if consecutive >= 2:
            fit_status = RunStatus.CONVERGED
            break

    scales_before = _parameter_scales(current_points)
    started = time.perf_counter()
    terminal, attempts = _run_inner(
        engine,
        current_state,
        max_iterations=max_inner_iterations,
        tolerance=inner_tolerance,
        damping=damp,
        max_retries=max_inner_retries,
    )
    timings["inference"] += time.perf_counter() - started
    phases.append(
        RSLDSFitPhase(
            iteration=completed_iterations + 1,
            coordinate_replaced=False,
            attempts=attempts,
            parameter_scales_before=scales_before,
            parameter_scales_after=scales_before,
            fixed_point_residual=None,
        )
    )
    error = None
    if terminal.status is not RunStatus.CONVERGED:
        fit_status = (
            RunStatus.BUDGET_EXHAUSTED
            if terminal.status is RunStatus.BUDGET_EXHAUSTED
            else RunStatus.DIVERGED
        )
        error = f"terminal inner inference {terminal.status.value}"
    elif not terminal.free_energies or not np.all(
        np.isfinite(terminal.free_energies)
    ):
        fit_status = RunStatus.DIVERGED
        error = "terminal inner Bethe free energy is missing or non-finite"
    else:
        free_energies.append(float(terminal.free_energies[-1]))

    return _result(
        engine=engine,
        state=terminal.state,
        status=fit_status,
        inner_status=(
            RunStatus.DIVERGED
            if error and terminal.status is RunStatus.CONVERGED
            else terminal.status
        ),
        iterations=completed_iterations,
        residuals=residuals,
        free_energies=free_energies,
        points=current_points,
        initialization=initialization,
        learning=current_learning,
        last_inference=terminal,
        phases=phases,
        timings=timings,
        error=error,
    )
