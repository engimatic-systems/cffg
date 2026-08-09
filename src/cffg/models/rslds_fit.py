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
from typing import Protocol

import jax
import numpy as np

from cffg import conj
from cffg.compiler import compile_template
from cffg.engine import Engine, RunResult, RunStatus, State, _replace
from cffg.learn import FAMILIES, Learnables
from cffg.models.rslds import RSLDSLearning, rslds


@dataclass(frozen=True)
class RSLDSInitialization:
    """Point parameters and fixed continuous-state prior for the first run."""

    dynamics_matrix: np.ndarray
    dynamics_offset: np.ndarray
    dynamics_covariance: np.ndarray
    recurrence_weights: np.ndarray
    recurrence_offsets: np.ndarray
    emission_matrix: np.ndarray
    emission_offset: np.ndarray
    emission_covariance: np.ndarray
    initial_state_mean: np.ndarray
    initial_state_covariance: np.ndarray
    initial_regime_probability: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(value).copy()
            for name, value in vars(self).items()
        }


class RSLDSInitializer(Protocol):
    """Observation-only initializer used before an rSLDS engine is compiled."""

    def __call__(
        self,
        observations: np.ndarray,
        *,
        regimes: int,
        latent_dimension: int,
        key: jax.Array,
    ) -> RSLDSInitialization: ...


def _covariance(
    rows: np.ndarray,
    *,
    reference_scale: float | None = None,
) -> np.ndarray:
    rows = np.asarray(rows)
    dimension = rows.shape[1]
    centered = rows - np.mean(rows, axis=0)
    covariance = centered.T @ centered / max(rows.shape[0], 1)
    covariance = 0.5 * (covariance + covariance.T)
    dtype = covariance.dtype
    if reference_scale is None:
        reference_scale = float(np.max(np.abs(covariance)))
    scale = max(float(reference_scale), float(np.finfo(dtype).tiny))
    # The dtype-scaled diagonal protects invertibility of a rank-deficient sample.
    guard = 16.0 * dimension * np.finfo(dtype).eps * scale
    return covariance + np.asarray(guard, dtype=dtype) * np.eye(
        dimension, dtype=dtype
    )


def _canonicalize_components(components: np.ndarray) -> np.ndarray:
    components = components.copy()
    for row in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[row])))
        if components[row, pivot] < 0:
            components[row] *= -1
    return components


def _repair_empty_clusters(
    labels: np.ndarray,
    distances: np.ndarray,
    regimes: int,
) -> np.ndarray:
    labels = labels.copy()
    counts = np.bincount(labels, minlength=regimes)
    for empty in np.flatnonzero(counts == 0):
        assigned_distance = distances[np.arange(labels.size), labels]
        candidates = np.argsort(assigned_distance)[::-1]
        moved = next(index for index in candidates if counts[labels[index]] > 1)
        source = labels[moved]
        labels[moved] = empty
        counts[source] -= 1
        counts[empty] += 1
    return labels


def _transition_kmeans(
    states: np.ndarray,
    regimes: int,
    key: jax.Array,
    iterations: int,
) -> np.ndarray:
    features = np.concatenate((states[:-1], states[1:]), axis=1)
    centered = features - np.mean(features, axis=0)
    scale = np.std(centered, axis=0)
    dtype = features.dtype
    threshold = np.sqrt(np.finfo(dtype).eps) * max(
        float(np.max(np.abs(centered))), float(np.finfo(dtype).tiny)
    )
    # Constant coordinates carry no clustering information and use unit scale.
    scale = np.where(scale > threshold, scale, np.asarray(1.0, dtype=dtype))
    normalized = centered / scale
    try:
        indices = np.asarray(
            jax.random.choice(
                key,
                normalized.shape[0],
                shape=(regimes,),
                replace=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("key must be an explicit JAX PRNG key") from error
    centers = normalized[indices].copy()
    previous = None
    for _ in range(iterations):
        distances = np.sum(
            np.square(normalized[:, None, :] - centers[None, :, :]), axis=2
        )
        labels = _repair_empty_clusters(
            np.argmin(distances, axis=1), distances, regimes
        )
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        centers = np.stack(
            [np.mean(normalized[labels == regime], axis=0) for regime in range(regimes)]
        )
    return labels


@dataclass(frozen=True)
class PCAARHMMInitializer:
    """PCA coordinates plus transition-pair clustering and local AR fits.

    PCA fixes an observation-derived latent coordinate system. K-means on
    consecutive latent pairs assigns regime labels, after which regularized
    local autoregressions initialize the switching dynamics. The explicit JAX
    key selects the initial cluster centers and is the only random input.
    """

    kmeans_iterations: int = 100
    regression_prior_strength: float = 1.0
    transition_pseudocount: float = 1.0

    def __post_init__(self) -> None:
        if self.kmeans_iterations <= 0:
            raise ValueError("kmeans_iterations must be positive")
        if (
            not np.isfinite(self.regression_prior_strength)
            or self.regression_prior_strength <= 0
        ):
            raise ValueError("regression_prior_strength must be finite and positive")
        if (
            not np.isfinite(self.transition_pseudocount)
            or self.transition_pseudocount <= 0
        ):
            raise ValueError("transition_pseudocount must be finite and positive")

    def __call__(
        self,
        observations: np.ndarray,
        *,
        regimes: int,
        latent_dimension: int,
        key: jax.Array,
    ) -> RSLDSInitialization:
        observations = _validate_observations(observations)
        steps, observation_dimension = observations.shape
        if regimes < 2:
            raise ValueError("regimes must be at least two")
        if not 1 <= latent_dimension <= min(steps, observation_dimension):
            raise ValueError(
                "latent_dimension must be positive and no larger than the data rank"
            )
        if steps - 1 < regimes:
            raise ValueError("initializer needs at least one transition per regime")

        observation_mean = np.mean(observations, axis=0)
        centered = observations - observation_mean
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        components = _canonicalize_components(right[:latent_dimension])
        states = centered @ components.T
        labels_tail = _transition_kmeans(
            states, regimes, key, self.kmeans_iterations
        )
        labels = np.concatenate((labels_tail[:1], labels_tail))

        dimension = latent_dimension
        augmented_identity = np.concatenate(
            (
                np.eye(dimension, dtype=states.dtype),
                np.zeros((dimension, 1), dtype=states.dtype),
            ),
            axis=1,
        )
        dynamics_matrix = np.empty((regimes, dimension, dimension), dtype=states.dtype)
        dynamics_offset = np.empty((regimes, dimension), dtype=states.dtype)
        dynamics_covariance = np.empty(
            (regimes, dimension, dimension), dtype=states.dtype
        )
        state_scale = max(
            float(np.max(np.abs(_covariance(states)))),
            float(np.finfo(states.dtype).tiny),
        )
        for regime in range(regimes):
            selected = labels_tail == regime
            design = np.concatenate(
                (
                    states[:-1][selected],
                    np.ones((int(np.sum(selected)), 1), dtype=states.dtype),
                ),
                axis=1,
            )
            targets = states[1:][selected]
            strength = np.asarray(self.regression_prior_strength, dtype=states.dtype)
            precision = design.T @ design + strength * np.eye(
                dimension + 1, dtype=states.dtype
            )
            natural = targets.T @ design + strength * augmented_identity
            coefficients = np.linalg.solve(precision.T, natural.T).T
            dynamics_matrix[regime] = coefficients[:, :-1]
            dynamics_offset[regime] = coefficients[:, -1]
            residuals = targets - design @ coefficients.T
            dynamics_covariance[regime] = _covariance(
                residuals, reference_scale=state_scale
            )

        counts = np.full(
            (regimes, regimes),
            self.transition_pseudocount,
            dtype=states.dtype,
        )
        np.add.at(counts, (labels[:-1], labels[1:]), 1.0)
        transition = counts / np.sum(counts, axis=1, keepdims=True)
        recurrence_offsets = np.empty((regimes, regimes - 1), dtype=states.dtype)
        for source in range(regimes):
            for stick in range(regimes - 1):
                remaining = np.sum(transition[source, stick:])
                accept = transition[source, stick] / remaining
                recurrence_offsets[source, stick] = np.log(accept) - np.log1p(-accept)

        initial_counts = np.full(
            regimes, self.transition_pseudocount, dtype=states.dtype
        )
        initial_counts[labels[0]] += 1.0
        emission_residuals = observations - (
            states @ components + observation_mean
        )
        observation_scale = max(
            float(np.max(np.abs(_covariance(observations)))),
            float(np.finfo(observations.dtype).tiny),
        )
        return RSLDSInitialization(
            dynamics_matrix=dynamics_matrix,
            dynamics_offset=dynamics_offset,
            dynamics_covariance=dynamics_covariance,
            recurrence_weights=np.zeros(
                (regimes, regimes - 1, dimension), dtype=states.dtype
            ),
            recurrence_offsets=recurrence_offsets,
            emission_matrix=components.T,
            emission_offset=observation_mean,
            emission_covariance=_covariance(
                emission_residuals, reference_scale=observation_scale
            ),
            initial_state_mean=states[0],
            initial_state_covariance=_covariance(states),
            initial_regime_probability=initial_counts / np.sum(initial_counts),
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
class RSLDSFitResult:
    """Terminal fitting status, fitted parameters, and last inner result.

    ``state`` is a fresh observation-bound state carrying ``parameters``. Its
    messages have not been run after the final parameter write;
    ``last_inference`` retains the preceding status-bearing engine result.
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
    timings: dict[str, float]
    error: str | None = None

    def require_converged(self) -> "RSLDSFitResult":
        if self.status is not RunStatus.CONVERGED:
            raise RuntimeError(
                f"rSLDS fitting {self.status.value} after {self.iterations} outer "
                f"iterations (residual={self.residual!r})"
            )
        return self


def _validate_observations(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[0] < 2 or observations.shape[1] < 1:
        raise ValueError("observations must have shape [T,p] with T>=2 and p>=1")
    if not np.issubdtype(observations.dtype, np.inexact):
        raise TypeError("observations must use a floating dtype")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite")
    return observations


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


def _validate_initialization(
    initialization: RSLDSInitialization,
    *,
    regimes: int,
    latent_dimension: int,
    observation_dimension: int,
) -> RSLDSInitialization:
    if not isinstance(initialization, RSLDSInitialization):
        raise TypeError("initializer must return RSLDSInitialization")
    arrays = initialization.arrays()
    expected = {
        "dynamics_matrix": (regimes, latent_dimension, latent_dimension),
        "dynamics_offset": (regimes, latent_dimension),
        "dynamics_covariance": (regimes, latent_dimension, latent_dimension),
        "recurrence_weights": (regimes, regimes - 1, latent_dimension),
        "recurrence_offsets": (regimes, regimes - 1),
        "emission_matrix": (observation_dimension, latent_dimension),
        "emission_offset": (observation_dimension,),
        "emission_covariance": (observation_dimension, observation_dimension),
        "initial_state_mean": (latent_dimension,),
        "initial_state_covariance": (latent_dimension, latent_dimension),
        "initial_regime_probability": (regimes,),
    }
    actual = {name: value.shape for name, value in arrays.items()}
    if actual != expected:
        raise ValueError(f"initializer shapes {actual} != {expected}")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("initializer parameters must be finite")
    for name in (
        "dynamics_covariance",
        "emission_covariance",
        "initial_state_covariance",
    ):
        try:
            np.linalg.cholesky(arrays[name])
        except np.linalg.LinAlgError:
            raise ValueError(f"initializer {name} must be positive definite") from None
    probability = arrays["initial_regime_probability"]
    if np.any(probability < 0) or np.sum(probability) <= 0:
        raise ValueError("initial regime probabilities must be nonnegative with positive mass")
    arrays["initial_regime_probability"] = probability / np.sum(probability)
    return RSLDSInitialization(**arrays)


def _emission_points(learning: Learnables) -> dict[str, np.ndarray]:
    values = FAMILIES["mniw"].rollover(
        learning.params["emission"], "gaussian_linear"
    )
    return {
        "emission_matrix": np.asarray(values["matrix"]).copy(),
        "emission_offset": np.asarray(values["offset"]).copy(),
        "emission_covariance": np.asarray(values["cov"]).copy(),
    }


def _point_parameters(
    model: RSLDSLearning,
    emission: Learnables,
) -> dict[str, np.ndarray]:
    return {**model.parameter_arrays(), **_emission_points(emission)}


def _posterior_arrays(
    model: RSLDSLearning,
    emission: Learnables,
) -> dict[str, np.ndarray]:
    arrays = {"posterior/initial_alpha": np.asarray(model.initial_alpha)}
    for index, posterior in enumerate(model.dynamics):
        arrays.update(
            {
                f"posterior/dynamics/{index}/m": np.asarray(posterior.m),
                f"posterior/dynamics/{index}/lam": np.asarray(posterior.lam),
                f"posterior/dynamics/{index}/psi": np.asarray(posterior.psi),
                f"posterior/dynamics/{index}/nu": np.asarray(posterior.nu),
            }
        )
    arrays["posterior/recurrence/eta"] = np.asarray(model.recurrence.eta)
    arrays["posterior/recurrence/precision"] = np.asarray(
        model.recurrence.precision
    )
    arrays.update(
        {
            f"posterior/emission/{key}": np.asarray(value)
            for key, value in emission.params["emission"].items()
        }
    )
    return arrays


def _snapshot(
    model: RSLDSLearning,
    emission: Learnables,
    points: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        **_posterior_arrays(model, emission),
        **{f"point/{name}": np.asarray(value) for name, value in points.items()},
    }


def _fixed_point_residual(
    old: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
) -> float:
    if old.keys() != new.keys():
        raise ValueError("fixed-point snapshots disagree on parameter families")
    residual = 0.0
    for name in old:
        before = np.asarray(old[name])
        after = np.asarray(new[name])
        if before.shape != after.shape:
            raise ValueError(f"fixed-point array {name!r} changed shape")
        if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            return float("inf")
        scale = max(
            1.0,
            float(np.max(np.abs(before), initial=0.0)),
            float(np.max(np.abs(after), initial=0.0)),
        )
        residual = max(
            residual,
            float(np.max(np.abs(after - before), initial=0.0)) / scale,
        )
    return residual


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
    model: RSLDSLearning,
    emission: Learnables,
) -> State:
    state = model.install(
        engine, _cast_state(engine.init_state(), observations.dtype)
    )
    params = emission.write(engine, state.params, only=("emission",))
    state = _replace(state, params=params)
    return engine.bind_data(state, _data(observations))


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
    model: RSLDSLearning,
    emission: Learnables,
    last_inference: RunResult | None,
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
        model_posterior=model,
        emission_posterior={
            key: np.asarray(value).copy()
            for key, value in emission.params["emission"].items()
        },
        last_inference=last_inference,
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
) -> RSLDSFitResult:
    """Fit one rSLDS sequence without receiving generating variables.

    Every outer coordinate reruns hidden-state inference with one compiled
    engine, replaces all conjugate posteriors from ``priors``, and installs the
    resulting point parameters into a fresh state. Two consecutive
    fixed-point residuals at or below ``outer_tolerance`` certify convergence.
    """
    observations = _validate_observations(observations)
    if max_outer_iterations <= 0 or max_inner_iterations <= 0:
        raise ValueError("outer and inner iteration budgets must be positive")
    if not np.isfinite(inner_tolerance) or inner_tolerance <= 0:
        raise ValueError("inner_tolerance must be finite and positive")
    if not np.isfinite(damp) or not 0.0 <= damp < 1.0:
        raise ValueError("damp must be finite and in [0, 1)")
    if outer_tolerance is None:
        outer_tolerance = 8.0 * np.sqrt(np.finfo(observations.dtype).eps)
    if not np.isfinite(outer_tolerance) or outer_tolerance <= 0:
        raise ValueError("outer_tolerance must be finite and positive")

    model_prior = _validate_priors(
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
    emission_prior = Learnables(engine, priors={"emission": priors.emission})
    current_model = model_prior
    current_emission = emission_prior
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
    current_snapshot = _snapshot(current_model, current_emission, current_points)
    current_state = engine.bind_data(
        _cast_state(engine.init_state(), observations.dtype), _data(observations)
    )
    residuals: list[float] = []
    free_energies: list[float] = []
    consecutive = 0
    last_inference: RunResult | None = None
    timings = {
        "initialization": initialization_seconds,
        "compile": compile_seconds,
        "inference": 0.0,
        "coordinate": 0.0,
    }

    for iteration in range(1, max_outer_iterations + 1):
        started = time.perf_counter()
        inner = engine.run(
            current_state,
            max_inner_iterations,
            tol=inner_tolerance,
            damp=damp,
        )
        timings["inference"] += time.perf_counter() - started
        last_inference = inner
        if inner.status is not RunStatus.CONVERGED:
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
                model=current_model,
                emission=current_emission,
                last_inference=inner,
                timings=timings,
                error=f"inner inference {inner.status.value}",
            )
        if not inner.free_energies or not np.all(np.isfinite(inner.free_energies)):
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
                model=current_model,
                emission=current_emission,
                last_inference=inner,
                timings=timings,
                error="inner Bethe free energy is missing or non-finite",
            )
        free_energies.append(float(inner.free_energies[-1]))

        started = time.perf_counter()
        try:
            candidate_model = model_prior.update(engine, inner.state)
            candidate_emission = Learnables(
                engine, priors={"emission": priors.emission}
            )
            candidate_emission.replace(engine, inner, only=("emission",))
            candidate_points = _point_parameters(candidate_model, candidate_emission)
            candidate_snapshot = _snapshot(
                candidate_model, candidate_emission, candidate_points
            )
            residual = _fixed_point_residual(current_snapshot, candidate_snapshot)
            if not np.isfinite(residual):
                raise ValueError("outer fixed-point residual is non-finite")
            candidate_state = _install(
                engine, observations, candidate_model, candidate_emission
            )
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            timings["coordinate"] += time.perf_counter() - started
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
                model=current_model,
                emission=current_emission,
                last_inference=inner,
                timings=timings,
                error=str(error),
            )
        timings["coordinate"] += time.perf_counter() - started

        residuals.append(residual)
        consecutive = consecutive + 1 if residual <= outer_tolerance else 0
        current_model = candidate_model
        current_emission = candidate_emission
        current_points = candidate_points
        current_snapshot = candidate_snapshot
        current_state = candidate_state
        if consecutive >= 2:
            return _result(
                engine=engine,
                state=current_state,
                status=RunStatus.CONVERGED,
                inner_status=inner.status,
                iterations=iteration,
                residuals=residuals,
                free_energies=free_energies,
                points=current_points,
                initialization=initialization,
                model=current_model,
                emission=current_emission,
                last_inference=inner,
                timings=timings,
            )

    return _result(
        engine=engine,
        state=current_state,
        status=RunStatus.BUDGET_EXHAUSTED,
        inner_status=(
            RunStatus.CONVERGED if last_inference is None else last_inference.status
        ),
        iterations=max_outer_iterations,
        residuals=residuals,
        free_energies=free_energies,
        points=current_points,
        initialization=initialization,
        model=current_model,
        emission=current_emission,
        last_inference=last_inference,
        timings=timings,
    )
