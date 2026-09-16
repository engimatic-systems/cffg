"""Deterministic full-batch Adam training of certified neural Gaussian means."""

from dataclasses import asdict, dataclass
import json

import jax
import jax.numpy as jnp
import numpy as np

from cffg.bundle import save_bundle
from cffg.compiler import compile_template
from cffg.core.serialize import from_dict, structure_hash, to_dict
from cffg.engine import Engine
from cffg.implicit import FixedPointDifferentiator, FixedPointGradient, GradientStatus
from cffg.neural_artifacts import array_digest, read_artifact, write_artifact
from cffg.runtime_types import RunResult, RunStatus, _replace
from cffg.trainable import NeuralParameters


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = .03
    beta1: float = .9
    beta2: float = .999
    objective_scale: float = 1.
    inference_iterations: int = 50
    inference_tolerance: float = 1e-8
    gradient_tolerance: float | None = None
    adjoint_iterations: int = 20
    krylov_dimension: int = 32

    def __post_init__(self):
        for name in ("learning_rate", "objective_scale", "inference_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("beta1", "beta2"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in [0,1)")
        for name in ("inference_iterations", "adjoint_iterations", "krylov_dimension"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.gradient_tolerance is not None and (
            not np.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0
        ):
            raise ValueError("gradient_tolerance must be finite and positive")


@dataclass(frozen=True)
class TrainingState:
    identity: str
    weights: jax.Array
    first_moment: jax.Array
    second_moment: jax.Array
    step: int = 0


@dataclass(frozen=True)
class TrainingStep:
    state: TrainingState
    applied: bool
    message: str
    objective: float
    inference: RunResult
    derivative: FixedPointGradient | None = None

    def require_applied(self):
        if not self.applied:
            raise ValueError(f"neural update refused: {self.message}")
        return self.state


class NeuralTrainer:
    """A fixed training problem with explicit, immutable optimizer state.

    Recreate with the original template, initial blobs, observations and settings
    before loading a continuation. Those inputs determine the reference chart and
    starting messages, and are bound by the checkpoint identity. This full-batch
    API owns no random stream, minibatch position or stochastic data generator.
    Export trained point parameters separately as an ordinary model bundle.
    """

    def __init__(self, template, arrays, observations, *, config=None, factors=None):
        self.config = config or TrainingConfig()
        self.template = from_dict(to_dict(template))
        self.arrays = {name: np.asarray(value).copy() for name, value in arrays.items()}
        self.engine = Engine(compile_template(self.template, arrays=self.arrays))
        if (self.engine.arena.semiring != "sum" or self.engine.arena.mu3_groups
                or len(self.engine.arena.delta_pairs)):
            raise ValueError("neural training requires a deterministic smooth sum-product graph")
        state = self.engine.bind_data(self.engine.init_state(), observations)
        self.parameters = NeuralParameters.from_engine(self.engine, state.params, factors)
        self.initial_weights = self.parameters.extract(state.params)
        self.dtype = np.asarray(self.initial_weights).dtype
        self.reference = self.engine.run(state, self.config.inference_iterations,
                                         tol=self.config.inference_tolerance)
        self.reference.require_converged()
        self._derivative = FixedPointDifferentiator(
            self.engine, self.reference, factors=factors, tolerance=self.config.gradient_tolerance,
            max_iterations=self.config.adjoint_iterations, krylov_dimension=self.config.krylov_dimension,
        )
        self._metadata = {
            "format": "cffg.neural.adam", "version": 1,
            "structure": structure_hash(self.template), "configuration": asdict(self.config),
            "dtype": self.dtype.str,
            "parameters": [asdict(blob) for blob in self.parameters.blobs],
            "initial_arrays": array_digest(self.arrays),
            "reference": array_digest({
                name: getattr(self.reference.state, name)
                for name in ("messages", "g_eta", "g_lam", "ef_eta", "g_obs", "g_mask", "ef_obs", "ef_mask")
            }),
        }
        # Round-trip tuples to their JSON representation for exact load comparison.
        self._metadata = json.loads(json.dumps(self._metadata, sort_keys=True))
        self.identity = array_digest({"metadata": np.asarray(json.dumps(self._metadata, sort_keys=True))})

    def initial_state(self):
        zeros = jnp.zeros_like(self.initial_weights)
        return TrainingState(self.identity, self.initial_weights, zeros, zeros)

    def _validate(self, state):
        if state.identity != self.identity:
            raise ValueError("training state belongs to another model, data set or configuration")
        if type(state.step) is not int or not 0 <= state.step < np.iinfo(np.int64).max:
            raise ValueError("optimizer step must be a nonnegative representable integer")
        for name in ("weights", "first_moment", "second_moment"):
            value = np.asarray(getattr(state, name))
            if value.shape != self.initial_weights.shape or value.dtype != self.dtype or not np.isfinite(value).all():
                raise ValueError(f"optimizer {name} shape, dtype or finite-value mismatch")
        if np.any(np.asarray(state.second_moment) < 0):
            raise ValueError("Adam second moments must be nonnegative")
        if state.step == 0 and (np.any(np.asarray(state.first_moment)) or np.any(np.asarray(state.second_moment))):
            raise ValueError("unstarted optimizer must have zero moments")

    def infer(self, state):
        self._validate(state)
        reference = self.reference.state
        initial = _replace(reference, params=self.parameters.inject(state.weights, reference.params))
        return self.engine.run(initial, self.config.inference_iterations, tol=self.config.inference_tolerance)

    def step(self, state):
        result = self.infer(state)
        if result.status is not RunStatus.CONVERGED:
            return TrainingStep(state, False, f"inference {result.status.value}", float("nan"), result)
        derivative = self._derivative(result)
        objective = derivative.value * self.config.objective_scale
        if derivative.status is not GradientStatus.CONVERGED:
            return TrainingStep(state, False, derivative.message, objective, result, derivative)
        config = self.config
        gradient = derivative.gradient * config.objective_scale
        first = config.beta1 * state.first_moment + (1 - config.beta1) * gradient
        second = config.beta2 * state.second_moment + (1 - config.beta2) * jnp.square(gradient)
        step = state.step + 1
        first_hat = first / (1 - config.beta1 ** step)
        second_hat = second / (1 - config.beta2 ** step)
        # Keep Adam's denominator representable at zero gradient; no model covariance is altered.
        epsilon = jnp.sqrt(jnp.asarray(np.finfo(self.dtype).eps, self.dtype))
        weights = state.weights - config.learning_rate * first_hat / (jnp.sqrt(second_hat) + epsilon)
        if not all(np.isfinite(np.asarray(value)).all() for value in (weights, first, second)):
            return TrainingStep(state, False, "non-finite optimizer update", objective, result, derivative)
        updated = TrainingState(self.identity, weights, first, second, step)
        return TrainingStep(updated, True, "certified gradient applied", objective, result, derivative)

    def save_checkpoint(self, path, state):
        self._validate(state)
        write_artifact(path, self._metadata, {
            "weights": state.weights, "first_moment": state.first_moment,
            "second_moment": state.second_moment, "step": np.asarray(state.step, dtype=np.int64),
        })

    def load_checkpoint(self, path):
        metadata, arrays = read_artifact(path, format_name="cffg.neural.adam",
                                         fields=("weights", "first_moment", "second_moment", "step"))
        if metadata != self._metadata:
            raise ValueError("neural checkpoint model, data, parameter layout or configuration mismatch")
        step = arrays["step"]
        if step.shape != () or step.dtype != np.dtype("int64"):
            raise ValueError("optimizer step must be a scalar int64")
        state = TrainingState(self.identity, *(jnp.asarray(arrays[name]) for name in
                              ("weights", "first_moment", "second_moment")), int(step))
        # Check stored dtypes before JAX can silently downcast a foreign checkpoint.
        if any(arrays[name].dtype != self.dtype for name in ("weights", "first_moment", "second_moment")):
            raise ValueError("optimizer checkpoint dtype mismatch")
        self._validate(state)
        return state

    def save_bundle(self, path, state):
        self._validate(state)
        arrays = {**self.arrays, **self.parameters.values(state.weights)}
        save_bundle(path, self.template, arrays)
