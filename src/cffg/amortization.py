"""Deterministic polynomial recognition of Gaussian message starting points.

The learned output layer predicts message coordinates from fixed polynomial
features. It supplies no importance proposal or particles. Proposed messages
must be proper without clipping, and ordinary inference certifies every root.
"""

from dataclasses import asdict, dataclass
import json

import jax
import jax.numpy as jnp
import numpy as np

from cffg.compiler import compile_template
from cffg.core.serialize import structure_hash
from cffg.engine import Engine
from cffg.fixed_point import RootProgram, root_context
from cffg.neural_artifacts import array_digest, read_artifact, write_artifact
from cffg.outer import OuterArrays, build_outer_chart, chart_coordinates
from cffg.runtime_types import State, _replace


@dataclass(frozen=True)
class PolynomialFeatures:
    ports: tuple[str, ...]
    weights: tuple[float, ...]
    center: float = 0.
    scale: float = 1.
    degree: int = 7

    def __post_init__(self):
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "weights", tuple(self.weights))
        if not self.ports or len(set(self.ports)) != len(self.ports) or len(self.ports) != len(self.weights):
            raise ValueError("features require distinct scalar Gaussian ports and matching weights")
        if not np.isfinite([self.center, self.scale, *self.weights]).all() or self.scale <= 0:
            raise ValueError("feature coefficients must be finite with a positive scale")
        if type(self.degree) is not int or self.degree < 0:
            raise ValueError("polynomial degree must be a nonnegative integer")

    def evaluate(self, values):
        """Pure JAX feature map, also usable under jit and vmap."""
        value = (jnp.dot(values, jnp.asarray(self.weights, values.dtype)) - self.center) / self.scale
        return value ** jnp.arange(self.degree + 1)


@dataclass(frozen=True)
class InitializationProposal:
    state: State
    accepted: bool
    message: str


_REFERENCE_FIELDS = ("messages", "g_eta", "g_lam", "ef_eta", "g_obs", "g_mask", "ef_obs", "ef_mask")
_FIELDS = (*_REFERENCE_FIELDS, "coefficients", "output_center", "output_scale")


class GaussianAmortizer:
    """A learned initializer bound to one fixed model and observation-mask layout."""

    @staticmethod
    def _setup(template, arrays, features, tolerance, n_iters):
        if not np.isfinite(tolerance) or tolerance <= 0 or type(n_iters) is not int or n_iters <= 0:
            raise ValueError("amortizer requires a positive finite tolerance and iteration budget")
        engine = Engine(compile_template(template, arrays=arrays))
        if (not engine.has_adaptive_gaussian or engine.arena.semiring != "sum"
                or engine.arena.mu3_groups or len(engine.arena.delta_pairs)):
            raise ValueError("amortizer requires a deterministic smooth adaptive Gaussian graph")
        for name in features.ports:
            if name not in engine.arena.gauss_data or engine.arena.gauss_data[name][1] != 1:
                raise ValueError(f"feature port {name!r} must be scalar Gaussian data")
        return engine

    @classmethod
    def fit(cls, template, arrays, observations, *, features, reference_index=0, tolerance=1e-8, n_iters=25):
        observations = tuple(observations)
        if type(reference_index) is not int or not 0 <= reference_index < len(observations):
            raise ValueError("reference index must name a training observation")
        engine = cls._setup(template, arrays, features, tolerance, n_iters)
        targets, states = [], []
        for data in observations:
            initial = engine.bind_data(engine.init_state(), data)
            states.append(engine.run(initial, n_iters, tol=tolerance).require_converged())
        result = cls.__new__(cls)
        result._initialize(engine, template, arrays, features, states[reference_index], tolerance, n_iters)
        for state in states:
            result._check_context(state)
            targets.append(np.asarray(chart_coordinates(result.chart, result._arrays(state)))[result.indices])
        inputs = np.stack([np.asarray(features.evaluate(result._values(state))) for state in states])
        targets = np.stack(targets)
        result.output_center = jnp.asarray(targets.mean(axis=0))
        spread = targets.std(axis=0)
        # A constant output needs no normalization; this changes no model covariance.
        result.output_scale = jnp.asarray(np.where(spread == 0, 1., spread))
        normalized = (targets - np.asarray(result.output_center)) / np.asarray(result.output_scale)
        threshold = np.finfo(inputs.dtype).eps * max(inputs.shape)
        coefficients, _, rank, _ = np.linalg.lstsq(inputs, normalized, rcond=threshold)
        if rank != inputs.shape[1]:
            raise ValueError("recognition features are rank deficient; provide independent training inputs")
        result.coefficients = jnp.asarray(coefficients)
        result._validate_parameters()
        return result

    @staticmethod
    def _arrays(state):
        return OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)

    def _initialize(self, engine, template, arrays, features, reference, tolerance, n_iters):
        self.engine, self.features, self.reference = engine, features, reference
        self.tolerance, self.n_iters = tolerance, n_iters
        self.chart = build_outer_chart(engine.arena, self._arrays(reference), engine._outer_eliminated)
        if not self.chart.proper_reference:
            raise ValueError("amortizer reference requires proper Gaussian beliefs")
        self.dtype = np.asarray(reference.g_eta).dtype
        indices, offset = [], sum(block.coordinates.size for block in self.chart.categorical)
        for block in self.chart.vectors:
            if block.field == "g_eta":
                indices.extend(range(offset, offset + block.indices.size))
            offset += block.indices.size
        for block in self.chart.precisions:
            indices.extend(range(offset, offset + len(block.lower)))
            offset += len(block.lower)
        self.indices = np.asarray(indices, dtype=np.int64)
        if not self.indices.size:
            raise ValueError("amortizer has no free Gaussian message coordinates")
        self.rows = np.asarray([engine.arena.gauss_data[name][0] for name in features.ports])
        self.observation_indices = engine.arena.g_obs_indices(self.rows, 1).reshape(-1)
        self.parameter_digest = array_digest(reference.params)
        self._metadata = json.loads(json.dumps({
            "format": "cffg.neural.recognition", "version": 1,
            "structure": structure_hash(template), "arrays": array_digest(arrays),
            "features": asdict(features), "coordinates": indices, "dtype": self.dtype.str,
            "tolerance": tolerance, "iterations": n_iters,
        }, sort_keys=True))
        self.program = RootProgram(self.chart, engine._iter_step, engine._reconstruct)
        self._decode = jax.jit(self.program.arrays)
        self._check_context(reference)

    def _values(self, state):
        return state.g_obs[self.observation_indices]

    def _check_context(self, state):
        if array_digest(state.params) != self.parameter_digest:
            raise ValueError("amortizer model parameters changed")
        for name in ("g_mask", "ef_mask"):
            if not np.array_equal(getattr(state, name), getattr(self.reference, name)):
                raise ValueError("amortizer observation-mask layout changed")
        if not np.all(np.asarray(state.g_mask)[self.rows] == 1):
            raise ValueError("all recognition feature ports must be observed")
        if self.chart.support_changed(state.messages):
            raise ValueError("amortizer reference exact support changed")
        if np.asarray(state.g_eta).dtype != self.dtype or not self.engine._state_is_finite(state):
            raise ValueError("amortizer state dtype or finite-value mismatch")

    def _validate_parameters(self):
        for name, shape in (("coefficients", (self.features.degree + 1, self.indices.size)),
                            ("output_center", (self.indices.size,)), ("output_scale", (self.indices.size,))):
            values = np.asarray(getattr(self, name))
            if values.shape != shape or values.dtype != self.dtype or not np.isfinite(values).all():
                raise ValueError(f"recognition {name} shape, dtype or finite-value mismatch")
        if np.any(np.asarray(self.output_scale) <= 0):
            raise ValueError("recognition output scales must be positive")

    def predict_coordinates(self, observation_values):
        """Evaluate the trained output layer; accepts jit/vmap over feature-port vectors."""
        return self.output_center + self.output_scale * (self.features.evaluate(observation_values) @ self.coefficients)

    def propose(self, state):
        try:
            self._check_context(state)
            self._validate_parameters()
            arrays = self._arrays(state)
            coordinates = chart_coordinates(self.chart, arrays)
            coordinates = coordinates.at[self.indices].set(self.predict_coordinates(self._values(state)))
            context = root_context(self.chart, state)._replace(base=arrays._replace(ef_eta=state.ef_eta.reshape(-1)))
            proposed = self._decode(coordinates, context)
            # Only Gaussian messages are owned by this initializer.
            candidate = _replace(state, g_eta=proposed.g_eta, g_lam=proposed.g_lam)
            if not self.engine._state_is_finite(candidate) or self.engine._has_zero_support(candidate):
                raise ValueError("recognition proposal is non-finite or has empty exact support")
            chart = build_outer_chart(self.engine.arena, self._arrays(candidate), self.engine._outer_eliminated)
            if not chart.proper_reference or not np.isfinite(self.engine.bfe(candidate)):
                raise ValueError("recognition proposal has improper beliefs or cavities")
            _, _, _, _, local, status = self.engine._iter_step(
                candidate.messages, candidate.g_eta, candidate.g_lam, candidate.g_obs, candidate.g_mask,
                candidate.ef_eta, candidate.ef_obs, candidate.ef_mask, candidate.params, 0., False)
            if int(status) or not np.isfinite(float(local)):
                raise ValueError("recognition proposal has invalid local updates")
        except (ValueError, np.linalg.LinAlgError) as error:
            return InitializationProposal(state, False, str(error))
        return InitializationProposal(candidate, True, "proper deterministic initialization proposed")

    def run(self, observations):
        initial = self.engine.bind_data(self.engine.init_state(), observations)
        proposal = self.propose(initial)
        return proposal, self.engine.run(proposal.state, self.n_iters, tol=self.tolerance)

    def save(self, path):
        self._validate_parameters()
        arrays = {name: getattr(self.reference, name) for name in _REFERENCE_FIELDS}
        arrays.update({name: getattr(self, name) for name in ("coefficients", "output_center", "output_scale")})
        write_artifact(path, self._metadata, arrays)

    @classmethod
    def load(cls, path, template, arrays):
        metadata, values = read_artifact(path, format_name="cffg.neural.recognition", fields=_FIELDS)
        if set(metadata) != {"format", "version", "structure", "arrays", "features", "coordinates",
                             "dtype", "tolerance", "iterations"}:
            raise ValueError("recognition checkpoint metadata inventory mismatch")
        features = PolynomialFeatures(**metadata["features"])
        engine = cls._setup(template, arrays, features, metadata["tolerance"], metadata["iterations"])
        initial = engine.init_state()
        for name in _REFERENCE_FIELDS:
            expected = np.asarray(getattr(initial, name))
            if values[name].shape != expected.shape or values[name].dtype != expected.dtype or not np.isfinite(values[name]).all():
                raise ValueError(f"recognition reference {name} shape, dtype or finite-value mismatch")
        reference = _replace(initial, **{name: jnp.asarray(values[name]) for name in _REFERENCE_FIELDS})
        result = cls.__new__(cls)
        result._initialize(engine, template, arrays, features, reference, metadata["tolerance"], metadata["iterations"])
        if metadata != result._metadata:
            raise ValueError("recognition model, architecture or coordinate schema mismatch")
        for name in ("coefficients", "output_center", "output_scale"):
            if values[name].dtype != result.dtype:
                raise ValueError("recognition checkpoint dtype mismatch")
            setattr(result, name, jnp.asarray(values[name]))
        result._validate_parameters()
        coordinates = chart_coordinates(result.chart, result._arrays(reference))
        facts = result.program.certificate(coordinates, root_context(result.chart, reference), engine._ledger.belief_kernel)
        energy, next_energy, residual, local, status, messages = facts
        tau = max(result.tolerance, 64 * np.finfo(result.dtype).eps)
        if (not np.isfinite([energy, next_energy, residual, local]).all() or int(status)
                or float(residual) > tau or abs(float(energy - next_energy)) > tau * (1 + abs(float(energy)))
                or result.chart.support_changed(messages)):
            raise ValueError("recognition reference is not a certified fixed point")
        return result
