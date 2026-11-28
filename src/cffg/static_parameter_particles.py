"""Sequential Monte Carlo for the bounded static parameter target in D6d."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from threading import RLock

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit, log_expit, logsumexp

from cffg.core.functions import FunctionSpec
from cffg.finite_hypotheses import ImpossibleEvidence, _condition, _digest, _finite, _precision
from cffg.gaussian_taps import negative_log_density
from cffg.mean_functions import resolve_mean_function
from cffg.neural_artifacts import read_artifact, write_artifact

__all__ = ["ParticleSupportError", "ParticleUpdate", "StaticParameterParticles"]

_FORMAT = "cffg.static_parameter_particles"
_MODEL_FIELDS = {"bounds", "beta_shapes", "covariance"}
_FIELDS = _MODEL_FIELDS | {"z", "log_weights", "key", "inputs", "observations"}


class ParticleSupportError(ValueError):
    """The current population gives a zero evidence estimate for one event."""


@dataclass(frozen=True)
class ParticleUpdate:
    """Evidence increment, pre-resampling ESS, ancestry, and move accept counts."""

    log_evidence_increment: float
    ess: float
    ancestors: tuple[int, ...] | None
    accepted: tuple[int, ...]


@lru_cache(maxsize=32)
def _kernels(function, parameter_shape, dimension, dtype):
    evaluate = resolve_mean_function(function).evaluate
    configuration = function.configuration

    def score(parameters, x, y, precision):
        def mean(theta):
            return jnp.asarray(evaluate(x, theta.reshape(parameter_shape), configuration))

        means = jax.vmap(mean)(parameters)
        if means.shape != (len(parameters), dimension) or means.dtype.kind not in "fiu":
            raise ValueError("registered mean must return a real vector of the observation dimension")
        return -negative_log_density(y, means.astype(dtype), precision)

    @jax.jit
    def move_delta(parameters, inputs, observations, count, precision):
        def step(i, carry):
            total, valid = carry
            scores = score(parameters, inputs[i], observations[i], precision).astype(jnp.float64)
            old, new = scores.reshape(2, -1)
            return total + (new - old), valid & jnp.isfinite(scores).all()

        return jax.lax.fori_loop(0, count, step, (jnp.zeros(len(parameters) // 2, jnp.float64), True))

    return jax.jit(score), move_delta


def _checked_count(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _key_data(key):
    if (not isinstance(key, jax.Array) or key.shape != ()
            or not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)):
        raise ValueError("key must be a scalar typed JAX random key")
    return np.asarray(jax.random.key_data(key))


def _weight_update(log_weights, scores):
    if not np.all(np.isfinite(scores) | np.isneginf(scores)):
        raise FloatingPointError("invalid log likelihood")
    try:
        return _condition(log_weights, scores)
    except ImpossibleEvidence as error:
        raise ParticleSupportError("zero evidence estimate on the sampled population") from error


def _log_ess(log_weights):
    if np.all(log_weights == log_weights[0]):
        return math.log(len(log_weights))
    with np.errstate(over="ignore"):
        # A doubled negative log weight can round to -inf. Its squared mass
        # is already below the range of the accumulation dtype.
        return min(math.log(len(log_weights)), float(-logsumexp(2 * log_weights)))


def _stratified(log_weights, key):
    n = len(log_weights)
    cdf = np.cumsum(np.exp(log_weights))
    cdf /= cdf[-1]
    points = (np.arange(n) + np.asarray(jax.random.uniform(key, (n,), dtype=jnp.float64))) / n
    # Roundoff in the last stratum must not move a sample outside [0, 1).
    return np.searchsorted(cdf, np.minimum(points, np.nextafter(1., 0.)), side="right")


class StaticParameterParticles:
    """Approximate a posterior on static parameters given applied inputs and readings.

    ``bounds`` and ``beta_shapes`` are [P,2] arrays in flattened registered
    parameter order. ``covariance`` is the known conditional observation noise.
    The registered mean is evaluated directly given the applied input and each
    physical parameter row. The likelihood uses ``dtype``. Logit coordinates,
    log weights, and evidence use float64. ``proposal_scale`` is a fixed
    standard deviation in logit coordinates, shared across coordinates.
    """

    def __init__(self, function, bounds, beta_shapes, covariance, *, input_dim,
                 n_particles, key, ess_fraction=0.5, proposal_scale=1., move_steps=1, dtype=None):
        _key_data(key)
        self._configure(function, bounds, beta_shapes, covariance, input_dim, n_particles,
                        ess_fraction, proposal_scale, move_steps, dtype)
        with jax.enable_x64(True):
            self._key, left_key, right_key = jax.random.split(key, 3)
            # A difference of log Gamma draws has the beta distribution in
            # logit coordinates without rounding a sampled fraction to 0 or 1.
            alpha, beta = self._beta_shapes.T
            shape = (self._size, len(alpha))
            self._z = np.array(jax.random.loggamma(left_key, alpha, shape, jnp.float64)
                               - jax.random.loggamma(right_key, beta, shape, jnp.float64))
        self._parameters(self._z)
        self._log_weights = np.full(self._size, -math.log(self._size))
        self._log_evidence = 0.
        self._count = 0
        self._inputs = np.zeros((8, self._input_dim), self._dtype)
        self._observations = np.zeros((8, self._dimension), self._dtype)

    def _configure(self, function, bounds, beta_shapes, covariance, input_dim, n_particles,
                   ess_fraction, proposal_scale, move_steps, dtype):
        if not isinstance(function, FunctionSpec):
            raise ValueError("the static target requires a registered FunctionSpec")
        impl = resolve_mean_function(function)
        if impl.version != function.version:
            raise ValueError("static parameter target requires the exact registered mean version")
        self._function = function
        self._input_dim = _checked_count(input_dim, "input_dim", 1)
        self._size = _checked_count(n_particles, "n_particles", 1)
        self._move_steps = _checked_count(move_steps, "move_steps", 0)
        scalars = _finite([ess_fraction, proposal_scale], np.float64, "ESS fraction and proposal scale")
        if scalars.shape != (2,) or not (0 < scalars[0] <= 1 and scalars[1] > 0):
            raise ValueError("ESS fraction must be in (0, 1] and proposal scale strictly positive")
        self._ess_fraction, self._proposal_scale = map(float, scalars)
        default = np.float64 if jax.config.x64_enabled else np.float32
        self._dtype = np.dtype(default if dtype is None else dtype)
        if self._dtype not in (np.float32, np.float64):
            raise ValueError("likelihood dtype must be float32 or float64")

        self._covariance = _finite(covariance, self._dtype, "covariance")
        if (self._covariance.ndim != 2 or not self._covariance.shape[0]
                or self._covariance.shape[0] != self._covariance.shape[1]):
            raise ValueError("covariance must be a nonempty square matrix")
        self._dimension = len(self._covariance)
        self._precision = _precision(self._covariance, self._dtype)
        shape = tuple(impl.parameter_shape(self._input_dim, self._dimension, function.configuration))
        self._bounds = _finite(bounds, np.float64, "bound endpoints")
        self._beta_shapes = _finite(beta_shapes, np.float64, "beta shape parameters")
        if (self._bounds.shape != (math.prod(shape), 2) or not len(self._bounds)
                or self._beta_shapes.shape != self._bounds.shape or np.any(self._beta_shapes <= 0)):
            raise ValueError("bounds and positive beta shapes must have two entries per parameter")
        lo, hi = _finite(self._bounds, self._dtype, "bound endpoints").T
        with np.errstate(over="ignore"):
            self._width = self._bounds[:, 1] - self._bounds[:, 0]
        if np.any(np.nextafter(lo, hi) >= hi) or not np.isfinite(self._width).all():
            raise ValueError("bounds must enclose a representable open finite interval")
        self._score, self._move_delta = _kernels(function, shape, self._dimension, self._dtype)
        self._lock = RLock()

    def _parameters(self, z):
        lo, hi = self._bounds.T
        tail = self._width * expit(-np.abs(z))
        theta = np.where(z >= 0, hi - tail, lo + tail).astype(self._dtype)
        if (not np.isfinite(z).all() or not np.isfinite(theta).all()
                or np.any((theta <= lo) | (theta >= hi))):
            raise FloatingPointError("parameter left the representable interior of the declared bounds")
        return theta

    @property
    def particles(self):
        """Physical parameter rows at the observation dtype."""
        with self._lock:
            return self._parameters(self._z)

    @property
    def log_weights(self):
        with self._lock:
            return self._log_weights.copy()

    @property
    def log_evidence(self):
        with self._lock:
            return self._log_evidence

    @property
    def n_observed(self):
        with self._lock:
            return self._count

    def _stage_reading(self, x, y):
        inputs, observations = self._inputs, self._observations
        if self._count == len(inputs):
            # Round capacity up geometrically to reuse the compiled replay.
            inputs = np.concatenate((inputs, np.zeros_like(inputs)))
            observations = np.concatenate((observations, np.zeros_like(observations)))
        inputs[self._count], observations[self._count] = x, y
        return inputs, observations

    def _move(self, z, parameters, log_weights, key, inputs, observations):
        proposal_key, accept_key = jax.random.split(key)
        with np.errstate(over="raise", invalid="raise", divide="ignore"):
            proposal = z + self._proposal_scale * np.asarray(
                jax.random.normal(proposal_key, z.shape, jnp.float64)
            )
            proposed_parameters = self._parameters(proposal)
            delta, valid = self._move_delta(np.concatenate((parameters, proposed_parameters)),
                                            inputs, observations, self._count + 1, self._precision)
            alpha, beta = self._beta_shapes.T
            log_ratio = np.asarray(delta) + np.sum(
                alpha * (log_expit(proposal) - log_expit(z))
                + beta * (log_expit(-proposal) - log_expit(-z)), axis=-1
            )
            if not bool(valid) or not np.isfinite(log_ratio).all():
                raise FloatingPointError("Metropolis target is not representable")
            u = np.asarray(jax.random.uniform(accept_key, (self._size,), dtype=jnp.float64))
            accept = np.isfinite(log_weights) & (np.log(u) < log_ratio)
        return (np.where(accept[:, None], proposal, z),
                np.where(accept[:, None], proposed_parameters, parameters), int(accept.sum()))

    def update(self, applied_input, observation):
        """Condition on one paired applied input and reading, atomically.

        ``observation=None`` retains the state and random key. Invalid or
        unrepresentable evaluations raise, leaving the last committed state.
        """
        with self._lock, jax.enable_x64(True), np.errstate(over="raise", invalid="raise"):
            if observation is None:
                return ParticleUpdate(0., math.exp(_log_ess(self._log_weights)), None, ())
            x = _finite(applied_input, self._dtype, "applied input")
            y = _finite(observation, self._dtype, "observation")
            if x.shape != (self._input_dim,) or y.shape != (self._dimension,):
                raise ValueError("applied input and observation must match the declared vector dimensions")
            z, parameters = self._z, self._parameters(self._z)
            scores = np.asarray(self._score(parameters, x, y, self._precision), dtype=np.float64)
            # Every proper Gaussian with a finite mean has positive density.
            # A nonfinite score here is an evaluation or numeric range error.
            if not np.isfinite(scores).all():
                raise FloatingPointError("Gaussian likelihood is not representable")
            logs, increment = _weight_update(self._log_weights, scores)
            evidence = math.fsum((self._log_evidence, increment))
            if not math.isfinite(evidence):
                raise FloatingPointError("accumulated log evidence is not representable")

            log_ess = _log_ess(logs)
            next_key, resample_key, move_key = jax.random.split(self._key, 3)
            ancestors = None
            if log_ess < math.log(self._size * self._ess_fraction):
                indices = _stratified(logs, resample_key)
                ancestors = tuple(indices.tolist())
                z, parameters = z[indices], parameters[indices]
                logs = np.full(self._size, -math.log(self._size))
            inputs, observations = self._stage_reading(x, y)
            accepted = []
            for key in jax.random.split(move_key, self._move_steps):
                z, parameters, count = self._move(z, parameters, logs, key, inputs, observations)
                accepted.append(count)

            self._z, self._log_weights, self._key, self._log_evidence = z, logs, next_key, evidence
            self._inputs, self._observations = inputs, observations
            self._count += 1
            return ParticleUpdate(increment, math.exp(log_ess), ancestors, tuple(accepted))

    def save(self, path):
        """Atomically save the target, applied history, and complete particle carry."""
        with self._lock:
            meta = dict(format=_FORMAT, version=1, function=self._function.to_dict(),
                        input_dim=self._input_dim, n_particles=self._size, dtype=self._dtype.name,
                        ess_fraction=self._ess_fraction, proposal_scale=self._proposal_scale,
                        move_steps=self._move_steps, key_impl=str(jax.random.key_impl(self._key)),
                        n_observed=self._count, log_evidence=self._log_evidence)
            arrays = dict(bounds=self._bounds, beta_shapes=self._beta_shapes, covariance=self._covariance,
                          z=self._z, log_weights=self._log_weights, key=_key_data(self._key),
                          inputs=self._inputs[:self._count], observations=self._observations[:self._count])
            meta["sha256"] = _digest(meta, arrays)
            write_artifact(path, meta, arrays)

    @classmethod
    def load(cls, path):
        """Restore the declared target and carry without drawing new particles."""
        meta, arrays = read_artifact(path, format_name=_FORMAT, fields=_FIELDS)
        if set(meta) != {"format", "version", "function", "input_dim", "n_particles", "dtype",
                         "ess_fraction", "proposal_scale", "move_steps", "key_impl",
                         "n_observed", "log_evidence", "sha256"}:
            raise ValueError("incomplete or unknown static particle checkpoint metadata")
        if meta.pop("sha256") != _digest(meta, arrays):
            raise ValueError("static particle checkpoint digest mismatch")
        if meta["dtype"] not in ("float32", "float64"):
            raise ValueError("invalid static particle checkpoint likelihood dtype")
        bank = cls.__new__(cls)
        bank._configure(FunctionSpec.from_dict(meta["function"]), arrays["bounds"], arrays["beta_shapes"],
                        arrays["covariance"], meta["input_dim"], meta["n_particles"], meta["ess_fraction"],
                        meta["proposal_scale"], meta["move_steps"], meta["dtype"])
        for name in _MODEL_FIELDS:
            if arrays[name].dtype != getattr(bank, "_" + name).dtype:
                raise ValueError("incompatible static particle target dtype")
        z, logs = arrays["z"], arrays["log_weights"]
        if z.shape != (bank._size, len(bank._bounds)) or z.dtype != np.float64:
            raise ValueError("invalid transformed particle state")
        bank._parameters(z)
        if (logs.shape != (bank._size,) or logs.dtype != np.float64
                or not np.all(np.isfinite(logs) | np.isneginf(logs))
                or abs(logsumexp(logs)) > 32 * np.finfo(logs.dtype).eps):
            raise ValueError("invalid normalized particle log weights")
        count = _checked_count(meta["n_observed"], "n_observed", 0)
        evidence = meta["log_evidence"]
        if (type(evidence) is not float or not math.isfinite(evidence)
                or (count == 0 and (evidence != 0. or np.any(logs != -math.log(bank._size))))):
            raise ValueError("invalid static particle checkpoint evidence")
        inputs, observations = arrays["inputs"], arrays["observations"]
        for rows, width in ((inputs, bank._input_dim), (observations, bank._dimension)):
            if rows.shape != (count, width) or rows.dtype != bank._dtype or not np.isfinite(rows).all():
                raise ValueError("invalid committed observation history")
        if arrays["key"].dtype != np.uint32 or not isinstance(meta["key_impl"], str):
            raise ValueError("invalid static particle checkpoint key")
        try:
            bank._key = jax.random.wrap_key_data(arrays["key"], impl=meta["key_impl"])
            _key_data(bank._key)
            if str(jax.random.key_impl(bank._key)) != meta["key_impl"]:
                raise ValueError("checkpoint key implementation must be canonical")
        except (TypeError, ValueError) as error:
            raise ValueError("invalid static particle checkpoint key") from error

        capacity = 1 << max(3, (count - 1).bit_length())
        bank._inputs = np.zeros((capacity, bank._input_dim), bank._dtype)
        bank._observations = np.zeros((capacity, bank._dimension), bank._dtype)
        bank._inputs[:count], bank._observations[:count] = inputs, observations
        bank._z, bank._log_weights, bank._log_evidence, bank._count = z, logs, evidence, count
        return bank
