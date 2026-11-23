"""Frozen positive-noise neural regression problem and independent baseline."""

import jax
import jax.numpy as jnp
import numpy as np

from cffg import initialize_neural_parameters, neural_mean
from cffg.core import BlobRef, FactorSpec, MomentProjection, PortRole, PortSpec, Template, VariableSpec, moment_matching
from cffg.neural import neural_value
from cffg.neural_schema import neural_architecture


def key(seed):
    return jax.random.key(seed, impl="threefry2x32")


def regression_model(count=17):
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    centers = jnp.linspace(-1.6, 1.6, count, dtype=dtype)
    latent = centers + .04 * jax.random.normal(key(3101), (count,), dtype=dtype)
    outputs = jnp.tanh(2 * latent) + .08 * jax.random.normal(key(3102), (count,), dtype=dtype)
    function = neural_mean((1, 2, 1))
    arrays = {"P": np.array([[.04 ** 2]]), "V": np.array([[.08 ** 2]]),
              "theta": np.asarray(initialize_neural_parameters(function, key(71), dtype=dtype))}
    factors, variables, ports, constraints, observations = {}, {}, {}, [], {}
    for i in range(count):
        x, y, prior, sensor, mean = (f"{name}{i}" for name in ("x", "y", "prior", "sensor", "m"))
        arrays[mean] = np.asarray(centers[i:i + 1])
        variables[x], variables[y] = VariableSpec(x, dim=1), VariableSpec(y, dim=1)
        factors[prior] = FactorSpec(prior, "gaussian", {"out": x}, param_blobs={"mean": mean, "cov": "P"})
        factors[sensor] = FactorSpec(sensor, "gaussian_nonlinear", {"in": x, "out": y},
                                     param_blobs={"parameters": "theta", "cov": "V"}, function=function)
        ports[y] = PortSpec(y, y, PortRole.DATA)
        observations[y] = np.asarray(outputs[i:i + 1])
        constraints.append(moment_matching(sensor, "in", MomentProjection.GAUSSIAN))
    template = Template("neural-regression", variables=variables, factors=factors, ports=ports,
                        constraints=constraints,
                        blobs={name: BlobRef(name, name + ".npy") for name in arrays})
    return template, arrays, observations, np.asarray(centers), np.asarray(outputs)


def mean_values(weights, x):
    architecture = neural_architecture(neural_mean((1, 2, 1)).configuration)
    return np.asarray(jax.vmap(lambda value: neural_value(value[None], weights, architecture)[0])(jnp.asarray(x)))


def affine_baseline(x, y):
    u, v = x - x.mean(), y - y.mean()
    m, c, q = np.mean(u * u), np.mean(u * v), np.mean(v * v)
    p, noise = .04 ** 2, .08 ** 2
    roots = np.roots([p * p, p * c, noise * (p + m) - p * q, -noise * c])
    candidates = [(float(a.real), float(y.mean() - a.real * x.mean()))
                  for a in roots if abs(a.imag) <= 64 * np.finfo(x.dtype).eps * (1 + abs(a.real))]

    def objective(pair):
        a, b = pair
        variance = noise + p * a * a
        return .5 * np.mean(np.log(2 * np.pi * variance) + (y - a * x - b) ** 2 / variance)

    return min(candidates, key=objective)
