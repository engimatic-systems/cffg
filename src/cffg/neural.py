"""JAX evaluation and explicit-key initialization of portable mean networks."""

import jax
import jax.numpy as jnp

from cffg.core.functions import FunctionSpec
from cffg.neural_schema import NeuralArchitecture, neural_architecture


def neural_value(x, parameters, architecture: NeuralArchitecture):
    if parameters.shape != (architecture.parameter_count,):
        raise ValueError("neural parameter vector does not match the architecture")
    if x.shape != (architecture.widths[0],):
        raise ValueError("neural input vector does not match the architecture")
    offset = 0
    value = x
    for index, (input_, output) in enumerate(zip(architecture.widths, architecture.widths[1:])):
        count = input_ * output
        matrix = parameters[offset:offset + count].reshape(output, input_)
        offset += count
        value = matrix @ value + parameters[offset:offset + output]
        offset += output
        if index < len(architecture.widths) - 2 and architecture.activation == "tanh":
            value = jnp.tanh(value)
    return value


def initialize_neural_parameters(function: FunctionSpec, key, *, dtype=jnp.float32):
    """Glorot normal weights and zero biases; random choices use only the given key."""
    if function.impl_id != "cffg.mean.mlp":
        raise ValueError("neural initialization requires the cffg.mean.mlp implementation")
    from cffg.mean_functions import resolve_mean_function

    resolve_mean_function(function)
    architecture = neural_architecture(function.configuration)
    keys = jax.random.split(key, len(architecture.widths) - 1)
    values = []
    for key, input_, output in zip(keys, architecture.widths, architecture.widths[1:]):
        matrix = jax.random.normal(key, (output, input_), dtype)
        matrix *= jnp.sqrt(jnp.asarray(2 / (input_ + output), dtype))
        values.extend((matrix.reshape(-1), jnp.zeros(output, dtype)))
    return jnp.concatenate(values)
