"""Host-only neural architecture and flat parameter layout."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cffg.core.functions import FunctionSpec


@dataclass(frozen=True)
class NeuralArchitecture:
    widths: tuple[int, ...]
    activation: str

    @property
    def parameter_count(self) -> int:
        return sum(output * (input_ + 1) for input_, output in zip(self.widths, self.widths[1:]))

    @property
    def dense_products(self) -> int:
        return sum(input_ * output for input_, output in zip(self.widths, self.widths[1:]))


def neural_architecture(configuration: Mapping) -> NeuralArchitecture:
    """Every layer packs its row-major matrix then bias; the output is affine."""
    if set(configuration) != {"widths", "activation"}:
        raise ValueError("neural architecture requires exactly widths and activation")
    widths = configuration["widths"]
    if not isinstance(widths, (tuple, list)) or len(widths) < 2:
        raise ValueError("neural widths must include input and output dimensions")
    if any(not isinstance(width, int) or isinstance(width, bool) or width <= 0 for width in widths):
        raise ValueError("neural widths must be positive integers")
    activation = configuration["activation"]
    if activation not in ("tanh", "identity"):
        raise ValueError("neural hidden activation must be tanh or identity")
    return NeuralArchitecture(tuple(widths), activation)


def neural_mean(widths: Sequence[int], *, activation="tanh", method="linearization") -> FunctionSpec:
    """Declare a portable Gaussian mean network, separately from weights and noise."""
    configuration = {"widths": list(widths), "activation": activation}
    neural_architecture(configuration)
    return FunctionSpec("cffg.mean.mlp", "1.0.0", configuration=configuration, method=method)


def parameter_shape(input_dim, output_dim, configuration):
    architecture = neural_architecture(configuration)
    if (architecture.widths[0], architecture.widths[-1]) != (input_dim, output_dim):
        raise ValueError("neural architecture input/output dimensions do not match Gaussian edges")
    return (architecture.parameter_count,)


def evaluate(x, parameters, configuration):
    # Resolution and inventory can import this callback without importing JAX.
    from cffg.neural import neural_value

    return neural_value(x, parameters, neural_architecture(configuration))


def work(input_dim, output_dim, configuration):
    parameter_shape(input_dim, output_dim, configuration)
    architecture = neural_architecture(configuration)
    return {
        "network_parameters": architecture.parameter_count,
        "dense_products_per_value": architecture.dense_products,
        "hidden_activations_per_value": sum(architecture.widths[1:-1]) if architecture.activation == "tanh" else 0,
    }
