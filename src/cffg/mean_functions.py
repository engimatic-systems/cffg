"""Host-only registry of reconstructible nonlinear mean implementations.

Declarations and shape providers do not need JAX. Numerical evaluation enters
only through a configured Gaussian kernel; weights are always dynamic arrays.
"""

from dataclasses import dataclass
from collections.abc import Callable

from cffg.core.functions import FunctionSpec


class MissingMeanImplementationError(ValueError):
    pass


class MeanVersionError(ValueError):
    pass


class MeanShapeError(ValueError):
    pass


@dataclass(frozen=True)
class MeanImplementation:
    impl_id: str
    version: str
    parameter_shape: Callable
    evaluate: Callable
    work: Callable | None = None


MEAN_IMPLEMENTATIONS: dict[str, MeanImplementation] = {}


def register_mean_function(implementation: MeanImplementation):
    FunctionSpec(implementation.impl_id, implementation.version)
    if not callable(implementation.parameter_shape) or not callable(implementation.evaluate):
        raise TypeError("mean implementation requires shape and evaluation callbacks")
    if implementation.work is not None and not callable(implementation.work):
        raise TypeError("mean work declaration must be callable")
    if implementation.impl_id in MEAN_IMPLEMENTATIONS:
        raise ValueError(f"mean implementation {implementation.impl_id!r} is already registered")
    MEAN_IMPLEMENTATIONS[implementation.impl_id] = implementation
    return implementation


def resolve_mean_function(spec: FunctionSpec) -> MeanImplementation:
    implementation = MEAN_IMPLEMENTATIONS.get(spec.impl_id)
    if implementation is None:
        raise MissingMeanImplementationError(f"missing mean implementation {spec.impl_id!r}")
    wanted = tuple(map(int, spec.version.split(".")))
    available = tuple(map(int, implementation.version.split(".")))
    if wanted[0] != available[0] or available < wanted:
        raise MeanVersionError(
            f"mean implementation {spec.impl_id!r}: version {implementation.version} "
            f"is incompatible with required version {spec.version}"
        )
    return implementation


def _affine_shape(di, do, configuration):
    if configuration:
        raise ValueError("affine mean has no configuration fields")
    return (do * (di + 1),)


def _affine(x, parameters, configuration):
    di = x.shape[0]
    do = parameters.shape[0] // (di + 1)
    return parameters[:do * di].reshape(do, di) @ x + parameters[do * di:]


def _cubic_shape(di, do, configuration):
    if configuration or di != do:
        raise MeanShapeError("elementwise cubic mean requires equal dimensions and no configuration")
    return (1,)


def _cubic(x, parameters, configuration):
    return x + parameters[0] * x ** 3


register_mean_function(MeanImplementation("cffg.mean.affine", "1.0.0", _affine_shape, _affine))
register_mean_function(MeanImplementation("cffg.mean.cubic", "1.0.0", _cubic_shape, _cubic))

from cffg import neural_schema as _neural_schema

register_mean_function(MeanImplementation(
    "cffg.mean.mlp", "1.0.0", _neural_schema.parameter_shape, _neural_schema.evaluate,
    _neural_schema.work,
))
