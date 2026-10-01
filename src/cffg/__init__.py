"""Supported pre-1.0 facade for common runtime and extension workflows."""

from importlib import import_module


_EXPORTS = {
    "CorridorEnv": "models.gridworld",
    "DiscreteBinding": "kernel_bindings",
    "DiscreteMode": "kernel_bindings",
    "EFBinding": "kernel_bindings",
    "Engine": "engine",
    "FactorKernel": "kernels",
    "FixedPointDifferentiator": "implicit",
    "FixedPointGradient": "implicit",
    "FunctionSpec": "core.functions",
    "GaussianAmortizer": "amortization",
    "GaussianBinding": "kernel_bindings",
    "GaussianConditioning": "kernel_bindings",
    "GradientStatus": "implicit",
    "HostileInputs": "kernels",
    "LoadedBundle": "bundle",
    "MeanImplementation": "mean_functions",
    "NeuralTrainer": "training",
    "ParameterSpec": "kernel_bindings",
    "PersistentArray": "learning_schema",
    "PersistentParameter": "learning_schema",
    "PersistentStateDescriptor": "learning_schema",
    "PersistentTarget": "learning_schema",
    "PolynomialFeatures": "amortization",
    "RollingAgent": "agent",
    "RunStatus": "engine",
    "TrainingConfig": "training",
    "build_gridworld": "models.gridworld",
    "compile_template": "compiler",
    "fixed_point_gradient": "implicit",
    "initialize_neural_parameters": "neural",
    "load_bundle": "bundle",
    "neural_mean": "neural_schema",
    "save_bundle": "bundle",
    "register_kernel": "kernels",
    "register_mean_function": "mean_functions",
    "resolve_persistent_state": "learning_schema",
    "set_params": "params",
}


def __getattr__(name):
    """Keep core imports independent of JAX while retaining the public facade."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"cffg.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = list(_EXPORTS)
