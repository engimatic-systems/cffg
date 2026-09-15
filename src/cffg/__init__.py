"""Supported pre-1.0 facade for the documented newcomer workflow."""

from importlib import import_module


_EXPORT_MODULES = {
    "agent": ("RollingAgent",),
    "bundle": ("LoadedBundle", "load_bundle", "save_bundle"),
    "compiler": ("compile_template",),
    "core.functions": ("FunctionSpec",),
    "mean_functions": ("MeanImplementation", "register_mean_function"),
    "neural_schema": ("neural_mean",),
    "neural": ("initialize_neural_parameters",),
    "engine": ("Engine", "RunStatus"),
    "models.gridworld": ("CorridorEnv", "build_gridworld"),
    "params": ("set_params",),
    "learning_schema": (
        "PersistentArray", "PersistentParameter", "PersistentStateDescriptor",
        "PersistentTarget", "resolve_persistent_state",
    ),
    "kernels": ("FactorKernel", "HostileInputs", "register_kernel"),
    "kernel_bindings": (
        "DiscreteBinding", "DiscreteMode", "EFBinding", "GaussianBinding",
        "GaussianConditioning", "ParameterSpec",
    ),
}
_EXPORTS = {name: module for module, names in _EXPORT_MODULES.items() for name in names}


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

__all__ = [
    "CorridorEnv",
    "DiscreteBinding",
    "DiscreteMode",
    "EFBinding",
    "Engine",
    "FactorKernel",
    "FunctionSpec",
    "GaussianBinding",
    "GaussianConditioning",
    "HostileInputs",
    "LoadedBundle",
    "MeanImplementation",
    "ParameterSpec",
    "PersistentArray",
    "PersistentParameter",
    "PersistentStateDescriptor",
    "PersistentTarget",
    "RollingAgent",
    "RunStatus",
    "build_gridworld",
    "compile_template",
    "initialize_neural_parameters",
    "load_bundle",
    "neural_mean",
    "save_bundle",
    "register_kernel",
    "register_mean_function",
    "resolve_persistent_state",
    "set_params",
]
