"""Supported pre-1.0 facade for the documented newcomer workflow."""

from cffg.agent import RollingAgent
from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus
from cffg.models.gridworld import CorridorEnv, build_gridworld
from cffg.params import set_params
from cffg.kernels import FactorKernel, HostileInputs, register_kernel
from cffg.kernel_bindings import (
    DiscreteBinding, DiscreteMode, EFBinding, GaussianBinding,
    GaussianConditioning, ParameterSpec,
)

__all__ = [
    "CorridorEnv",
    "DiscreteBinding",
    "DiscreteMode",
    "EFBinding",
    "Engine",
    "FactorKernel",
    "GaussianBinding",
    "GaussianConditioning",
    "HostileInputs",
    "ParameterSpec",
    "RollingAgent",
    "RunStatus",
    "build_gridworld",
    "compile_template",
    "load_bundle",
    "save_bundle",
    "register_kernel",
    "set_params",
]
