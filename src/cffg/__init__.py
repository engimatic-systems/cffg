"""Supported pre-1.0 facade for the documented newcomer workflow."""

from cffg.agent import RollingAgent
from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus
from cffg.models.gridworld import CorridorEnv, build_gridworld
from cffg.params import set_params

__all__ = [
    "CorridorEnv",
    "Engine",
    "RollingAgent",
    "RunStatus",
    "build_gridworld",
    "compile_template",
    "load_bundle",
    "save_bundle",
    "set_params",
]
