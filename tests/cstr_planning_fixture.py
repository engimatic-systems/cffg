"""Retain the original wide-prior investigation inputs alongside the public recipe."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

_spec = spec_from_file_location(
    "cstr_planning_example", Path(__file__).resolve().parents[1] / "examples/cstr_planning.py",
)
_example = module_from_spec(_spec)
_spec.loader.exec_module(_example)
CONFIGURATION = _example.CONFIGURATION
NOMINAL_PARAMETERS = _example.NOMINAL_PARAMETERS
cstr_rk4 = _example.cstr_rk4
planning_model = _example.planning_model


def planning_fixture(method="linearization", *, past=4, horizon=15,
                     process_scale=1., temperature_variance=None, dtype=np.float64,
                     affine=False):
    return planning_model(
        method, past=past, horizon=horizon, process_scale=process_scale,
        temperature_variance=temperature_variance, dtype=dtype, affine=affine,
        prior_diagonal=(1., 1., 1e-4, 1.),
    )
