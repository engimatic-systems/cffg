"""The recent PCGym native P4/H15 validation, not the older short CSTR example.

Retained sources: common/physics.py and validation/v2_cstr_nonlinear_chain.py
in the 2026-10-01 PCGym validation. The closed-loop invocation/observations were
not retained. This fixture uses the explicit open-grid budget 100, its initial
observation, and labeled synthetic observation changes. It never warm-starts.
"""

from functools import cache

import jax.numpy as jnp
import numpy as np

from cffg import Engine, FunctionSpec, MeanImplementation, compile_template, register_mean_function
from cffg.core import (
    BlobRef, FactorSpec, MomentProjection, PortRole, PortSpec, Template,
    VariableSpec, moment_matching,
)
from cffg.runtime_types import _replace

CONFIGURATION = dict(x_lo=[.7, 300.], x_hi=[1., 350.], u_lo=295., u_hi=302.,
                     nd=1, tracked=[0], dt=26 / 60, substeps=4)
PARAMETERS = np.array([np.log(7.2e10 * np.exp(-8750. / 330.)), .875, np.log(5e4), np.log(5e4)])
PAST, HORIZON, BUDGET, TOLERANCE = 4, 15, 100, 1e-6


def cstr_rk4(s, theta, cfg):
    """Input clipping only; every RK4 stage retains the validation's physics."""
    lo, hi = jnp.asarray(cfg["x_lo"]), jnp.asarray(cfg["x_hi"])
    ulo, uhi = cfg["u_lo"], cfg["u_hi"]
    nx, nd = 2, cfg["nd"]
    span = hi - lo
    x = jnp.clip((s[:nx] + 1) / 2 * span + lo, lo - 2 * span, hi + 2 * span)
    x = x.at[0].set(jnp.maximum(x[0], 0.))
    u = jnp.clip((s[nx + nd] + 1) / 2 * (uhi - ulo) + ulo, ulo - 20., uhi + 20.)
    k_ref, e_r, ua, dh = jnp.exp(theta[0]), theta[1] * 1e4, jnp.exp(theta[2]), jnp.exp(theta[3])
    q, volume, rho, heat_capacity, feed, inlet = 100., 100., 1000., .239, 1., 350.

    def derivative(value):
        ca, temperature = value
        reaction = k_ref * jnp.exp(-e_r * (1. / temperature - 1. / 330.)) * ca
        return jnp.stack([
            q / volume * (feed - ca) - reaction,
            q / volume * (inlet - temperature) + dh * reaction / (rho * heat_capacity)
            + ua * (u - temperature) / (rho * heat_capacity * volume),
        ])

    step = cfg["dt"] / cfg["substeps"]
    for _ in range(cfg["substeps"]):
        k1 = derivative(x)
        k2 = derivative(x + .5 * step * k1)
        k3 = derivative(x + .5 * step * k2)
        k4 = derivative(x + step * k3)
        x = x + step / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
    x = 2 * (x - lo) / (hi - lo) - 1
    if nd:
        x = x.at[jnp.asarray(cfg["tracked"])].add(s[nx:nx + nd])
    return jnp.concatenate([x, s[nx:]])


def parameter_shape(di, do, configuration):
    if di != do:
        raise ValueError("physics transition maps the augmented state to itself")
    return (4,)


@cache
def install_mean():
    register_mean_function(MeanImplementation("pcgym.cstr.rk4", "1.0.0", parameter_shape, cstr_rk4,
                                             lambda di, do, cfg: {"network_parameters": 4}))


def model(method="linearization"):
    install_mean()
    template = Template(name="nlwindow_p4_h15_d4")
    identity = np.eye(4)
    noise = np.diag([1e-5, 1e-5, 4e-6, .25])
    arrays = dict(theta=PARAMETERS.copy(), Q=noise, Qp=noise.copy(), m0=np.zeros(4),
                  P0=np.diag([1., 1., 1e-4, 1.]),
                  G_past=identity[[0, 1, 3]], R_past=np.diag([1e-4, 1e-4, 1e-6]),
                  G_now=identity[[0, 1]], R_now=np.eye(2) * 1e-4,
                  G_goal=identity[[0, 3]], R_goal=np.diag([4e-4, 1.]),
                  z_past=np.zeros(3), z_now=np.zeros(2), z_goal=np.zeros(2))
    template.blobs = {name: BlobRef(name, f"blobs/{name}.npy") for name in arrays}
    function = FunctionSpec("pcgym.cstr.rk4", "1.0.0", method=method, configuration=CONFIGURATION)

    def variable(name, dimension=4):
        template.variables[name] = VariableSpec(name, dim=dimension)
        return name

    template.factors["prior"] = FactorSpec("prior", "gaussian", {"out": variable("s0")},
                                           param_blobs={"mean": "m0", "cov": "P0"})
    for index in range(PAST + HORIZON + 1):
        kind = "past" if index < PAST else "now" if index == PAST else "goal"
        emission, source = f"s{index}", None
        if index < PAST + HORIZON:
            source, emission = variable(f"s{index}c"), variable(f"s{index}o")
            template.factors[f"eq{index}"] = FactorSpec(
                f"eq{index}", "equality", {"a": f"s{index}", "b": source, "c": emission})
        port = variable(f"e{index}", len(arrays[f"G_{kind}"]))
        template.factors[f"em{index}"] = FactorSpec(
            f"em{index}", "gaussian_linear", {"in": emission, "out": port},
            param_blobs={"matrix": f"G_{kind}", "offset": f"z_{kind}", "cov": f"R_{kind}"})
        template.ports[port] = PortSpec(port, port, PortRole.DATA)
        if source is not None:
            name = f"tr{index + 1}"
            template.factors[name] = FactorSpec(
                name, "gaussian_nonlinear", {"in": source, "out": variable(f"s{index + 1}")},
                param_blobs={"parameters": "theta", "cov": "Q" if index < PAST else "Qp"}, function=function)
            template.constraints.append(moment_matching(name, "in", MomentProjection.GAUSSIAN))
    return template, arrays


def make_engine(method="linearization"):
    template, arrays = model(method)
    return Engine(compile_template(template, arrays=arrays)), template, arrays


def observations(change=0, *, history=False, goal=0.):
    """Source-normalized initial window; changes/history are labeled synthetic."""
    lower, upper = np.array(CONFIGURATION["x_lo"]), np.array(CONFIGURATION["x_hi"])
    current = 2 * (np.array([.8, 330.]) - lower) / (upper - lower) - 1
    current += change * np.array([.01, .005])
    data = {f"e{i}": (np.r_[current + (PAST - i) * np.array([-.01, .01]), -.5]
                       if history else None) for i in range(PAST)}
    data[f"e{PAST}"] = current
    data.update({f"e{i}": np.array([goal, 0.]) for i in range(PAST + 1, PAST + HORIZON + 1)})
    return data


class Binder:
    """The validation's bulk Gaussian binding; inputs are checked before timing."""

    def __init__(self, engine):
        self.rows = {}
        for name, (row, dimension) in engine.arena.gauss_data.items():
            self.rows[name] = row, np.asarray(engine.arena.g_obs_indices(row, dimension))

    def bind(self, base, data):
        values, masks = np.array(base.g_obs), np.array(base.g_mask)
        for name, value in data.items():
            row, indices = self.rows[name]
            masks[row] = value is not None
            if value is not None:
                values[indices] = value
        return _replace(base, g_obs=jnp.asarray(values), g_mask=jnp.asarray(masks))
