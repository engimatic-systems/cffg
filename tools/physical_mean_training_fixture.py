"""Deterministic physical-mean training data without a simulator or saved state."""

from functools import cache

import numpy as np

from cffg import FunctionSpec, MeanImplementation, register_mean_function
from cffg.core import (
    BlobRef, FactorSpec, MomentProjection, PortRole, PortSpec, Template,
    VariableSpec, moment_matching,
)
from cstr_runtime_fixture import CONFIGURATION, PARAMETERS, cstr_rk4


TRANSITIONS = 60
RK4_MEAN = "benchmark.physical.cstr_rk4"
COOLING_MEAN = "benchmark.physical.cooling"
COOLING_DT = .6


def parameter_shape(width, count):
    def shape(di, do, _configuration):
        if di != width or do != width:
            raise ValueError(f"expected {width} coordinates in and out")
        return (count,)
    return shape


def cooling_rk4(value, theta, _configuration):
    rate, ambient = theta

    def slope(x):
        return -rate * (x - ambient)

    h = COOLING_DT
    k1 = slope(value)
    k2 = slope(value + h * k1 / 2)
    k3 = slope(value + h * k2 / 2)
    k4 = slope(value + h * k3)
    return value + h * (k1 + 2*k2 + 2*k3 + k4) / 6


@cache
def register_means():
    # Measure the existing point-training path on the pinned baseline.
    for name, dimensions, count, evaluate in (
        (RK4_MEAN, 4, 4, cstr_rk4),
        (COOLING_MEAN, 1, 2, cooling_rk4),
    ):
        def work(di, do, configuration, *, count=count):
            return {"network_parameters": count}

        register_mean_function(MeanImplementation(
            name, "1.0.0", parameter_shape(dimensions, count), evaluate, work,
        ))


def transition_graph(name, function, inputs, outputs, parameters, prior_cov, transition_cov,
                     projections, sensor_cov):
    """The retained training topology shares one parameter blob across all roots."""
    template = Template(name=name)
    dtype = inputs.dtype
    inputs_width, observed_width = inputs.shape[1], outputs.shape[1]
    arrays = {
        "theta": parameters, "Pz": prior_cov, "Q": transition_cov,
        "H": projections, "zero": np.zeros(observed_width, dtype), "R": sensor_cov,
    }
    observations = {}
    for i, (source, reading) in enumerate(zip(inputs, outputs, strict=True)):
        latent, successor, edge = f"a{i}", f"x{i}", f"y{i}"
        prior, transition, emission = f"pz{i}", f"dyn{i}", f"em{i}"
        arrays[f"z{i}"] = source
        for var, width in ((latent, inputs_width), (successor, inputs_width), (edge, observed_width)):
            template.variables[var] = VariableSpec(var, dim=width)
        template.factors[prior] = FactorSpec(
            prior, "gaussian", {"out": latent}, param_blobs={"mean": f"z{i}", "cov": "Pz"},
        )
        template.factors[transition] = FactorSpec(
            transition, "gaussian_nonlinear", {"in": latent, "out": successor},
            param_blobs={"parameters": "theta", "cov": "Q"}, function=function,
        )
        template.constraints.append(moment_matching(transition, "in", MomentProjection.GAUSSIAN))
        template.factors[emission] = FactorSpec(
            emission, "gaussian_linear", {"in": successor, "out": edge},
            param_blobs={"matrix": "H", "offset": "zero", "cov": "R"},
        )
        template.ports[edge] = PortSpec(edge, edge, PortRole.DATA)
        observations[edge] = reading
    template.blobs = {name: BlobRef(name, f"blobs/{name}.npy") for name in arrays}
    return template, arrays, observations


def synthetic_inputs():
    ca, temperature, coolant = np.meshgrid(
        np.linspace(.75, .93, 5), np.linspace(312., 337., 4),
        np.linspace(295.5, 301.5, 3), indexing="ij",
    )
    physical = np.column_stack([a.ravel() for a in (ca, temperature, coolant)])
    return physical


def _host_cstr(physical, theta):
    """Generate reference outputs in physical coordinates using NumPy RK4."""
    state = physical[:, :2].copy()
    coolant = physical[:, 2]
    k_ref, activation, ua, heat = np.exp(theta[0]), theta[1]*1e4, *np.exp(theta[2:])
    h = CONFIGURATION["dt"] / CONFIGURATION["substeps"]

    def rhs(x):
        ca, temp = x.T
        reaction = k_ref * ca * np.exp(-activation * (1 / temp - 1 / 330.))
        balance = 350. - temp + (heat*reaction + ua*(coolant-temp)/100.) / 239.
        return np.column_stack((1. - ca - reaction, balance))

    for _ in range(CONFIGURATION["substeps"]):
        a = rhs(state)
        b = rhs(state + h*a/2)
        c = rhs(state + h*b/2)
        d = rhs(state + h*c)
        state += (h/6)*(a + 2*b + 2*c + d)
    return state


def cstr_training(dtype):
    register_means()
    physical = synthetic_inputs()
    lo = np.array(CONFIGURATION["x_lo"])
    span = np.array(CONFIGURATION["x_hi"]) - lo
    normalized = 2*(physical[:, :2] - lo)/span - 1
    coolant = (2*(physical[:, 2] - CONFIGURATION["u_lo"])
               / (CONFIGURATION["u_hi"] - CONFIGURATION["u_lo"]) - 1)
    phase = np.arange(TRANSITIONS)
    bias = .008*np.sin(.37*phase)
    inputs = np.column_stack((normalized, bias, coolant))
    outputs = 2*(_host_cstr(physical, PARAMETERS) - lo)/span - 1
    outputs[:, 0] += bias
    outputs += .0004*np.column_stack((np.sin(.11*phase), np.cos(.21*phase)))
    theta = PARAMETERS + np.array([.03, -.01, .025, -.02])
    function = FunctionSpec(RK4_MEAN, "1.0.0", method="linearization",
                            configuration=CONFIGURATION)
    return transition_graph(
        "physical-mean-cstr-60", function, inputs.astype(dtype), outputs.astype(dtype),
        theta.astype(dtype), np.eye(4, dtype=dtype)*1e-6,
        np.diag(np.array([1e-5, 1e-5, 4e-6, .25], dtype)),
        np.eye(4, dtype=dtype)[:2], np.eye(2, dtype=dtype)*1e-6,
    )


def scalar_marginal(theta, centers, observations, prior_variance, output_variance):
    """Host exact Gaussian marginal and gradient from the RK4 stability polynomial."""
    rate, ambient = np.asarray(theta, dtype=np.float64)
    h = COOLING_DT
    p = -h*rate
    g = 1 + p + p**2/2 + p**3/6 + p**4/24
    g_rate = -h*(1 + p + p**2/2 + p**3/6)
    residual = observations - (ambient + g*(centers - ambient))
    variance = output_variance + g*g*prior_variance
    mean_jac = np.column_stack((g_rate*(centers - ambient), np.full(len(centers), 1-g)))
    variance_jac = np.array([2*g*g_rate*prior_variance, 0.])
    grad = np.sum(-residual[:, None] * mean_jac/variance
                  + .5*(1/variance - (residual/variance)**2)[:, None]*variance_jac, axis=0)
    nll = .5*np.sum(np.log(2*np.pi*variance) + residual**2/variance)
    return nll, grad, mean_jac


def cooling_training(dtype):
    register_means()
    centers = np.array([-1.2, -.45, .2, .8, 1.4], dtype)
    truth = np.array([.58, .23])
    p = -COOLING_DT*truth[0]
    gain = 1 + p + p**2/2 + p**3/6 + p**4/24
    readings = truth[1] + gain*(centers - truth[1]) + np.array([0., .02, -.015, -.012, .007])
    return transition_graph(
        "physical-mean-cooling-oracle", FunctionSpec(COOLING_MEAN, "1.0.0", method="linearization"),
        centers[:, None], readings.astype(dtype)[:, None], np.array([.4, .08], dtype),
        np.array([[.04]], dtype), np.array([[.018]], dtype), np.ones((1, 1), dtype),
        np.array([[.003]], dtype),
    )
