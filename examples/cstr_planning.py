"""Plan through registered nonlinear reactor physics, without caller linearization.

Run with ``uv run python examples/cstr_planning.py``. This is an explicitly
declared short-horizon reconstruction, not the missing failed PCGym invocation.
The Gaussian objective is approximate; convergence is not exact evidence or a
closed-loop performance guarantee. Narrow initial uncertainty is deliberate.

Coordinates are Ca=.85+.15*s[0], T=325+25*s[1] K, normalized concentration
bias s[2], and coolant Tc=298.5+3.5*s[3] K. Concentration and dt=26/60 use
the retained model's raw units. Evaluation clips only initial physical inputs,
not RK4 stages/final states or the action belief. There are no hard actuators.

The source of truth is the returned Template and arrays. A receiving process
must call install_mean() before load_bundle(), which validates and compiles;
Python callbacks and bound observations are not included in the model bundle.
"""

import argparse
import json
from dataclasses import asdict
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from cffg import (
    Engine, FunctionSpec, MeanImplementation, compile_template,
    register_mean_function, save_bundle,
)
from cffg.core import (
    BlobRef, FactorSpec, MomentProjection, PortRole, PortSpec, Template,
    VariableSpec, moment_matching,
)
FUNCTION_NAME = "cffg.example.cstr_rk4"
CONFIGURATION = {
    "x_lo": [.7, 300.], "x_hi": [1., 350.],
    "u_lo": 295., "u_hi": 302., "dt": 26 / 60, "substeps": 8,
}
NOMINAL_PARAMETERS = np.array([
    np.log(7.2e10) - 8750 / 330, .875, np.log(5e4), np.log(5e4),
])


def cstr_rk4(state, parameters, configuration):
    """Normalized [concentration, temperature, disturbance, action] transition.

    Only the initial physical state/action are clipped, as in the experiment.
    RK4 intermediate and final states are intentionally not clipped.
    """
    lower, upper = jnp.asarray(configuration["x_lo"]), jnp.asarray(configuration["x_hi"])
    span = upper - lower
    physical = jnp.clip((state[:2] + 1) * span / 2 + lower, lower - 2 * span, upper + 2 * span)
    physical = physical.at[0].set(jnp.maximum(physical[0], 0.))
    low_u, high_u = configuration["u_lo"], configuration["u_hi"]
    action = jnp.clip((state[3] + 1) * (high_u - low_u) / 2 + low_u, low_u - 20, high_u + 20)
    rate, activation, transfer, enthalpy = (
        jnp.exp(parameters[0]), parameters[1] * 1e4,
        jnp.exp(parameters[2]), jnp.exp(parameters[3]),
    )

    def derivative(value):
        concentration, temperature = value
        reaction = rate * jnp.exp(-activation * (1 / temperature - 1 / 330)) * concentration
        return jnp.stack([
            1 - concentration - reaction,
            350 - temperature + enthalpy * reaction / 239 + transfer * (action - temperature) / 23900,
        ])

    step = configuration["dt"] / configuration["substeps"]
    for _ in range(configuration["substeps"]):
        k1 = derivative(physical)
        k2 = derivative(physical + step * k1 / 2)
        k3 = derivative(physical + step * k2 / 2)
        k4 = derivative(physical + step * k3)
        physical = physical + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    normalized = 2 * (physical - lower) / span - 1
    normalized = normalized.at[0].add(state[2])
    return jnp.concatenate((normalized, state[2:]))


def _parameter_shape(input_dimension, output_dimension, configuration):
    if input_dimension != 4 or output_dimension != 4:
        raise ValueError("the CSTR model requires four-dimensional states")
    return (4,)


@cache
def install_mean():
    """Install the trusted callback needed to reconstruct this model's FunctionSpec."""
    register_mean_function(MeanImplementation(FUNCTION_NAME, "1.0.0", _parameter_shape, cstr_rk4))


def planning_model(method="linearization", *, past=0, horizon=2,
                   process_scale=1., temperature_variance=None, dtype=np.float64,
                   prior_diagonal=(.01, .001, 1e-4, .1), affine=False):
    """Return portable model sources and DATA bindings, not a compiled arena.

    Defaults declare the tested short-horizon candidate. Other horizons, priors
    and precision ratios are experiments, not promised convergence. ``affine``
    selects a separate linear limit for independent Gaussian-oracle checks.
    """
    install_mean()
    function = FunctionSpec(FUNCTION_NAME, "1.0.0", method=method, configuration=CONFIGURATION)
    parameters = NOMINAL_PARAMETERS.copy()
    if affine:
        matrix = np.array([[.9, .02, 1., .01], [0., .85, 0., .08],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]])
        parameters = np.r_[matrix.ravel(), np.zeros(4)]
        function = FunctionSpec("cffg.mean.affine", "1.0.0", method=method)
    template = Template(name=f"reconstructed_cstr_p{past}_h{horizon}")
    q = np.diag([1e-4, 4e-4, 4e-6, .25])
    arrays = {
        "theta": parameters, "Q": q, "Qp": q * process_scale,
        "m0": np.zeros(4), "P0": np.diag(prior_diagonal),
        "G_past": np.eye(4)[[0, 1, 3]], "R_past": np.diag([1e-4, 1e-4, 1e-6]),
        "G_now": np.eye(4)[[0, 1]], "R_now": np.diag([1e-4, 1e-4]),
        "G_goal": np.eye(4)[[0, 3]], "R_goal": np.diag([4e-4, 1.]),
    }
    if temperature_variance is not None:
        arrays.update(G_goal=np.eye(4)[[0, 1, 3]], R_goal=np.diag([4e-4, temperature_variance, 1.]))
    for kind in ("past", "now", "goal"):
        arrays[f"offset_{kind}"] = np.zeros(len(arrays[f"G_{kind}"]))
    arrays = {name: np.asarray(value, dtype=dtype) for name, value in arrays.items()}
    template.blobs = {name: BlobRef(name, f"{name}.npy") for name in arrays}

    def variable(name, dimension=4):
        template.variables[name] = VariableSpec(name, dim=dimension)
        return name

    template.factors["prior"] = FactorSpec("prior", "gaussian", {"out": variable("s0")},
                                            param_blobs={"mean": "m0", "cov": "P0"})
    observations = {}
    length = past + horizon + 1
    for index in range(length):
        kind = "past" if index < past else "now" if index == past else "goal"
        emission = source = f"s{index}"
        if index < length - 1:
            emission, source = variable(f"s{index}o"), variable(f"s{index}c")
            template.factors[f"eq{index}"] = FactorSpec(
                f"eq{index}", "equality", {"a": f"s{index}", "b": source, "c": emission},
            )
            name = f"tr{index + 1}"
            template.factors[name] = FactorSpec(
                name, "gaussian_nonlinear", {"in": source, "out": variable(f"s{index + 1}")},
                param_blobs={"parameters": "theta", "cov": "Q" if index < past else "Qp"},
                function=function,
            )
            template.constraints.append(moment_matching(name, "in", MomentProjection.GAUSSIAN))
        port = variable(f"e{index}", len(arrays[f"G_{kind}"]))
        template.factors[f"em{index}"] = FactorSpec(
            f"em{index}", "gaussian_linear", {"in": emission, "out": port},
            param_blobs={"matrix": f"G_{kind}", "offset": f"offset_{kind}", "cov": f"R_{kind}"},
        )
        template.ports[port] = PortSpec(port, port, PortRole.DATA)
        observations[port] = (None if kind == "past" else np.array([-1 / 3, .2], dtype=dtype)
                              if kind == "now" else np.zeros(len(arrays["G_goal"]), dtype=dtype))
    return template, arrays, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("linearization", "sigma_point"), default="linearization")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--schedule", choices=("auto", "flood"), default="auto")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--process-scale", type=float, default=1.)
    parser.add_argument("--temperature-variance", type=float)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--save", help="Save the source model as a .cffgz bundle; DATA is supplied separately")
    args = parser.parse_args()
    dtype = np.dtype(args.dtype).type
    with jax.enable_x64(dtype is np.float64):
        template, arrays, data = planning_model(
            args.method, horizon=args.horizon, dtype=dtype,
            process_scale=args.process_scale, temperature_variance=args.temperature_variance,
        )
        if args.save:
            save_bundle(args.save, template=template, arrays=arrays)
        engine = Engine(compile_template(template, arrays=arrays), schedule=args.schedule)
        result = engine.run(engine.bind_data(engine.init_state(), data), args.budget, tol=1e-6)
        report = {
            "status": result.status.value, "iterations": result.iterations,
            "adaptive_residual": result.outer_residual,
            "energy": result.free_energies[-1] if result.free_energies else None,
            "failure": asdict(result.failure) if result.failure else None,
            "objective": engine.objective_contract,
        }
        print(json.dumps(report, indent=2))
        state = result.require_converged()
        mean, covariance = engine.gaussian_marginals(state)["s1"]
        print("first planned coolant mean (K):", 298.5 + 3.5 * float(mean[3]))
        print("coolant variance (K squared):", 3.5**2 * float(covariance[3, 3]))
        return result


if __name__ == "__main__":
    main()
