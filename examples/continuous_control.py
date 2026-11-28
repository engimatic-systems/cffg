"""Plan with an augmented Gaussian and learn tied dynamics from executed transitions.

Run with uv run python examples/continuous_control.py. Set JAX_ENABLE_X64=1
to use float64. The planner binds a future target to a noisy goal emission.
The learner uses a separate graph with exact taps on measured regressors.
"""

import jax
import numpy as np

from cffg import Engine, compile_template, conj
from cffg.core import (
    BlobRef, FactorSpec, LearnableSpec, PortRole, PortSpec, Template,
    VariableSpec, observe_interior,
)
from cffg.learn import Learnables


def add_transition(model, name):
    """An affine conditional from [state, command] to the next scalar state."""
    regressor, successor = f"{name}_regressor", f"{name}_next"
    model.variables[regressor] = VariableSpec(regressor, dim=2)
    model.variables[successor] = VariableSpec(successor, dim=1)
    model.factors[f"{name}_prior"] = FactorSpec(
        f"{name}_prior", "gaussian", {"out": regressor},
        param_blobs={"mean": "prior_mean", "cov": "prior_cov"},
    )
    model.factors[name] = FactorSpec(
        name, "gaussian_linear", {"in": regressor, "out": successor},
        param_blobs={"matrix": "A", "offset": "b", "cov": "Q"},
    )
    return regressor, successor


def window(name, arrays):
    return Template(name, blobs={key: BlobRef(key, f"{key}.npy") for key in arrays})


def compile_window(model, arrays, dtype):
    return Engine(compile_template(model, arrays={
        name: np.asarray(value, dtype=dtype) for name, value in arrays.items()
    }))


def propose(current, target, dynamics, dtype):
    arrays = {
        **dynamics, "prior_mean": [current, 0.], "prior_cov": [[.04, 0.], [0., .5]],
        "H": [[1.]], "sensor_offset": [0.], "R": [[.12]],
    }
    model = window("one_step_gaussian_plan", arrays)
    regressor, successor = add_transition(model, "motion")
    model.variables["desired_reading"] = VariableSpec("desired_reading", dim=1)
    model.factors["goal_emission"] = FactorSpec(
        "goal_emission", "gaussian_linear", {"in": successor, "out": "desired_reading"},
        param_blobs={"matrix": "H", "offset": "sensor_offset", "cov": "R"},
    )
    model.ports["goal"] = PortSpec("goal", "desired_reading", PortRole.DATA)
    engine = compile_window(model, arrays, dtype)
    trial = engine.bind_data(engine.init_state(), {"goal": [target]})
    state = engine.run(trial, 24, tol=1e-6).require_converged()
    mean, covariance = engine.gaussian_marginal(state, regressor)
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ArithmeticError("the plan has no proper command marginal")
    return float(mean[1])


def learn_executed(records, dynamics, dtype):
    # Every source uses the same three model blobs and one posterior.
    arrays = {**dynamics, "prior_mean": [0., 0.], "prior_cov": np.eye(2)}
    model = window("executed_gaussian_transitions", arrays)
    data = {}
    sources = tuple(f"transition_{i}" for i in range(len(records)))
    for name, (measured, applied, next_measured) in zip(sources, records, strict=True):
        regressor, successor = add_transition(model, name)
        output_port = f"{name}_reading"
        model.ports[output_port] = PortSpec(output_port, successor, PortRole.DATA)
        model, input_port = observe_interior(model, regressor)
        data[input_port] = [measured, applied]
        data[output_port] = [next_measured]

    model.learnables["dynamics"] = LearnableSpec(
        "dynamics", "mniw", ("A", "b", "Q"), sources,
    )
    engine = compile_window(model, arrays, dtype)
    # The tap substitutes the full known input. The transition's exterior
    # DATA output conditions its likelihood on the measured successor.
    observed = engine.bind_data(engine.init_state(), data)
    result = engine.run(observed, 24, tol=1e-6)
    state = result.require_converged()
    prior = conj.vague_mniw(1, 2, scale=.2)
    learner = Learnables(engine, priors={"dynamics": prior})
    learner.accumulate(engine, result)
    assert learner.params["dynamics"]["nu"] == prior.nu + len(records)

    stats = engine.gaussian_node_stats(state, centered=True)
    for name, (measured, applied, next_measured) in zip(sources, records, strict=True):
        np.testing.assert_array_equal(
            stats[name]["mean"], np.asarray([next_measured, measured, applied], dtype),
        )
        np.testing.assert_array_equal(stats[name]["cov_xx"], np.zeros((2, 2)))

    design = np.array([[measured, applied, 1.] for measured, applied, _ in records])
    responses = np.array([[next_measured] for _, _, next_measured in records])
    coefficients = np.linalg.solve(
        prior.lam + design.T @ design, prior.lam @ prior.m.T + design.T @ responses,
    ).T
    np.testing.assert_allclose(learner.expected("dynamics", "matrix"), coefficients[:, :2], atol=2e-6)
    np.testing.assert_allclose(learner.expected("dynamics", "offset"), coefficients[:, 2], atol=2e-6)
    return learner


def main():
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    dynamics = {"A": [[.7, .8]], "b": [0.], "Q": [[.08]]}
    current = .2
    records = []
    proposals = []
    for target in (1., -.7):
        proposal = propose(current, target, dynamics, dtype)
        applied = float(np.clip(proposal, -.35, .35))
        next_measured = .75 * current + 1.15 * applied + .08  # Tiny stand-in plant.
        records.append((current, applied, next_measured))
        proposals.append(proposal)
        current = next_measured

    assert proposals[0] > .35 and proposals[1] < -.35
    learner = learn_executed(records, dynamics, dtype)
    learned = {
        blob: learner.rollover("dynamics", role)
        for blob, role in (("A", "matrix"), ("b", "offset"), ("Q", "cov"))
    }
    next_proposal = propose(current, 1., learned, dtype)
    print("continuous control: proposed =", proposals, "applied =", [r[1] for r in records])
    print("learned matrix =", learner.expected("dynamics", "matrix").tolist())
    print("posterior mean covariance =", learner.expected("dynamics", "cov").tolist(),
          "next-solve covariance =", learned["Q"].tolist(), "next proposal =", next_proposal)


if __name__ == "__main__":
    main()
