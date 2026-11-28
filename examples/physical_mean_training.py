"""Fit one cooling rate shared by repeated temperature measurements."""

import jax.numpy as jnp
import numpy as np

from cffg import FunctionSpec, MeanImplementation, MeanTrainer, TrainingConfig, register_mean_function
from cffg.core import (
    BlobRef, FactorSpec, MomentProjection, PlateSpec, PortRole, PortSpec, Template,
    VariableSpec, moment_matching,
)


def parameter_shape(input_dim, output_dim, config):
    if input_dim != 1 or output_dim != 1 or set(config) != {"seconds"}:
        raise ValueError("cooling requires scalar temperatures and an elapsed time")
    if not np.isfinite(config["seconds"]) or config["seconds"] < 0:
        raise ValueError("elapsed time must be finite and nonnegative")
    return (1,)


def cooling(excess_temperature, log_rate, config):
    """Newton cooling of temperature above ambient, with rate in inverse seconds."""
    return excess_temperature * jnp.exp(-jnp.exp(log_rate[0]) * config["seconds"])


implementation = MeanImplementation(
    "example.newton_cooling", "1.0.0", parameter_shape, cooling, point_trainable=True,
)


def problem(measurements=(6.0, 6.2, 6.1), *, seconds=1.0):
    arrays = {
        "initial_mean": np.array([10.0]),
        "initial_variance": np.array([[0.04]]),
        "sensor_variance": np.array([[0.25]]),
        "log_rate": np.array([np.log(0.25)]),
    }
    function = FunctionSpec(implementation.impl_id, implementation.version,
                            configuration={"seconds": seconds})
    trial = Template(
        name="cooling-trial",
        variables={v: VariableSpec(v, dim=1) for v in ("initial", "reading")},
        factors={
            "start": FactorSpec("start", "gaussian", {"out": "initial"},
                                param_blobs={"mean": "initial_mean", "cov": "initial_variance"}),
            "cool": FactorSpec("cool", "gaussian_nonlinear", {"in": "initial", "out": "reading"},
                               param_blobs={"parameters": "log_rate", "cov": "sensor_variance"},
                               function=function),
        },
        constraints=[moment_matching("cool", "in", MomentProjection.GAUSSIAN)],
        ports={"reading": PortSpec("reading", "reading", PortRole.DATA)},
        blobs={name: BlobRef(name, name + ".npy") for name in arrays},
    )
    template = Template(name="cooling", plates={"trial": PlateSpec(trial, (len(measurements),))})
    observations = {f"trial/{i}/reading": np.array([value]) for i, value in enumerate(measurements)}
    return template, arrays, observations


def main():
    register_mean_function(implementation)
    trainer = MeanTrainer(*problem(), config=TrainingConfig(learning_rate=0.1))
    state = trainer.initial_state()
    for _ in range(5):
        step = trainer.step(state)
        state = step.require_applied()  # Refuses failed inference or an uncertified derivative.
        print(f"step {state.step}   objective before update {step.objective:.5f}")
    result = trainer.infer(state)
    result.require_converged()
    rate = np.exp(trainer.parameters.values(state.weights)["log_rate"][0])
    print(f"fitted cooling rate {rate:.5f} / s   final objective {result.free_energies[-1]:.5f}")


if __name__ == "__main__":
    main()
