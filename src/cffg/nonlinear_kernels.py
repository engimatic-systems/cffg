"""Compiler-side specialization of portable nonlinear Gaussian means."""

import numpy as np

from cffg.core.ir import FACTOR_TYPES
from cffg._immutable import FrozenMap
from cffg.gaussian_kernels import validate_observation
from cffg.kernel_bindings import GaussianBinding, GaussianConditioning, ParameterSpec
from cffg.mean_functions import MeanShapeError, resolve_mean_function


def _unconfigured(*args, **kwargs):
    raise ValueError("nonlinear Gaussian binding requires a resolved mean function")


def _pack(template, members, resolved):
    weights, precisions = [], []
    for name in members:
        factor = template.factors[name]
        theta = resolved[factor.param_blobs["parameters"]]
        if not np.isfinite(theta).all():
            raise ValueError(f"factor {name!r}: non-finite mean parameters")
        weights.append(theta)
        precisions.append(np.linalg.inv(resolved[factor.param_blobs["cov"]]))
    return {"theta": np.stack(weights), "W": np.stack(precisions)}


def _parameters(shape):
    return {
        "parameters": ParameterSpec(lambda sizes: shape, "theta"),
        "cov": ParameterSpec(lambda s: (s["out"], s["out"]), "W", "covariance_to_precision"),
    }


def configure(factor, sizes):
    import jax

    spec = factor.function
    implementation = resolve_mean_function(spec)
    config = FrozenMap(spec.configuration)
    try:
        shape = tuple(implementation.parameter_shape(sizes["in"], sizes["out"], config))
        if len(shape) != 1 or shape[0] < 0 or not isinstance(shape[0], int):
            raise MeanShapeError("mean parameters require a flat nonnegative-size vector")
        evaluate = lambda x, parameters: implementation.evaluate(x, parameters, config)
        output = jax.eval_shape(
            evaluate, jax.ShapeDtypeStruct((sizes["in"],), np.float64),
            jax.ShapeDtypeStruct(shape, np.float64),
        )
        if output.shape != (sizes["out"],):
            raise MeanShapeError(f"mean output shape {output.shape} != {(sizes['out'],)}")
    except (ValueError, TypeError, AttributeError, IndexError) as error:
        raise MeanShapeError(f"factor {factor.name!r}: mean shape/configuration error: {error}") from error

    def results(eta, precision, params, dims, observed_role, observation, mask):
        import jax.numpy as jnp
        from cffg.nonlinear import local_result

        do, di = dims
        if observation is None:
            observation = jnp.zeros((eta.shape[0], do), eta.dtype)
            mask = jnp.zeros(eta.shape[0], eta.dtype)

        def one(e, p, theta, w, y, present):
            noise = jnp.linalg.solve(w, jnp.eye(do, dtype=w.dtype))
            return local_result(
                evaluate, e[1, :di], p[1, :di, :di], e[0, :do], p[0, :do, :do],
                theta, noise, spec.method,
                y if observed_role is not None else None,
                present if observed_role is not None else None,
            )

        return jax.vmap(one)(eta, precision, params["theta"], params["W"], observation, mask)

    def message(*args):
        import jax.numpy as jnp
        from cffg.execution.contracts import ExecutionDelta
        from cffg.failure_facts import FailureReason
        from cffg.nonlinear import NonlinearFailure

        value = results(*args)
        status = value["status"]
        reasons = jnp.zeros_like(value["failure_code"])
        for local, shared in (
            (NonlinearFailure.IMPROPER_CAVITY, FailureReason.IMPROPER_CAVITY),
            (NonlinearFailure.INVALID_NOISE, FailureReason.INVALID_NOISE),
            (NonlinearFailure.NONFINITE_FUNCTION, FailureReason.NONFINITE_FUNCTION),
            (NonlinearFailure.IMPROPER_TILT, FailureReason.IMPROPER_TILT),
            (NonlinearFailure.UNREPRESENTABLE_SIGMA_POINTS,
             FailureReason.UNREPRESENTABLE_SIGMA_POINTS),
        ):
            reasons = jnp.where(value["failure_code"] == local, shared, reasons)
        return ExecutionDelta(
            ((value["eta_out"], value["precision_out"]),
             (value["eta_in"], value["precision_in"])),
            jnp.where(status == 0, 0.0, jnp.inf), status, reasons,
        )

    def energy(*args):
        return results(*args)["energy"]

    work = {
        "mean_value_evaluations": 1 if spec.method == "linearization" else 2 * sizes["in"] + 1,
        "jacobian_forward_directions": sizes["in"] if spec.method == "linearization" else 0,
    }
    if implementation.work is not None:
        work.update(implementation.work(sizes["in"], sizes["out"], config))

    return GaussianBinding(
        ("out", "in"), _parameters(shape), ("theta", "W"), _pack, message, energy,
        conditioning=GaussianConditioning("out", validate_observation),
        statistics=results, readout=results,
        moment_projections=frozenset({"gaussian_first_second"}), iterative=True,
        objective=f"{spec.method}_frozen_gaussian_bethe",
        work=work,
    )


def install(register, kernel, hostile):
    register(kernel(
        FACTOR_TYPES["gaussian_nonlinear"], "cffg.gaussian.nonlinear", "1.0.0",
        hostile=hostile,
        binding=GaussianBinding(
            ("out", "in"), _parameters((0,)), ("theta", "W"), _pack,
            _unconfigured, _unconfigured,
            conditioning=GaussianConditioning("out", validate_observation),
            moment_projections=frozenset({"gaussian_first_second"}),
            configure=configure, iterative=True, objective="configured_gaussian_surrogate",
        ),
    ))
