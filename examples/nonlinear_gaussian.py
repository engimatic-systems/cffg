"""Run sigma-point inference for a nonlinear Gaussian sensor.

The reported objective freezes the local Gaussian approximation at the fixed
point. It is approximate and is not an evidence bound.
"""

import numpy as np

from cffg import Engine, FunctionSpec, compile_template
from cffg.core import (
    BlobRef,
    FactorSpec,
    MomentProjection,
    Template,
    VariableSpec,
    moment_matching,
)


def main() -> tuple[np.ndarray, np.ndarray]:
    arrays = {
        "prior_mean": np.array([0.25]),
        "prior_cov": np.array([[0.5]]),
        "evidence_mean": np.array([0.8]),
        "evidence_cov": np.array([[0.09]]),
        "noise": np.array([[0.16]]),
        "parameters": np.array([0.3]),
    }
    template = Template(
        name="nonlinear-sensor",
        variables={name: VariableSpec(name, dim=1) for name in ("x", "y")},
        factors={
            "prior": FactorSpec(
                "prior",
                "gaussian",
                edges={"out": "x"},
                param_blobs={"mean": "prior_mean", "cov": "prior_cov"},
            ),
            "sensor": FactorSpec(
                "sensor",
                "gaussian_nonlinear",
                edges={"in": "x", "out": "y"},
                param_blobs={"parameters": "parameters", "cov": "noise"},
                function=FunctionSpec(
                    "cffg.mean.cubic",
                    "1.0.0",
                    method="sigma_point",
                ),
            ),
            "evidence": FactorSpec(
                "evidence",
                "gaussian",
                edges={"out": "y"},
                param_blobs={"mean": "evidence_mean", "cov": "evidence_cov"},
            ),
        },
        blobs={name: BlobRef(name, f"{name}.npy") for name in arrays},
        constraints=[
            moment_matching("sensor", "in", MomentProjection.GAUSSIAN)
        ],
    )
    engine = Engine(compile_template(template, arrays=arrays))
    result = engine.run(engine.init_state(), 30, tol=1e-6)
    state = result.require_converged()
    mean, covariance = engine.gaussian_marginals(state)["x"]
    objective = engine.objective_metadata["sensor"]

    np.testing.assert_allclose(mean, [0.5103029], atol=2e-6)
    np.testing.assert_allclose(covariance, [[0.1372675]], atol=2e-6)
    assert objective["approximate"] and objective["evidence_bound"] is None
    print(
        "nonlinear Gaussian: x mean =",
        mean.tolist(),
        "covariance =",
        covariance.tolist(),
        "objective =",
        objective["objective"],
    )
    return mean, covariance


if __name__ == "__main__":
    main()
