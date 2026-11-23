"""Smooth a small linear Gaussian state-space model.

This factor graph is a tree, so Gaussian belief propagation is exact and its
Bethe free energy is the negative log likelihood of the bound observations.
"""

import numpy as np

from cffg import Engine, compile_template
from cffg.models import lgssm


def main() -> tuple[np.ndarray, np.ndarray]:
    template, arrays = lgssm(
        3,
        f=np.array([[0.8]]),
        q=np.array([[0.1]]),
        h=np.array([[1.0]]),
        r=np.array([[0.2]]),
        m0=np.array([0.0]),
        p0=np.array([[1.0]]),
    )
    engine = Engine(compile_template(template, arrays=arrays))
    initial = engine.bind_data(
        engine.init_state(),
        {"y0": np.array([0.2]), "y1": None, "y2": np.array([0.8])},
    )
    result = engine.run(initial, 20, tol=1e-6)
    state = result.require_converged()
    mean, covariance = engine.gaussian_marginals(state)["x1"]

    np.testing.assert_allclose(mean, [0.3985194], atol=2e-6)
    np.testing.assert_allclose(covariance, [[0.14342995]], atol=2e-6)
    assert np.isfinite(engine.bfe(state))
    print(
        "linear Gaussian: x1 mean =",
        mean.tolist(),
        "covariance =",
        covariance.tolist(),
    )
    return mean, covariance


if __name__ == "__main__":
    main()
