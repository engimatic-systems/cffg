"""Infer continuous states and discrete regimes in a tiny rSLDS.

The mixed Gaussian and categorical messages use moment-matched mixture
collapse and a variational recurrence update. The returned objective is the
declared approximation, not exact model evidence.
"""

import numpy as np

from cffg import Engine, compile_template
from cffg.models import rslds


def main() -> tuple[np.ndarray, np.ndarray]:
    template, arrays = rslds(
        2,
        dynamics_matrix=np.array([[[0.85]], [[-0.45]]]),
        dynamics_offset=np.array([[0.15], [-0.25]]),
        dynamics_covariance=np.array([[[0.18]], [[0.28]]]),
        recurrence_weights=np.array([[[1.1]], [[-0.9]]]),
        recurrence_offsets=np.array([[-0.1], [0.2]]),
        emission_matrix=np.ones((1, 1)),
        emission_covariance=np.array([[0.12]]),
        initial_state_mean=np.array([0.1]),
        initial_state_covariance=np.array([[0.7]]),
        initial_regime_probability=np.array([0.55, 0.45]),
    )
    engine = Engine(compile_template(template, arrays=arrays))
    initial = engine.bind_data(
        engine.init_state(),
        {"y0": None, "y1": np.array([0.35])},
    )
    result = engine.run(initial, 80, tol=1e-7)
    state = result.require_converged()
    regime = engine.marginals(state)["z1"]
    state_mean, state_covariance = engine.gaussian_marginals(state)["x1"]

    np.testing.assert_allclose(regime.sum(), 1.0, atol=2e-6)
    assert np.isfinite(state_mean).all()
    assert np.linalg.eigvalsh(state_covariance).min() > 0
    assert np.isfinite(engine.bfe(state))
    print(
        "hybrid rSLDS: z1 =",
        regime.tolist(),
        "x1 mean =",
        state_mean.tolist(),
    )
    return regime, state_mean


if __name__ == "__main__":
    main()
