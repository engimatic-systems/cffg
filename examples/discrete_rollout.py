"""Run one generic fixed-A discrete GFE planning slice.

The action posterior uses the registered one-step P-substituted objective; it
is not a claim that arbitrary loopy graphs or other constraint combinations
have exact global inference.
"""

import numpy as np

from cffg import Engine, compile_template
from cffg.models import build_rollout


def main() -> np.ndarray:
    template, arrays = build_rollout(
        "two-action-rollout",
        steps=1,
        d=np.array([0.5, 0.5]),
        A=np.eye(2),
        c=np.array([0.9, 0.1]),
        e=np.array([0.5, 0.5]),
        Bs=np.array(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ]
        ),
    )
    engine = Engine(compile_template(template, arrays=arrays))
    result = engine.run(engine.init_state(), 100, tol=1e-6, harden_last=0)
    state = result.require_converged()
    action_posterior = engine.marginals(state)["u1"]

    np.testing.assert_allclose(action_posterior, [0.75, 0.25], atol=2e-6)
    assert np.isfinite(engine.bfe(state))
    print("discrete rollout: action posterior =", action_posterior.tolist())
    return action_posterior


if __name__ == "__main__":
    main()
