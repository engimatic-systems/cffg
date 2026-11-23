"""Build, run, bundle, and checkpoint the supported discrete agent workflow.

The standalone graph is a tree, so its Bethe free energy is exact for the
bound observation. The rolling agent uses the registered fixed-A discrete
planning objective.
"""

from functools import partial
from pathlib import Path

import numpy as np

from cffg import (
    CorridorEnv,
    Engine,
    RollingAgent,
    build_gridworld,
    compile_template,
    load_bundle,
    save_bundle,
)


def main() -> np.dtype:
    output_dir = Path.cwd()

    template, arrays = build_gridworld(past=1, steps=1, k=2, acc=1.0)
    model_path = output_dir / "corridor.cffgz"
    save_bundle(model_path, template, arrays)
    bundle = load_bundle(model_path)

    engine = Engine(compile_template(bundle.template, arrays=bundle.arrays))
    initial = engine.bind_data(
        engine.init_state(),
        {"act1": 1, "obs1": 0},
    )
    result = engine.run(initial, 20, tol=1e-6)
    state = result.require_converged()
    posterior = engine.marginals(state)["zm1"]
    free_energy = engine.bfe(state)
    np.testing.assert_allclose(posterior, [1.0, 0.0], atol=2e-6)
    np.testing.assert_allclose(
        free_energy,
        -np.log(bundle.arrays["d"][0]),
        rtol=2e-6,
    )

    builder = partial(build_gridworld, k=2, acc=1.0)
    initial_b_counts = np.ones((3, 2, 2))

    def make_agent() -> RollingAgent:
        return RollingAgent(
            builder,
            horizon=1,
            learn_b=True,
            initial_b_counts=initial_b_counts,
            n_sweeps=100,
            select="sample",
            seed=7,
        )

    environment = CorridorEnv(k=2, acc=1.0, seed=11)
    agent = make_agent()
    first_action = agent.step(environment.step)
    checkpoint_path = output_dir / "agent.npz"
    agent.checkpoint(checkpoint_path)

    resumed = make_agent()
    resumed.restore(checkpoint_path)
    np.testing.assert_array_equal(resumed.belief, agent.belief)
    np.testing.assert_array_equal(resumed.b_counts, agent.b_counts)
    assert resumed.tick == agent.tick

    print(
        "discrete agent:",
        f"action={first_action}",
        f"posterior={posterior.tolist()}",
        f"free_energy={float(free_energy):.6f}",
        f"message dtype: {state.messages.dtype}",
    )
    return state.messages.dtype


if __name__ == "__main__":
    main()
