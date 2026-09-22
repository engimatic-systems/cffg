"""Continue exact categorical conjugate learning from a portable bundle.

Each trial is an exact one-node BP update. The bundle stores both immutable
Dirichlet priors and current posterior counts so the next trial can continue.
"""

from pathlib import Path

import equinox as eqx
import numpy as np

from cffg import Engine, compile_template, load_bundle, save_bundle
from cffg.core import (
    BlobRef,
    FactorSpec,
    LearnableSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.learn import Learnables


def main() -> np.ndarray:
    template = Template(
        name="learning-sensor",
        variables={"x": VariableSpec("x", cardinality=2)},
        factors={
            "prior": FactorSpec(
                "prior",
                "categorical",
                edges={"out": "x"},
                param_blobs={"statistic": "probabilities"},
            )
        },
        blobs={"probabilities": BlobRef("probabilities", "probabilities.npy")},
        ports={
            "observation": PortSpec("observation", "x", PortRole.DATA)
        },
        learnables={
            "probabilities": LearnableSpec(
                "probabilities",
                "dirichlet",
                ("probabilities",),
            )
        },
    )
    arrays = {"probabilities": np.array([0.5, 0.5])}
    engine = Engine(compile_template(template, arrays=arrays))
    learning = Learnables(engine, priors={"probabilities": np.ones(2)})

    bundle_path = Path.cwd() / "learning-sensor.cffgz"
    save_bundle(bundle_path, template, arrays, learnables=learning)
    bundle = load_bundle(bundle_path)
    assert bundle.resumable
    engine = Engine(compile_template(bundle.template, arrays=bundle.arrays))
    learning = bundle.learnables

    for observation in (1, 0):
        state = engine.init_state()
        params = learning.write(engine, state.params)
        state = eqx.tree_at(lambda current: current.params, state, params)
        state = engine.bind_data(state, {"observation": observation})
        result = engine.run(state, 10)
        result.require_converged()
        learning.accumulate(engine, result)

    save_bundle(
        bundle_path,
        bundle.template,
        bundle.arrays,
        learnables=learning,
    )
    resumed = load_bundle(bundle_path).learnables
    counts = resumed.params["probabilities"]["alpha"]
    np.testing.assert_array_equal(counts, [2.0, 2.0])
    np.testing.assert_allclose(resumed.expected("probabilities"), [0.5, 0.5])
    print("learning bundle: posterior counts =", counts.tolist())
    return counts


if __name__ == "__main__":
    main()
