"""Infer exact marginals and a MAP assignment under a hard constraint.

The graph is a tree with one hard-constraint factor, so sum-product returns
exact normalized marginals and Bethe free energy. Max-product returns the exact
minimum-energy assignment for this bounded CSP.
"""

import numpy as np

from cffg import Engine, compile_template
from cffg.core import BlobRef, FactorSpec, Template, VariableSpec


def main() -> tuple[int, ...]:
    priors = (
        np.array([0.80, 0.15, 0.05]),
        np.array([0.20, 0.70, 0.10]),
        np.array([0.10, 0.20, 0.70]),
    )
    template = Template(
        name="three-way-assignment",
        variables={f"x{i}": VariableSpec(f"x{i}", 3) for i in range(3)},
        factors={
            **{
                f"prior{i}": FactorSpec(
                    f"prior{i}",
                    "categorical",
                    edges={"out": f"x{i}"},
                    param_blobs={"statistic": f"p{i}"},
                )
                for i in range(3)
            },
            "distinct": FactorSpec(
                "distinct",
                "alldifferent",
                edges={f"x{i}": f"x{i}" for i in range(3)},
            ),
        },
        blobs={f"p{i}": BlobRef(f"p{i}", f"p{i}.npy") for i in range(3)},
    )
    arrays = {f"p{i}": prior for i, prior in enumerate(priors)}

    sum_engine = Engine(compile_template(template, arrays=arrays))
    sum_result = sum_engine.run(sum_engine.init_state(), 20, tol=1e-6)
    sum_state = sum_result.require_converged()
    marginals = sum_engine.marginals(sum_state)
    marginal_modes = tuple(int(np.argmax(marginals[f"x{i}"])) for i in range(3))

    assert len(set(marginal_modes)) == 3
    assert np.isfinite(sum_engine.bfe(sum_state))

    max_engine = Engine(compile_template(template, arrays=arrays, semiring="max"))
    max_result = max_engine.run(
        max_engine.init_state(),
        20,
        tol=1e-6,
        harden_last=0,
    )
    max_state = max_result.require_converged()
    assignment = tuple(max_engine.decode(max_state)[f"x{i}"] for i in range(3))

    assert assignment == (0, 1, 2)
    assert np.isfinite(max_engine.objective(max_state))
    print("hard constraint: marginal modes =", marginal_modes, "MAP =", assignment)
    return assignment


if __name__ == "__main__":
    main()
