"""Infer an exact Gamma posterior from Poisson observations.

The equality chain represents one shared rate without violating the
Forney-style degree limit. Conjugate belief propagation on this tree gives an
exact posterior and exact negative log evidence.
"""

import numpy as np

from cffg import Engine, compile_template
from cffg.core import (
    BlobRef,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)


def gamma_poisson() -> tuple[Template, dict[str, np.ndarray]]:
    template = Template(name="gamma-poisson")
    template.variables["rate0"] = VariableSpec("rate0", ef="gamma")
    template.factors["prior"] = FactorSpec(
        "prior",
        "gamma",
        edges={"out": "rate0"},
        param_blobs={"shape": "shape", "rate": "rate"},
    )
    template.blobs = {
        "shape": BlobRef("shape", "shape.npy"),
        "rate": BlobRef("rate", "rate.npy"),
    }

    tail = "rate0"
    for index in range(3):
        if index < 2:
            branch = f"rate_branch{index}"
            next_tail = f"rate{index + 1}"
            template.variables[branch] = VariableSpec(branch, ef="gamma")
            template.variables[next_tail] = VariableSpec(next_tail, ef="gamma")
            template.factors[f"copy{index}"] = FactorSpec(
                f"copy{index}",
                "equality",
                edges={"a": tail, "b": branch, "c": next_tail},
            )
            observation_rate = branch
            tail = next_tail
        else:
            observation_rate = tail

        count = f"count{index}"
        template.variables[count] = VariableSpec(count, ef="poisson")
        template.factors[f"observation{index}"] = FactorSpec(
            f"observation{index}",
            "poisson_obs",
            edges={"out": count, "rate": observation_rate},
        )
        template.ports[count] = PortSpec(count, count, PortRole.DATA)

    return template, {"shape": np.array(2.0), "rate": np.array(1.5)}


def main() -> tuple[float, float]:
    template, arrays = gamma_poisson()
    engine = Engine(compile_template(template, arrays=arrays))
    initial = engine.bind_data(
        engine.init_state(),
        {"count0": 4, "count1": 0, "count2": 7},
    )
    result = engine.run(initial, 10, tol=1e-6)
    state = result.require_converged()
    posterior = engine.ef_marginals(state)["rate0"]

    assert posterior.proper
    np.testing.assert_allclose(posterior.params, (13.0, 4.5), rtol=2e-6)
    assert np.isfinite(engine.bfe(state))
    print(
        "exponential family: Gamma(shape, rate) =",
        posterior.params,
        "mean =",
        posterior.mean,
    )
    return posterior.params


if __name__ == "__main__":
    main()
