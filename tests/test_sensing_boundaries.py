"""Public rejection boundaries for predictive categorical information."""

import numpy as np
import pytest

from cffg import Engine, compile_template
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from test_constraint_data import constraint_model
from test_dirichlet_param_edges import _make_template
from test_gaussian_tap_execution import chain
from test_gfe import K, M, make_model
from test_sensing_information import make_engine
from test_vmp import A, D, make_conjugate


@pytest.mark.parametrize(("mean_field", "semiring", "match"),
                         [(False, "max", "sum-product"), (True, "sum", "VMP.*lik")])
def test_factorization_and_semiring(mean_field, semiring, match):
    t = make_conjugate(mean_field)
    engine = Engine(compile_template(t, arrays={"d": D, "A": A}, semiring=semiring))
    with pytest.raises(ValueError, match=match):
        engine.eig(engine.init_state(), "z", ["px"])


def test_constraint_and_goal_composites():
    t, arrays = constraint_model("exactly_k", 1)
    engine = Engine(compile_template(t, arrays=arrays))
    with pytest.raises(ValueError, match="constraint.*limit"):
        engine.eig(engine.init_state(), "a", ["b"])

    probs = np.full((M, K), 1 / M)
    goal, prior = np.full(M, 1 / M), np.full(K, 1 / K)
    t = make_model(probs, goal, prior)
    del t.factors["prior"]
    t.ports["sensor"] = PortSpec("sensor", "z", PortRole.DATA)
    engine = Engine(compile_template(t, arrays={"A": probs, "c": goal, "d": prior}))
    with pytest.raises(ValueError, match="GFE composite.*obs"):
        engine.eig(engine.init_state(), "x", ["sensor"])


def test_parameter_and_continuous_edges():
    t = _make_template("matrix_cat", (2, 2))
    engine = Engine(compile_template(t, arrays={"alpha0": np.ones((2, 2))}))
    with pytest.raises(ValueError, match="theta.*categorical"):
        engine.eig(engine.init_state(), "theta", ["out"])
    with pytest.raises(ValueError, match="VMP.*lik"):
        engine.eig(engine.init_state(), "in", ["out"])

    t, arrays = chain(2)
    t.ports["coin"] = PortSpec("coin", "coin", PortRole.DATA)
    t.variables["coin"] = VariableSpec("coin", 2)
    t.factors["coin_prior"] = FactorSpec("coin_prior", "categorical", {"out": "coin"},
                                          param_blobs={"statistic": "coin_p"})
    t.blobs["coin_p"] = BlobRef("coin_p", "p.npy")
    engine = Engine(compile_template(t, arrays={**arrays, "coin_p": np.array([.5, .5])}))
    with pytest.raises(ValueError, match="x0.*categorical"):
        engine.eig(engine.init_state(), "x0", ["y0"])
    with pytest.raises(ValueError, match="y0.*categorical"):
        engine.eig(engine.init_state(), "coin", ["y0"])


def test_cycle_in_query_component():
    t = Template("cycle", variables={name: VariableSpec(name, 2) for name in ("x", "y", "s")},
                 factors={"mix": FactorSpec("mix", "transition_mixture",
                                            {"in": "y", "out": "x", "selector": "s"},
                                            param_blobs={"matrices": "mix"}),
                          "back": FactorSpec("back", "matrix_cat", {"in": "x", "out": "y"},
                                             param_blobs={"matrix": "back"})},
                 ports={"reading": PortSpec("reading", "s", PortRole.DATA)})
    engine = make_engine(t, {"mix": np.full((2, 2, 2), .5), "back": np.eye(2)}, np.float64)
    with pytest.raises(ValueError, match="acyclic"):
        engine.eig(engine.init_state(), "x", ["reading"])
