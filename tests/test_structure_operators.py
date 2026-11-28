"""Structural candidate edits, exact observable laws and replay."""

import json
from dataclasses import replace
from itertools import product

import jax
import numpy as np
import pytest
from scipy.stats import norm

from cffg import Engine, compile_template
from cffg.core import BlobRef, ConstraintKind, ConstraintSpec, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.structure_operators import (
    AddRegime, MergeStates, PruneGaussianInput, PruneState, RejectedEdit,
    SplitState, StructuralCandidate, propose_operator,
)
from cffg.structure_proposals import ProposalError, replay_proposal
from test_gaussian_learnables import _learning_template


def mixture(weights, values, *, dirichlet=False, gaussian=False):
    size = len(weights)
    t = Template("observed-emission")
    t.variables["z"] = VariableSpec("z", cardinality=size)
    a = {"w": weights}
    if dirichlet:
        t.variables["theta"] = VariableSpec("theta", size)
        t.factors["hyper"] = FactorSpec("hyper", "dirichlet", {"out": "theta"}, param_blobs={"counts": "w"})
        t.factors["prior"] = FactorSpec("prior", "categorical", {"out": "z"}, param_edges={"statistic": "theta"})
        t.constraints = [ConstraintSpec(ConstraintKind.MEAN_FIELD, "prior")]
    else:
        t.factors["prior"] = FactorSpec("prior", "categorical", {"out": "z"}, param_blobs={"statistic": "w"})
    if gaussian:
        t.variables["x"] = VariableSpec("x", dim=1)
        t.variables["y"] = VariableSpec("y", dim=1)
        t.factors["measure"] = FactorSpec("measure", "gaussian_linear", {"in": "x", "out": "y"},
                                          param_blobs={"matrix": "H", "offset": "b", "cov": "R"})
        a.update(H=np.eye(1, dtype=values.dtype), b=np.zeros(1, dtype=values.dtype),
                 R=np.eye(1, dtype=values.dtype) * .125)
        t.factors["emit"] = FactorSpec("emit", "gaussian_mixture", {"selector": "z", "out": "x"},
                                       param_blobs={"mean": "mu", "cov": "V"})
        a.update(mu=values[:, None], V=np.ones((size, 1, 1), dtype=values.dtype))
    else:
        t.variables["y"] = VariableSpec("y", cardinality=values.shape[0])
        t.factors["emit"] = FactorSpec("emit", "matrix_cat", {"in": "z", "out": "y"}, param_blobs={"matrix": "E"})
        a["E"] = values
    t.ports["sensor"] = PortSpec("sensor", "y", PortRole.DATA)
    t.blobs = {k: BlobRef(k, k) for k in a}
    return t, a


def candidate(model, operation):
    models = {"fit": model}
    result = propose_operator("fit", models, operation)
    assert isinstance(result, StructuralCandidate), result
    record = json.loads(json.dumps(result.record))
    t, a = replay_proposal(record, models, inputs=result.inputs)
    assert t.ports == model[0].ports
    assert not any(np.shares_memory(v, model[1][k]) for k, v in a.items() if k in model[1])
    return t, a


def pair_probability(alpha, rates):
    """Probability of two successes under common Dirichlet mixture weights."""
    total = alpha.sum()
    return sum(
        alpha[i] * (alpha[j] + (i == j)) * rates[i] * rates[j]
        for i, j in product(range(len(alpha)), repeat=2)
    ) / (total * (total + 1))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dirichlet_edits_transport_concentration_and_preserve_observed_alphabet(dtype):
    with jax.enable_x64(dtype == np.float64):
        alpha = np.ones(3, dtype=dtype)
        table = np.array([[1, 0, .5], [0, 1, .5]], dtype=dtype)
        model = mixture(alpha, table, dirichlet=True)
        t, a = candidate(model, MergeStates("z", 0, 1))
        assert t.variables["theta"].cardinality == t.variables["z"].cardinality == 2
        np.testing.assert_array_equal(a["w"], [2, 1])
        assert a["w"].dtype == a["E"].dtype == dtype
        assert pair_probability(alpha, table[0]) == pytest.approx(7/24)
        assert pair_probability(a["w"], a["E"][0]) == pytest.approx(1/4)
        _, split = candidate(model, SplitState("z", 2, .25))
        assert pair_probability(split["w"], split["E"][0]) == pytest.approx(7/24)
        _, prune = candidate(model, PruneState("z", 0))
        np.testing.assert_array_equal(prune["w"], [1, 1])
        np.testing.assert_array_equal(prune["E"], table[:, 1:])
        np.testing.assert_array_equal(alpha, np.ones(3))
        model[1]["w"][0] = 0
        assert isinstance(propose_operator("fit", {"fit": model}, PruneState("z", 0)), RejectedEdit)


def test_subdivided_emissions_use_the_same_sensor_and_native_tree_likelihood():
    weights = np.array([.25, .75])
    model = mixture(weights, np.array([[1., .5], [0., .5]]))
    op = SplitState("z", 1, .5, ([.4, .6], [.8, .2]))
    t, a = candidate(model, op)
    engine = Engine(compile_template(t, arrays=a))
    observed = engine.bind_data(engine.init_state(), {"sensor": 1})
    result = engine.run(observed, 100)
    result.require_converged()
    assert result.free_energies[-1] == pytest.approx(-np.log(.375 * .6 + .375 * .2))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gaussian_birth_and_regime_prune_at_scale(dtype):
    with jax.enable_x64(dtype == np.float64):
        means = np.arange(40, dtype=dtype)
        weights = np.zeros(40, dtype=dtype)
        weights[0] = 1
        model = mixture(weights, means, gaussian=True)
        op = AddRegime("z", np.array([10.], dtype=dtype), np.array([[.25]], dtype=dtype), .25, -5.)
        t, a = candidate(model, op)
        assert t.variables["z"].cardinality == 41
        assert a["mu"][-1, 0] == 10 and a["V"][-1, 0, 0] == .25
        np.testing.assert_allclose(a["w"][[0, -1]], [.8, .2])
        np.testing.assert_array_equal(a["w"][1:-1], 0)
        # Integrate the sensor noise independently for one future measurement.
        def predictive(parameters, y):
            scales = np.sqrt(parameters["V"][:, 0, 0] + parameters["R"][0, 0])
            return parameters["w"] @ norm.pdf(y, loc=parameters["mu"][:, 0], scale=scales)

        assert predictive(a, 9.5) > predictive(model[1], 9.5)
        assert predictive(a, 0.1) < predictive(model[1], 0.1)
        assert propose_operator("fit", {"fit": model}, replace(op, capacity=40)).code == "capacity"
        at_threshold = replace(op, observation=np.zeros(1), concentration=1.,
                               log_base_density=-.5 * np.log(2 * np.pi))
        assert propose_operator("fit", {"fit": model}, at_threshold).code == "existing_regime"
        _, undo = candidate((t, a), PruneState("z", 40))
        for k in undo:
            np.testing.assert_array_equal(undo[k], model[1][k])
        fixed = AddRegime("z", np.array([0.]), np.eye(1), 1.0, -1.)
        assert propose_operator("fit", {"fit": model}, fixed).code == "existing_regime"
        assert propose_operator("fit", {"fit": model}, PruneState("z", 0)).code == "empty_support"
        assert isinstance(propose_operator("fit", {"fit": model}, SplitState("z", 0)), RejectedEdit)


def test_pruned_affine_input_changes_only_the_scored_gaussian_law():
    t, a = _learning_template(1)
    assert isinstance(propose_operator("fit", {"fit": (t, a)}, PruneGaussianInput("transition")), RejectedEdit)
    t.learnables.clear()  # Point estimates retained as the source artifact.
    pruned = candidate((t, a), PruneGaussianInput("transition"))
    assert set(pruned[0].variables) == {"x1", "y"}
    assert set(pruned[1]) == {"b", "Q", "H", "h", "R"}
    for model, variance in [((t, a), 1.335), (pruned, .6)]:
        engine = Engine(compile_template(model[0], arrays=model[1]))
        for y in [0., 2.]:
            state = engine.bind_data(engine.init_state(), {"observation": np.array([y])})
            result = engine.run(state, 100)
            result.require_converged()
            assert result.free_energies[-1] == pytest.approx(-norm.logpdf(y, scale=np.sqrt(variance)))


def test_rejected_and_tampered_candidates_leave_the_parent_intact():
    model = mixture(np.array([.6, 0., .4]), np.ones((2, 3))/2)
    models = {"fit": model}
    for op in [MergeStates("z", 0, 0), SplitState("z", -1), SplitState("z", 1, 0.),
               SplitState("z", 0, .5, ([1., 1.], [0., 1.])), PruneGaussianInput("emit")]:
        assert isinstance(propose_operator("fit", models, op), RejectedEdit)
    result = propose_operator("fit", models, MergeStates("z", 2, 1))
    np.testing.assert_array_equal(result.inputs["w"], [.6, .4])
    result.inputs["E"][:, 0] = [.25, .75]
    with pytest.raises(ProposalError, match="identity"):
        replay_proposal(result.record, models, inputs=result.inputs)
    np.testing.assert_array_equal(model[1]["E"], .5)
