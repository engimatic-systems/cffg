"""Independent held-out likelihoods, partition isolation, and public solver parity."""

from dataclasses import replace
import math

import jax
import numpy as np
import pytest
from scipy.special import digamma
from test_compiler_contracts import model as contract_model

from cffg import Engine, compile_template
from cffg.core import BlobRef, FactorSpec, LearnableSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.serialize import structure_hash
from cffg.structure_scoring import Candidate, Partition, compare_candidates


EMPTY = Partition("train", {})


def categorical(name, probabilities, *, learned=False):
    probabilities = np.asarray(probabilities)
    template = Template(
        "readout",
        variables={"x": VariableSpec("x", len(probabilities))},
        factors={"p": FactorSpec("p", "categorical", {"out": "x"}, param_blobs={"statistic": "p"})},
        ports={"x": PortSpec("x", "x", PortRole.DATA)},
        blobs={"p": BlobRef("p", "p.npy")},
    )
    priors = None
    if learned:
        template.learnables["p"] = LearnableSpec("p", "dirichlet", ("p",))
        priors = {"p": np.ones_like(probabilities)}
    return Candidate(name, template, {"p": probabilities}, priors)


def partition(*values):
    return Partition("validation", {"run-B": [{"x": value} for value in values]})


def test_train_only_posterior_and_separate_repeated_heldout_events():
    candidate = categorical("learn", [.5, .5], learned=True)
    train = Partition("train", {"run-A": [{"x": 1}, {"x": 1}, {"x": 0}]})
    comparison = [candidate, categorical("fixed", [.9, .1])]
    scores = compare_candidates(comparison, train, partition(1, 1, 0))
    learned = scores["learn"]
    # Conjugate rollover uses E[log θ]. A Dirichlet(1,1) becomes (2,3).
    expected = digamma(5) - digamma([3, 3, 2])
    np.testing.assert_allclose([e.energies[-1] for e in learned.evaluation_events], expected, atol=1e-12)
    assert learned.rank == 1 and scores["fixed"].rank == 2
    assert learned.total_bfe == pytest.approx(sum(expected))
    assert learned.template_hash == structure_hash(candidate.template)
    assert (learned.fit_artifacts, learned.evaluation_artifacts) == (("run-A",), ("run-B",))
    assert [e.index for e in learned.fit_events] == [0, 1, 2]
    reverse = compare_candidates([candidate], train, partition(0, 1), grouped=False)["learn"]
    assert learned.fitted_params_sha256 == reverse.fitted_params_sha256
    assert reverse.evaluation_events[0].energies == learned.evaluation_events[2].energies
    np.testing.assert_array_equal(candidate.priors["p"], [1, 1])
    with pytest.raises(ValueError, match="original artifact"):
        compare_candidates(comparison, train, Partition("another-name", train.artifacts))
    with pytest.raises(ValueError, match="partition IDs"):
        compare_candidates(comparison, train, Partition("train", {"other": [{}]}))


def hidden(name, prior, matrix, *, loop=False):
    template = Template(
        "mixture",
        variables={"z": VariableSpec("z", len(prior)), "x": VariableSpec("x", 2)},
        ports={"x": PortSpec("x", "x", PortRole.DATA)},
        factors={"p": FactorSpec("p", "categorical", {"out": "z"}, param_blobs={"statistic": "p"}),
                 "A": FactorSpec("A", "matrix_cat", {"in": "z", "out": "x"}, param_blobs={"matrix": "A"})},
        blobs={"p": BlobRef("p", "p.npy"), "A": BlobRef("A", "A.npy")},
    )
    arrays = {"p": np.asarray(prior), "A": np.asarray(matrix)}
    if loop:
        del template.factors["p"], template.blobs["p"], arrays["p"]
        for edge in ("a", "b", "c"):
            template.variables[edge] = VariableSpec(edge, len(prior))
        template.factors["eq"] = FactorSpec("eq", "equality", {"a": "z", "b": "a", "c": "b"})
        for name_, inp, out in (("B1", "a", "c"), ("B2", "c", "b")):
            template.factors[name_] = FactorSpec(name_, "matrix_cat", {"in": inp, "out": out},
                                                 param_blobs={"matrix": "B"})
        template.blobs["B"] = BlobRef("B", "B.npy")
        arrays["B"] = np.array([[.7, .2], [.3, .8]], dtype=arrays["A"].dtype)
    return Candidate(name, template, arrays)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_analytic_tree_and_relabelled_latent_tie_in_native_precision(dtype):
    prior = np.array([.3, .7], dtype=dtype)
    matrix = np.array([[.1, .8], [.9, .2]], dtype=dtype)
    candidates = [hidden("original", prior, matrix), hidden("swapped", prior[::-1], matrix[:, ::-1]),
                  hidden("other", prior, matrix[::-1])]
    with jax.enable_x64(dtype == np.float64):
        scores = compare_candidates(candidates, EMPTY, partition(0))
        expected = -math.log(matrix[0] @ prior)
        for candidate in candidates[:2]:
            row = scores[candidate.candidate_id]
            assert row.rank == 1 and row.approximation is None and row.dtype == np.dtype(dtype).name
            assert row.total_bfe == pytest.approx(expected, abs=12*np.finfo(dtype).eps)
            engine = Engine(compile_template(candidate.template, arrays=candidate.arrays))
            result = engine.run(engine.bind_data(engine.init_state(), {"x": 0}), 100, tol=1e-6)
            assert row.evaluation_events[0].energies == result.free_energies
        assert scores["other"].rank == 3


@pytest.mark.parametrize("schedule", ["auto", pytest.param("flood", marks=pytest.mark.slow)])
def test_loop_grouped_and_single_retain_the_same_bethe_evidence(schedule):
    model = hidden("A", [.5, .5], [[.2, .7], [.8, .3]], loop=True)
    peer = replace(model, candidate_id="B", arrays={**model.arrays, "A": model.arrays["A"][::-1]})
    inputs = [model, peer], EMPTY, partition(1, None)
    grouped = compare_candidates(*inputs, schedule=schedule)
    singles = compare_candidates(*inputs, schedule=schedule, grouped=False)
    for cid in grouped:
        actual, reference = grouped[cid], singles[cid]
        assert actual.rank == reference.rank and actual.status == "converged"
        assert "Loopy Bethe approximation" in actual.approximation
        assert actual.schedule == ("colored" if schedule == "auto" else "flood")
        for a, b in zip(actual.evaluation_events, reference.evaluation_events, strict=True):
            assert a.status == b.status and a.iterations == b.iterations
            np.testing.assert_allclose(a.energies, b.energies, atol=1e-11)


def test_sparse_vacuous_exhausted_invalid_and_empty_keep_their_native_evidence():
    impossible, fixed = categorical("zero", [1., 0.]), categorical("fixed", [.5, .5])
    bad = replace(fixed, candidate_id="bad", arrays={"p": np.ones(3)})
    gfe = Candidate("planning", *contract_model("gfe"))
    rows = compare_candidates([impossible, fixed, bad, gfe], EMPTY, partition(1, None))
    assert rows["zero"].status == "infeasible" and rows["zero"].rank is None
    assert math.isinf(rows["zero"].evaluation_events[0].energies[-1])
    assert rows["zero"].evaluation_events[1].status.value == "converged"
    assert rows["fixed"].rank == 1
    for cid in ("planning", "bad"):
        assert rows[cid].status == "invalid" and rows[cid].rank is None
    assert "P-substitution" in rows["planning"].error
    engine = Engine(compile_template(fixed.template, arrays=fixed.arrays))
    result = engine.run(engine.bind_data(engine.init_state(), {"x": None}), 100, tol=1e-6)
    assert rows["fixed"].evaluation_events[1].energies == result.free_energies
    exhausted = compare_candidates([fixed], EMPTY, partition(0), n_iters=0)["fixed"]
    assert exhausted.status == "budget_exhausted" and exhausted.total_bfe is None
    empty = compare_candidates([fixed], EMPTY, partition())["fixed"]
    assert empty.status == "empty_evaluation" and empty.rank is None


def test_improper_unanchored_gaussian_preserves_engine_failure():
    template = Template(
        "unanchored",
        variables={edge: VariableSpec(edge, dim=1) for edge in ("x", "y")},
        factors={name: FactorSpec(name, "gaussian_linear", {"in": a, "out": b},
                                  param_blobs={"matrix": "F", "offset": "b", "cov": "Q"})
                 for name, a, b in (("forward", "x", "y"), ("reverse", "y", "x"))},
        blobs={name: BlobRef(name, name) for name in ("F", "b", "Q")},
    )
    candidate = Candidate("improper", template, {"F": np.eye(1), "b": np.zeros(1), "Q": np.eye(1)})
    result = compare_candidates([candidate], EMPTY, Partition("heldout", {"tape": [{}]}))["improper"]
    native = Engine(compile_template(template, arrays=candidate.arrays))
    direct = native.run(native.init_state(), 100, tol=1e-6)
    assert result.status == direct.status.value and result.status in ("infeasible", "diverged")
    assert result.rank is None and result.total_bfe is None
    assert result.evaluation_events[0].energies == direct.free_energies


def test_forty_candidates_with_twenty_latent_labels():
    priors = np.full(20, .05)
    candidates = [hidden(str(i), priors, np.tile([[.2 + i/100], [.8 - i/100]], (1, 20)))
                  for i in range(40)]
    rows = compare_candidates(candidates, EMPTY, partition(0))
    for i, row in enumerate(rows.values()):
        assert row.rank == 40 - i and row.status == "converged"
        assert row.total_bfe == pytest.approx(-math.log(.2+i/100))


def test_invalid_fit_and_invalid_data_leave_valid_peers_scoreable():
    learned = categorical("learn", [.4, .6], learned=True)
    invalid = replace(learned, candidate_id="invalid-prior", priors={"p": np.array([0., 1.])})
    fit = Partition("train", {"run-A": [{"x": 0}]})
    fixed = categorical("fixed", [.4, .6])
    rows = compare_candidates([learned, invalid, fixed], fit, partition(1))
    assert rows["invalid-prior"].status == "invalid" and rows["invalid-prior"].rank is None
    assert rows["learn"].rank is not None and rows["fixed"].rank is not None
    bad_binding = partition(1, 0.5)
    event = compare_candidates([fixed], EMPTY, bad_binding)["fixed"]
    assert event.status == "invalid" and event.rank is None
    assert len(event.evaluation_events) == 1 and "run-B[1]" in event.error
    incomplete = compare_candidates([learned, fixed], fit, partition(1), n_iters=0)
    assert incomplete["learn"].fit_events[-1].status.value == "budget_exhausted"
    assert not incomplete["learn"].evaluation_events and incomplete["learn"].rank is None


def test_one_measurement_cannot_change_domain_between_ranked_models():
    with pytest.raises(ValueError, match="incomparable observation domains"):
        compare_candidates([categorical("two", [.5, .5]), categorical("three", [.4, .4, .2])],
                           EMPTY, partition(0))
