"""The outer search chooses among replayed ordinary models while the actor chooses readings."""
from copy import deepcopy
from dataclasses import replace
import math

import jax
import numpy as np
import pytest

from cffg.structure_operators import AddRegime, PruneState, RejectedEdit, StructuralCandidate, propose_operator
from cffg.structure_proposals import record_proposal, replay_proposal
from cffg.structure_scoring import Candidate, Partition, compare_candidates
from cffg.structure_search import ProposalTrigger, search_round, source_candidate
from test_structure_operators import mixture


def readings(name, *values):
    return Partition(name, {name: [{"sensor": y} for y in values]})


def test_trigger_counts_consecutive_completed_evaluations_of_one_incumbent():
    models = {"map": mixture(np.array([.8, .2]), np.eye(2))}
    root = source_candidate("map", models)
    parent = root.record["candidate_id"]
    candidate = Candidate(parent, *models["map"])
    fit = readings("fit", 0)
    def evaluate(value, name):
        return compare_candidates([candidate], fit, readings(name, value))[parent]
    high = evaluate(1, "first-validation")
    low = evaluate(0, "second-validation")
    assert high.total_bfe == pytest.approx(-math.log(.2))
    assert low.total_bfe == pytest.approx(-math.log(.8))
    trigger = ProposalTrigger(.7, 2)
    assert not trigger.observe(high)
    assert not trigger.observe(low)
    assert not trigger.observe(high)
    assert trigger.observe(high)
    assert not trigger.observe(high)
    assert not trigger.observe(replace(high, status="diverged", total_bfe=None))
    assert not trigger.observe(high)
    assert not trigger.observe(replace(high, candidate_id="another"))
    assert not trigger.observe(high)
    assert trigger.observe(high)
    assert not ProposalTrigger(high.total_bfe, 1).observe(high)
    for threshold, count in ((0, 1), (float("inf"), 1), (np.nan, 1), (.5, 0), (.5, 1.5), (True, 1)):
        with pytest.raises(ValueError):
            ProposalTrigger(threshold, count)


def test_foreign_tampered_failed_and_duplicate_proposals_leave_valid_peers_scoreable():
    models = {"map": mixture(np.array([.5, .5]), np.array([[.95, .05], [.05, .95]]))}
    parent = source_candidate("map", models)
    parent_id = parent.record["candidate_id"]
    prune = PruneState("z", 0)
    child = propose_operator("map", models, prune, parent_id=parent_id)
    other_parent = propose_operator("map", models, prune, parent_id="other")
    altered = StructuralCandidate(deepcopy(child.record), child.inputs)
    altered.record["proposal_source"] = "changed"
    fit, heldout = readings("train", 0), readings("heldout", 1, 1)
    round_ = search_round("map", models, [altered, other_parent, RejectedEdit("capacity", "full"),
                          child, prune], fit, heldout, incumbent=parent)
    assert [(f.index, f.code) for f in round_.failures] == [
        (0, "invalid_record"), (1, "invalid_record"), (2, "capacity"), (4, "duplicate_candidate"),
    ]
    selected = round_.selected_id
    assert selected == child.record["candidate_id"] and round_.reason == "selected_challenger"
    assert round_.scores[selected].total_bfe == pytest.approx(-2 * math.log(.95))
    assert round_.scores[parent_id].total_bfe == pytest.approx(-2 * math.log(.5))
    assert round_.candidates[selected].record["parent_id"] == parent_id

    # An independent producer using the same edit gets the same score, with its own provenance.
    external = StructuralCandidate(record_proposal(
        source_id="map", parent_id=parent_id, proposal_source="external",
        steps=child.record["steps"], models=models, inputs=child.inputs,
    ), child.inputs)
    tied = search_round("map", models, [child, external], fit, heldout, grouped=False)
    assert tied.selected_id is None and tied.reason == "tied_challengers"
    saved = round_.candidates[selected]
    models["accepted"] = replay_proposal(saved.record, models, inputs=saved.inputs)
    next_round = search_round("accepted", models, [], fit, heldout, incumbent=saved)
    assert next_round.selected_id == selected and next_round.reason == "incumbent_best"
    with pytest.raises(ValueError, match="differs"):
        search_round("map", models, [], fit, heldout, incumbent=saved)
    no_rank = search_round("map", models, [prune], fit, heldout, n_iters=0)
    assert no_rank.selected_id is None and no_rank.reason == "no_ranked_model"
    assert all(r.status == "budget_exhausted" for r in no_rank.scores.values())


def test_twenty_prunes_on_twenty_labels_preserve_native_scores_and_ties():
    counts = np.arange(1, 21, dtype=np.float64)
    prior = counts / counts.sum()
    emissions = np.repeat([[.75], [.25]], 20, axis=1)
    models = {"source": mixture(prior, emissions)}
    report = search_round("source", models, [PruneState("z", i) for i in range(20)],
                          readings("fit", 1), readings("eval", 0, 1))
    assert len(report.scores) == 21 and not report.failures
    assert report.reason == "incumbent_tied" and report.selected_id == report.incumbent_id
    assert all(row.rank == 1 and row.status == "converged" for row in report.scores.values())
    np.testing.assert_allclose([r.total_bfe for r in report.scores.values()], -math.log(.75*.25))


@pytest.mark.slow
def test_native_d5_sampling_and_ordinary_structure_comparison():
    from test_hybrid_gfe import _probe_template, _run

    energies, risks = {}, {}
    for action in ("idle", "probe"):
        world, acting, arrays = _probe_template(action)
        engine, result = _run(acting, arrays)
        result.require_converged()
        diag = engine.hybrid_gfe_diagnostics(result.state)["emission"]
        energies[action], risks[action] = diag["gfe_lower_bound"], diag["goal_risk"]
    chosen = min(energies, key=energies.get)
    assert chosen == "probe" and risks["idle"] < risks["probe"]
    _, offsets, noise = world.emission(chosen)

    def sample(regimes, seed):
        rng = np.random.default_rng(seed)
        return [np.array([offsets[k, 0] + rng.normal(0, np.sqrt(noise[k, 0, 0]))]) for k in regimes]

    train = sample([0, 1], 101)
    eval_data = sample([0, 1, 0, 1], 202)
    fit, heldout = readings("training-101", *train), readings("evaluation-202", *eval_data)
    totals = []
    for use_x64 in (False, True):
        with jax.enable_x64(use_x64):
            dtype = np.float64 if use_x64 else np.float32
            model = mixture(np.array([.5, .5], dtype=dtype), np.array([-1., -1.], dtype=dtype), gaussian=True)
            model[1]["V"] *= .0625
            models = {"sensor": model}
            growth = AddRegime("z", train[0], np.array([[.0625]]), 1., -1.)
            report = search_round("sensor", models, [growth], fit, heldout)
            assert report.reason == "selected_challenger" and not report.failures
            rows = [report.scores[report.incumbent_id], report.scores[report.selected_id]]
            totals.append([row.total_bfe for row in rows])
            for row in rows:
                assert row.dtype == np.dtype(dtype).name
                assert row.fit_artifacts == ("training-101",) and row.evaluation_artifacts == ("evaluation-202",)
                assert row.total_bfe == math.fsum(event.energies[-1] for event in row.evaluation_events)
                assert len(row.evaluation_events) == len(eval_data)
            # Both incumbent regimes have mean -1, so even the marginal mixture
            # is one ordinary Gaussian with process and sensor variances added.
            variance = float(model[1]["V"][0, 0, 0] + model[1]["R"][0, 0])
            exact = math.fsum(
                .5 * (math.log(2 * math.pi * variance) + (float(y[0]) + 1) ** 2 / variance)
                for y in eval_data
            )
            assert rows[0].total_bfe == pytest.approx(exact, rel=2e-6)
            assert rows[1].total_bfe < rows[0].total_bfe
    np.testing.assert_allclose(*totals, rtol=2e-6)
