"""Behavioral epistemics beyond the canonical T-maze golden.

The oracle is a binary Bayes decision in a ground-truth simulator, not a
restatement of the GFE update.  The suite asks whether information is relevant,
calibrated, self-terminating, and useful in realized task return.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from cffg.engine import RunStatus
from cffg.experiments.behavioral_epistemics import (
    CHECK_CUE,
    CHECK_DISTRACTOR,
    CUE,
    CUE_LEFT,
    CUE_RIGHT,
    DISTRACTOR,
    GO_LEFT,
    GO_RIGHT,
    N_CONTEXT,
    CueTask,
    candidate_mutual_information,
    goal_statistic,
    observation_matrix,
    plan_decision,
    run_benchmark,
)


@pytest.fixture(scope="module")
def reports():
    task = CueTask()
    return task, run_benchmark(task, gfe=True), run_benchmark(task, gfe=False)


@pytest.mark.slow
def test_costly_cue_improves_realized_return_over_bfe(reports):
    """GFE buys the cue and then exploits; the BFE ablation guesses an arm.

    Return and regret are measured in the simulator over every seed.  MI is
    computed from the normalized context/observation joint, while convergence
    and FE come from the public runtime result/ledger.
    """
    task, gfe, bfe = reports

    assert gfe.first_plan.action == CHECK_CUE
    assert bfe.first_plan.action in (GO_LEFT, GO_RIGHT)
    assert gfe.conditional_plans[CUE_LEFT].action == GO_LEFT
    assert gfe.conditional_plans[CUE_RIGHT].action == GO_RIGHT

    assert np.all(gfe.task_return_by_seed > bfe.task_return_by_seed + 0.65)
    assert np.all(gfe.regret_by_seed < bfe.regret_by_seed - 0.65)
    assert abs(gfe.task_return_by_seed.mean() - task.bayes_expected_return) < 0.12

    for report in (gfe, bfe):
        assert all(status is RunStatus.CONVERGED for status in report.convergence_statuses)
        assert all(np.isfinite(report.free_energies))
        assert report.first_plan.messages_finite
        assert all(plan.messages_finite for plan in report.conditional_plans.values())

    # The realized first-cue KL is constant in this symmetric binary channel
    # and therefore equals its posterior mutual information in expectation.
    np.testing.assert_allclose(
        gfe.realized_information_gain_by_seed,
        gfe.candidate_posterior_mi[CHECK_CUE],
        atol=1e-12,
    )


@pytest.mark.slow
def test_relevant_cue_outranks_random_distractor(reports):
    """Both sensors have a uniform observation marginal, but only the cue's
    outcome covaries with the reward context."""
    task, gfe, _ = reports
    probabilities = gfe.first_plan.action_probabilities
    assert probabilities[CHECK_CUE] > probabilities[CHECK_DISTRACTOR] + 0.04

    mi = candidate_mutual_information(task)
    expected_binary_mi = (
        np.log(2.0)
        + task.cue_quality * np.log(task.cue_quality)
        + (1.0 - task.cue_quality) * np.log(1.0 - task.cue_quality)
    )
    np.testing.assert_allclose(mi[CHECK_CUE], expected_binary_mi, atol=1e-12)
    np.testing.assert_allclose(mi[CHECK_DISTRACTOR], 0.0, atol=1e-12)

    matrix = observation_matrix(task)
    cue_columns = matrix[:, CUE * N_CONTEXT : (CUE + 1) * N_CONTEXT]
    distractor_columns = matrix[
        :, DISTRACTOR * N_CONTEXT : (DISTRACTOR + 1) * N_CONTEXT
    ]
    cue_marginal = cue_columns.mean(axis=1)
    distractor_marginal = distractor_columns.mean(axis=1)
    cue_entropy = -np.sum(cue_marginal[cue_marginal > 0] * np.log(cue_marginal[cue_marginal > 0]))
    distractor_entropy = -np.sum(
        distractor_marginal[distractor_marginal > 0]
        * np.log(distractor_marginal[distractor_marginal > 0])
    )
    np.testing.assert_allclose(cue_entropy, distractor_entropy, atol=1e-12)


@pytest.mark.slow
def test_information_satiates_after_observing_the_cue(reports):
    task, gfe, _ = reports
    before = gfe.first_plan.action_probabilities
    for observation, correct_arm in ((CUE_LEFT, GO_LEFT), (CUE_RIGHT, GO_RIGHT)):
        after = gfe.conditional_plans[observation]
        assert after.action == correct_arm
        assert after.action_probabilities[CHECK_CUE] < before[CHECK_CUE] - 0.08
        assert after.action_probabilities[[GO_LEFT, GO_RIGHT]].sum() > before[
            [GO_LEFT, GO_RIGHT]
        ].sum() + 0.12
        assert after.context_belief.max() > task.cue_quality - 0.03


@pytest.mark.slow
def test_sensing_cost_is_monotone_and_crosses_bayes_threshold():
    """The external two-action Bayes calculation gives c*=1.2 utility units.
    The engine changes its MAP action in the immediately adjacent cost bin."""
    base = CueTask()
    threshold = base.bayes_sensing_threshold
    np.testing.assert_allclose(threshold, 1.2, atol=1e-12)
    costs = np.array([0.6, 0.9, 1.1, threshold, 1.3, 1.5])
    plans = [plan_decision(CueTask(cue_cost=float(cost))) for cost in costs]
    sensing_probability = np.array(
        [plan.action_probabilities[CHECK_CUE] for plan in plans]
    )
    chosen_sensing = np.array([plan.action == CHECK_CUE for plan in plans])

    assert np.all(np.diff(sensing_probability) < 0)
    assert np.all(np.diff(chosen_sensing.astype(int)) <= 0)
    first_exploit = int(np.flatnonzero(~chosen_sensing)[0])
    assert costs[first_exploit - 1] <= threshold <= costs[first_exploit]
    assert costs[first_exploit] - costs[first_exploit - 1] <= 0.11


@pytest.mark.slow
def test_exact_observation_epistemics_tracks_realisable_sensor_quality():
    """Observation GFE weakens with the actual channel MI.  This is the exact,
    realizable channel—not the GFE-TM latent-next-state optimistic bound in
    LIMITATIONS S1c."""
    qualities = np.array([0.5, 0.65, 0.8, 0.9, 0.99])
    plans = [plan_decision(CueTask(cue_quality=float(q))) for q in qualities]
    cue_probability = np.array(
        [plan.action_probabilities[CHECK_CUE] for plan in plans]
    )
    cue_mi = np.array(
        [candidate_mutual_information(CueTask(cue_quality=float(q)))[CHECK_CUE] for q in qualities]
    )

    np.testing.assert_allclose(
        plans[0].action_probabilities[CHECK_CUE],
        plans[0].action_probabilities[CHECK_DISTRACTOR],
        atol=1e-10,
    )
    assert np.all(np.diff(cue_probability) > 0)
    assert np.all(np.diff(cue_mi) > 0)


@pytest.mark.slow
def test_preference_offset_invariance_and_goal_calibration():
    task = CueTask(cue_cost=1.0)
    np.testing.assert_allclose(
        goal_statistic(task), goal_statistic(task, utility_offset=1_000.0), atol=0.0
    )
    base = plan_decision(task)
    shifted = plan_decision(task, utility_offset=1_000.0)
    np.testing.assert_allclose(
        base.action_probabilities, shifted.action_probabilities, atol=1e-10
    )

    # Increasing goal contrast eventually dominates the bounded epistemic
    # pull: curiosity cannot swamp the reward/loss preference indefinitely.
    reward_scales = (1.0, 1.5, 4.0)
    plans = [
        plan_decision(CueTask(cue_cost=1.0, reward_utility=scale))
        for scale in reward_scales
    ]
    cue_probability = np.array(
        [plan.action_probabilities[CHECK_CUE] for plan in plans]
    )
    assert np.all(np.diff(cue_probability) < 0)
    assert plans[0].action == CHECK_CUE
    assert plans[-1].action in (GO_LEFT, GO_RIGHT)


def test_sparse_end_to_end_behavior_survives_float32():
    """The model starts from vacuous messages and contains exact log-zeros in
    deterministic A/B tables; consumer-default float32 must retain behavior."""
    with jax.enable_x64(False):
        plan = plan_decision()
        assert plan.action_probabilities.dtype == np.float32
        assert plan.status is RunStatus.CONVERGED
        assert plan.action == CHECK_CUE
        assert plan.messages_finite
        assert np.isfinite(plan.free_energy)
