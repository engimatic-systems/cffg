"""Finite-world and hand-scored checks for the AXIOM rollout/proposal rules."""

from dataclasses import replace
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.experiments.axiom_planning import SearchSettings, initial_proposal, plan, refit
from cffg.experiments.axiom_prediction import ObjectDynamics, predict_objects


def dynamics(dtype):
    return ObjectDynamics(
        jnp.array([2, 6, 2], dtype), jnp.array([True, True, False]),
        jnp.array([[.6, .4], [.2, .8], [0, 1]], dtype),
        jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype),
        jnp.broadcast_to(jnp.eye(2, 3, dtype=dtype), (2, 2, 3)),
        jnp.zeros(2, dtype))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_score_precedence_and_tracking(dtype):
    bank = dynamics(dtype)
    # Dormant component has a high likelihood and still contributes to total alpha.
    scores = jnp.array([[0, -np.inf, 9], [-np.inf, 0, 9], [-15, -np.inf, 9],
                        [-15, -np.inf, 9], [np.nan, np.nan, 9]], dtype)
    state = jnp.zeros((5, 3), dtype).at[:, -1].set(7)
    state = state.at[3, 0].set(1.2).at[1, 1].set(-1.1)
    tracked = jnp.array([True, True, True, True, False])
    obs = jax.jit(predict_objects)(bank, state, tracked, scores, jax.random.key(1))
    # The first object wins the +/-1 absolute reward tie. Bonuses use all five slots.
    assert obs.reward == -1
    assert float(obs.novelty) == pytest.approx((.4 + (1/6 - .1) + .9 - 1 + 0) / 5, abs=1e-7)
    np.testing.assert_array_equal(obs.state, state)
    np.testing.assert_array_equal(obs.component, [0, 1, 0, 0, -1])
    np.testing.assert_array_equal(bank.alpha, [2, 6, 2])


def test_interaction_then_conditional_motion_enumeration():
    bank = dynamics(jnp.float64)
    # Class 0 adds one unit to x and class 1 subtracts one.
    bank = bank._replace(affine=bank.affine.at[:, 0, -1].set(jnp.array([1., -1.])))
    evidence = jnp.log(jnp.array([[.25, .75, 0.]]))
    state = jnp.zeros((1, 2))
    pred = jax.jit(jax.vmap(lambda k: predict_objects(
        bank, state, jnp.array([True]), evidence, k
    )))(jax.random.split(jax.random.key(12), 4096))
    labels = 2 * pred.component[:, 0] + pred.motion[:, 0]
    np.testing.assert_allclose(np.bincount(labels, minlength=4) / len(labels),
                               [.15, .10, .15, .60], atol=.025)
    np.testing.assert_array_equal(pred.state[:, 0, 0], np.where(pred.motion[:, 0] == 0, 1, -1))
    # Selected positive head overrides the mixture expectation. Other draws use E[r]=.5.
    np.testing.assert_allclose(pred.reward, np.where(pred.component[:, 0] == 1, 1., .5),
                               atol=2e-16, rtol=0)
    np.testing.assert_array_equal(state, [[0., 0.]])


def test_missing_motion_support_and_equal_threshold():
    bank = dynamics(jnp.float32)
    evidence = jnp.array([[0., -jnp.inf, -jnp.inf]], jnp.float32)
    state = jnp.ones((1, 2), jnp.float32)
    def read(model, scores=evidence):
        return predict_objects(model, state, jnp.array([True]), scores, jax.random.key(8))
    head = jnp.array([.55, 0., .45], dtype=state.dtype)
    assert float(read(bank._replace(reward_probability=bank.reward_probability.at[0].set(head))).reward
                 ) == pytest.approx(-.55)
    no_motion = bank._replace(motion_probability=bank.motion_probability.at[0].set(0))
    assert jnp.isnan(read(no_motion).reward)
    assert jnp.isnan(read(bank, jnp.full_like(evidence, -jnp.inf)).novelty)


def test_discounted_refit_can_reverse_first_action_ranking():
    settings = SearchSettings(candidates=20, discount=.5)
    prior = initial_proposal(2, horizon=2, dtype=jnp.float64)
    policies = jnp.zeros((20, 2), jnp.int32).at[1].set(1)
    reward = jnp.full((20, 2), -2.).at[0].set(jnp.array([2., 0.])).at[1].set(jnp.array([0., 3.]))
    fit = refit(policies, reward, jnp.zeros_like(reward), prior, settings)
    # Discounted scores 2, 1.5, -3. Normalize over the entire pool before the elite softmax.
    np.testing.assert_allclose(fit.proposal.probabilities, [np.exp([1., 0.]) / (np.e + 1)] * 2)
    assert fit.action == 1
    assert fit.discounted_return.argmax() == 0
    assert fit.total_return.argmax() == 1


def choice_step(snapshot, state, action, key):
    del key
    return state, (snapshot[action] * state, (action == 0).astype(snapshot.dtype))


@pytest.mark.parametrize("dtype", [jnp.float32, pytest.param(jnp.float64, marks=pytest.mark.slow)])
def test_replan_action_support_and_no_bonus(dtype):
    previous = initial_proposal(2, horizon=1, dtype=dtype)
    options = SearchSettings(candidates=12)
    snapshot = jnp.array([0., .05], dtype)
    def decide(p, seen, key, mask, settings):
        return plan(choice_step, snapshot, seen, p, key, support=mask, settings=settings)
    run = jax.jit(decide, static_argnames=("settings",))
    all_actions = jnp.ones(2, dtype=bool)
    key = jax.random.key(123)
    first = run(previous, 1., key, all_actions, options)
    disabled = run(previous, 1., key, all_actions, replace(options, novelty_weight=0.))
    assert first.valid and first.action == 0 and disabled.action == 1
    np.testing.assert_array_equal(first.reward, disabled.reward)
    np.testing.assert_array_equal(first.novelty, disabled.novelty)
    # A different real observation reverses the ranking without changing model parameters.
    second = run(first.proposal, 3., jax.random.key(124), all_actions, options)
    assert second.valid and second.action == 1
    only_one = run(first.proposal, 1., key, jnp.array([False, True]), options)
    assert only_one.valid and only_one.action == 1 and (only_one.policies == 1).all()
    assert (only_one.proposal.probabilities == jnp.array([0., 1.])).all()
    empty = run(previous, 1., key, ~all_actions, options)
    assert not empty.valid and empty.action == -1
    np.testing.assert_array_equal(empty.proposal.probabilities, previous.probabilities)
    bad = run(previous, jnp.nan, key, all_actions, options)
    assert not bad.valid and bad.action == -1


def test_smoothing_injected_strings_replan_and_constant_return_ties():
    previous = initial_proposal(3, horizon=5)
    previous = previous._replace(best=jnp.array([2, 1, 0, 1, 0], jnp.int32))
    uniform_tie = SearchSettings(candidates=40, novelty_weight=0.)
    def run(options):
        return plan(choice_step, jnp.zeros(3), 1., previous, jax.random.key(5), settings=options)
    independent = run(uniform_tie)
    assert independent.valid
    assert independent.action == independent.policies[-1, 0]
    np.testing.assert_array_equal(independent.policies[0], [1, 0, 1, 0, 2])
    np.testing.assert_array_equal(independent.policies[1:4], [[0]*5, [1]*5, [2]*5])
    random_half = independent.policies[20:]
    np.testing.assert_array_equal(random_half[:, 0], random_half[:, 1])
    assert (random_half[:, 1:-1] != random_half[:, 2:]).any()
    repeated = run(replace(uniform_tie, repeat_probability=1.)).policies[20:]
    assert (repeated == repeated[:, :1]).all()
    # Stable elite order: the shifted plan and the three constant strings.
    histogram = np.eye(3)[np.asarray(independent.policies[:4])].mean(axis=0)
    np.testing.assert_allclose(independent.proposal.probabilities, histogram)
    constant = refit(independent.policies, jnp.ones_like(independent.reward),
                     independent.novelty, previous, uniform_tie)
    assert constant.valid and constant.action == independent.action
    np.testing.assert_allclose(constant.proposal.probabilities, histogram)


@pytest.mark.slow
def test_reference_budget():
    """One CPU development decision on an injected, action-conditioned object model."""
    import platform
    import resource
    import sys

    bank = dynamics(jnp.float32)
    bank = bank._replace(affine=bank.affine.at[:, 0, -1].set(jnp.array([.03, -.03], jnp.float32)))
    objects = jnp.array([[0., 0.], [.5, -.5]], jnp.float32)
    settings = SearchSettings()
    prior = initial_proposal(3)

    def object_step(model, scene, action, key):
        x = scene[:, 0]
        evidence = jnp.stack((x + action, -x - action, jnp.zeros_like(x)), axis=1)
        prediction = predict_objects(model, scene, jnp.ones(x.shape, dtype=bool), evidence, key)
        return prediction.state, (prediction.reward, prediction.novelty)

    def decide(model, scene, proposal, key):
        return plan(object_step, model, scene, proposal, key, settings=settings)

    args = bank, objects, prior, jax.random.key(171)
    before = perf_counter()
    compiled = jax.jit(decide).lower(*args).compile()
    compile_s = perf_counter() - before
    first_start = perf_counter()
    choice = jax.block_until_ready(compiled(*args))
    first_s = perf_counter() - first_start
    warm_start = perf_counter()
    repeated = jax.block_until_ready(compiled(*args))
    warm_s = perf_counter() - warm_start
    assert bool(choice.valid) and choice.policies.shape == (512, 32)
    np.testing.assert_array_equal(choice.reward, repeated.reward)
    np.testing.assert_array_equal(choice.novelty, repeated.novelty)
    assert int(choice.action) == int(choice.proposal.best[0])
    assert bool(jnp.isfinite(choice.reward).all() & jnp.isfinite(choice.novelty).all())
    buffers = compiled.memory_analysis()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(dict(hardware=platform.machine(), device=str(jax.devices()[0]), dtype="float32",
               real_transitions=0, imagined_transitions=32*512*3, slots=2, components=3,
               compile_s=compile_s, first_blocked_s=first_s, warm_blocked_s=warm_s,
               peak_process_rss_bytes=peak_rss if sys.platform == "darwin" else peak_rss*1024,
               xla_temp_bytes=buffers.temp_size_in_bytes, output_bytes=buffers.output_size_in_bytes,
               argument_bytes=buffers.argument_size_in_bytes))


def test_restart_after_one_real_move(tmp_path):
    settings = SearchSettings(candidates=12)
    initial = initial_proposal(2, horizon=3, dtype=jnp.float32)
    snapshot = jnp.array([1., -1.], jnp.float32)
    real_state = jnp.float32(2)
    key, next_key = jax.random.split(jax.random.key(39))
    decide = jax.jit(lambda bank, observed, proposal, k: plan(
        choice_step, bank, observed, proposal, k, settings=settings))
    first = decide(snapshot, real_state, initial, key)
    # Apply only the selected first move to the real fixture and receive a new datum.
    real_state = real_state - 3 * (first.action == 0)
    expected = decide(snapshot, real_state, first.proposal, next_key)
    path = tmp_path / "planner.npz"
    np.savez(path, snapshot=snapshot, state=real_state, probs=first.proposal.probabilities,
             best=first.proposal.best, next_key=jax.random.key_data(next_key))
    with np.load(path) as saved:
        restored = decide(saved["snapshot"], saved["state"],
                          first.proposal._replace(probabilities=saved["probs"], best=saved["best"]),
                          jax.random.wrap_key_data(saved["next_key"]))
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(left, right)
    assert first.action == 0 and restored.action == 1
