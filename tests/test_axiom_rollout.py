"""Full-bank oracles for the learned scene sampler and a bounded capacity probe."""

from functools import partial
import resource
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import component_bank_device
from cffg.component_bank import ComponentBank
from cffg.experiments.axiom_interaction import (
    InteractionConfig, interaction_batch, learn_interactions, score_context,
)
from cffg.experiments.axiom_motion import acquire_motion, initial_motion, observed_state
from cffg.experiments.axiom_planning import SearchSettings, initial_proposal, plan
from cffg.experiments.axiom_prediction import ObjectDynamics, predict_objects
from cffg.experiments.axiom_rollout import (
    WorldStatus, _context_scores, freeze_world, predict_scene, rollout_step,
)
from cffg.mixed_state import MixedBank


def mixture(k, active, config, dtype, *, prior=False):
    d = 5 if config.fixed_radius else 7
    ids = jnp.asarray(active, jnp.int32)
    ones = jnp.ones(k, dtype)
    widths = (config.identities + 1, config.identities + 1, 2, config.actions, 3, config.motions)
    heads = [jnp.full((k, width), 1. if i == 4 else 1e-4, dtype)
             for i, width in enumerate(widths)]
    if not prior:
        # Learned contexts and outcomes have categorical evidence. Components
        # at different physical indices need not have equal counts.
        alt = jnp.arange(len(active))
        for i, head in enumerate(heads):
            heads[i] = head.at[ids, alt % widths[i]].add(1 + alt)
    return MixedBank(jnp.zeros((k, d), dtype), ones, ones[:, None, None] * jnp.eye(d, dtype=dtype),
                     (d + 3) * ones, (.1 * ones).at[ids].add(jnp.arange(len(active)) + 1),
                     tuple(heads), jnp.zeros(k, bool).at[ids].set(True))


def inputs(slots, config, dtype, tracked):
    x = jnp.zeros((slots, 11), dtype)
    x = x.at[:, 0].set(jnp.linspace(-.8, .8, slots, dtype=dtype)).at[:, 3].set(.01)
    x = x.at[:, 6:8].set(.03)
    return (x, jnp.arange(slots) % config.identities,
            jnp.zeros(slots, bool).at[jnp.asarray(tracked, jnp.int32)].set(True),
            jnp.asarray([1.201, 1.208], dtype))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_sparse_snapshot_matches_full_component_and_slot_axes(dtype):
    config = InteractionConfig(actions=3, identities=3, motions=5)
    model = mixture(10, [1, 4, 8], config, dtype)
    # Resolving a concentrated posterior can change its released scoring scale.
    model = model._replace(psi=model.psi.at[:, 0, 0].set(2e-4))
    scene, ids, tracked, absent = inputs(7, config, dtype, [1, 3, 6])
    motion = initial_motion(capacity=5, dtype=dtype)
    snap = freeze_world(model, motion, ids, tracked, absent, config=config)
    np.testing.assert_array_equal(snap.component_rows, [1, 4, 8, -1])
    assert snap.gaussian.mu.shape[0] == snap.focused_slots.size == 4
    expected_model = ObjectDynamics(model.alpha, model.active,
                                   model.categories[-1] / model.categories[-1].sum(axis=1, keepdims=True),
                                   model.categories[-2] / model.categories[-2].sum(axis=1, keepdims=True),
                                   motion.affine, jnp.zeros(6, dtype))
    key = jax.random.key(31)
    evidence = interaction_batch(scene, ids, tracked, 2, absent, config=config)
    scores, q, valid = score_context(model, evidence)
    assert valid
    focus = np.flatnonzero(tracked)
    live = np.flatnonzero(model.active)
    compact, relative = _context_scores(snap, jax.tree.map(lambda a: a[focus], evidence.records))
    np.testing.assert_allclose(compact[:, :len(live)], scores[np.ix_(focus, live)], rtol=3e-6)
    np.testing.assert_allclose(relative[:, :len(live)], q[np.ix_(focus, live)], rtol=3e-6)
    assert (relative[:, len(live):] == 0).all()
    reference = predict_objects(expected_model, scene, tracked, scores, key, responsibilities=q)
    pred = jax.jit(partial(predict_scene, config=config))(snap, scene, 2, key)
    for actual, expected in zip(pred, reference, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
    np.testing.assert_array_equal(pred.state[~tracked], scene[~tracked])
    assert float(snap.total_alpha) == pytest.approx(float(model.alpha.sum()))


def test_causal_acquired_motion_and_reward_change_imagined_action():
    cfg = InteractionConfig(actions=2, identities=3, motions=5, static_contacts=True)
    scene, ids, tracked, absent = inputs(3, cfg, jnp.float32, [2])
    scene = scene.at[1, :2].set(scene[2, :2])
    motion = initial_motion(capacity=5)
    raw = ComponentBank(mixture(4, [], cfg, jnp.float32, prior=True))
    bank = component_bank_device.from_host(raw)
    # One actual positive-reward move creates a new motion expert. The second
    # move gets a distinct interaction row for the other action and its loss.
    pos = scene[:, :2].at[2, 0].add(.04)
    next_state = observed_state(scene[:, :6], pos, jnp.ones(3, bool))
    observed = acquire_motion(motion, scene[:, :6], next_state, 0, tracked)
    assert observed.valid and observed.born[2] == 4
    for action, reward, switches in [(0, 1, observed.switch), (1, -1, jnp.zeros(3, jnp.int32))]:
        evidence = interaction_batch(scene, ids, tracked, action, absent, config=cfg,
                                     reward=reward, switches=switches)
        assert evidence.neighbor[2] == 1
        receipt = learn_interactions(bank, evidence, tracked, threshold=1000.)
        assert receipt.valid and receipt.written[2].any()
        bank = receipt.bank
    snap = freeze_world(bank.posterior, observed.bank, ids, tracked, absent, config=cfg)
    key = jax.random.key(2)
    win = predict_scene(snap, scene, 0, key, config=cfg)
    assert win.reward == 1 and win.component[2] == 0 and win.motion[2] == 4
    np.testing.assert_allclose(win.state[2, :6], next_state[2], rtol=2e-6, atol=1e-7)
    lose = predict_scene(snap, scene, 1, key, config=cfg)
    assert lose.reward < -.45 and lose.component[2] == 1 and lose.motion[2] == 0
    before = np.array(bank.posterior.alpha)
    def decide(b, x, k):
        return plan(partial(rollout_step, config=cfg), b, x, initial_proposal(2, horizon=2), k,
                    settings=SearchSettings(candidates=6, replicates=2))

    result = jax.jit(decide)(snap, scene, key)
    assert result.valid and result.action == 0
    np.testing.assert_array_equal(bank.posterior.alpha, before)


def test_empty_scene_missing_support_and_invalid_action():
    cfg = InteractionConfig(actions=2, identities=2, motions=4)
    x, ids, mask, absent = inputs(2, cfg, jnp.float32, [])
    empty = mixture(2, [], cfg, jnp.float32)
    motion = initial_motion(capacity=4)
    key = jax.random.key(17)
    def predict(tracked, action):
        snap = freeze_world(empty, motion, ids, tracked, absent, config=cfg)
        expected = WorldStatus.NO_TRACKED_INTERACTION_SUPPORT if tracked.any() else WorldStatus.READY
        assert snap.status == expected
        return predict_scene(snap, x, action, key, config=cfg)
    neutral = predict(mask, 0)
    assert neutral.reward == 0 and neutral.novelty == 0 and (neutral.component == -1).all()
    np.testing.assert_array_equal(neutral.state, x)
    for invalid in (predict(~mask, 0), predict(mask, 2)):
        assert jnp.isnan(invalid.reward) and jnp.isnan(invalid.novelty)
        np.testing.assert_array_equal(invalid.state, x)


@pytest.mark.slow
def test_sparse_reference_capacity_subset():
    """Float32, H=8, 16 policies x 3 draws; the release budget has 128x as many transitions."""
    cfg = InteractionConfig(actions=4, identities=32, motions=500)
    model = mixture(5000, [1, 23, 207, 419, 501, 700, 1900, 4057], cfg, jnp.float32)
    scene, ids, tracked, absent = inputs(32, cfg, jnp.float32, [1, 2, 3, 4, 18, 20])
    motion = initial_motion(capacity=500)
    prior = initial_proposal(4, horizon=8)
    fn = jax.jit(lambda s, x, p, k: plan(partial(rollout_step, config=cfg), s, x, p, k,
                                      settings=SearchSettings(candidates=16, replicates=3)))
    jax.block_until_ready((model, scene, motion, prior))
    def snapshot():
        return freeze_world(model, motion, ids, tracked, absent, config=cfg)
    start = perf_counter()
    snap = jax.block_until_ready(snapshot())
    first_prepare = perf_counter() - start
    start = perf_counter()
    snap = jax.block_until_ready(snapshot())
    warm_prepare = perf_counter() - start
    args = snap, scene, prior, jax.random.key(819)
    start = perf_counter()
    compiled = fn.lower(*args).compile()
    compile_time = perf_counter() - start
    first = jax.block_until_ready(compiled(*args))
    assert first.valid and np.isfinite(first.reward).all()
    times = []
    for _ in range(3):
        start = perf_counter()
        repeat = jax.block_until_ready(compiled(*args))
        times.append(perf_counter() - start)
        np.testing.assert_array_equal(first.total_return, repeat.total_return)
    mem = compiled.memory_analysis()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(dict(device=str(jax.devices()[0]), dtype="float32", physical_components=5000,
               occupied=8, switches=500, slots=32, tracked=6,
               horizon=8, policies=16, replicates=3, imagined_transitions=384,
               prepare_first_s=first_prepare, prepare_warm_s=warm_prepare,
               compile_s=compile_time, warm_s=times, xla_temp_bytes=mem.temp_size_in_bytes,
               executable_argument_bytes=mem.argument_size_in_bytes,
               peak_process_rss_bytes=rss if sys.platform == "darwin" else rss * 1024))
