"""Analytical tMM fixtures for the default b99f625 velocity/counter model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.experiments.axiom_motion import (
    MotionBank, MotionSettings, MotionStatus as S, acquire_motion, initial_motion,
    motion_step, observed_state,
)


def same_bank(left, right):
    for a, b in zip(left, right, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_seed_predictions_and_mask(dtype):
    bank = initial_motion(capacity=500, dtype=dtype)
    np.testing.assert_array_equal(bank.occupied, np.arange(500) < 4)
    state = jnp.tile(jnp.array([.1, -.2, .2, .05, -.1, .3], dtype), (6, 1))
    actual = jax.jit(motion_step)(bank, state, jnp.array([0, 1, 2, 3, 4, -1]))
    expected = [[.15, -.3, 0, .05, -.1, 0], [.1, -.2, .25, .05, -.1, 0],
                [.1, -.2, .05, 0, 0, 0], [.1, -.2, 0, 0, 0, 0]]
    np.testing.assert_allclose(actual[:4], expected, atol=5e-8)
    assert np.isnan(actual[4:]).all()
    np.testing.assert_array_equal(motion_step(bank, state[0], 0), actual[0])


def test_slot_observation_visibility_velocity_clip_and_teleport_boundary():
    cfg = MotionSettings(teleport_distance=.25, clip=1/1024)
    prev = jnp.zeros((4, 6)).at[:, 2].set(.1)
    xy = jnp.array([[1/2048, -1/1024], [.25, 0.], [jnp.nan, jnp.nan], [.15, .1]])
    actual = observed_state(prev, xy, jnp.array([True, True, False, True]), settings=cfg)
    np.testing.assert_allclose(actual, [
        [1/2048, -1/1024, 0, 0, -1/1024, 0], [.25, 0, 0, 0, 0, 0],
        [0, 0, .15, 0, 0, 0], [.15, .1, 0, .15, .1, 0]],
        atol=1e-14)
    again = observed_state(actual[2], xy[2], False, settings=cfg)
    np.testing.assert_allclose(again, [0, 0, .2, 0, 0, 0], atol=1e-14)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_ordered_acceleration_birth_reuse_and_action_handoff(dtype):
    bank = initial_motion(dtype=dtype)
    # Two tracked objects undergo the same acceleration at distinct positions.
    before = jnp.array([[0., 0, 0, .01, 0, 0],
                        [.25, .125, 0, .01, 0, 0],
                        [0, 0, 0, 0, 0, 0]], dtype)
    delta = jnp.array([.06, .0005], dtype)
    after = observed_state(before, before[:, :2] + delta, jnp.ones(3, bool))
    after = after.at[-1].set(jnp.nan)
    args = bank, before, after, 2, jnp.array([True, True, False])
    got = jax.jit(acquire_motion)(*args)
    eager = acquire_motion(*args)
    same_bank(got.bank, eager.bank)
    assert got.valid and got.action == 2
    np.testing.assert_array_equal(got.previous, before)
    np.testing.assert_array_equal(got.current, after)
    np.testing.assert_array_equal(got.status, [S.VELOCITY_BIRTH, S.MATCHED, S.SKIPPED])
    np.testing.assert_array_equal(got.born, [4, -1, -1])
    np.testing.assert_array_equal(got.switch, [4, 4, 0])
    np.testing.assert_array_equal(got.bank.affine[:4], bank.affine[:4])
    np.testing.assert_array_equal(got.bank.affine[5:], bank.affine[5:])
    # Pure affine oracle: p' = p+v+(.05,0), v' = v+(.05,0), unused columns zero.
    wanted = np.array([[1,0,0,1,0,0,.05], [0,1,0,0,1,0,0], [0]*7,
                       [0,0,0,1,0,0,.05], [0,0,0,0,1,0,0], [0]*7])
    np.testing.assert_allclose(got.bank.affine[4], wanted, atol=1e-8)
    np.testing.assert_allclose(got.log_likelihood[:2, 4], -3*np.log(4*np.pi)-.0005**2/4,
                               atol=1e-6, rtol=0)
    assert np.isneginf(got.log_likelihood[:2, 5:]).all()
    assert (got.log_likelihood[2] == 0).all()
    # Another real action has no influence on fitting the tMM.
    again = acquire_motion(got.bank, before, after, 0, args[-1])
    same_bank(again.bank, got.bank)
    np.testing.assert_array_equal(again.status, [S.MATCHED, S.MATCHED, S.SKIPPED])
    # Checkpoint row arrays carry the same rule identities to the next frame.
    restored = MotionBank(*(np.array(x) for x in got.bank))
    next_frame = jax.jit(acquire_motion)(restored, before, after, 0, args[-1])
    np.testing.assert_array_equal(next_frame.switch, again.switch)


def test_teleport_constant_map_exhaustion_empty_map_and_rescored_tie():
    cfg = MotionSettings(teleport_distance=.25)
    before = jnp.zeros((1, 6))
    bank = initial_motion(capacity=5, dtype=before.dtype, settings=cfg)
    def record(table, old, xy):
        current = observed_state(old, jnp.asarray([xy]), jnp.array([True]), settings=cfg)
        return acquire_motion(table, old, current, 1, jnp.array([True]), settings=cfg)
    warp = record(bank, before, [.75, .5])
    assert warp.valid and warp.status[0] == S.TELEPORT_BIRTH and warp.switch[0] == 4
    np.testing.assert_array_equal(warp.bank.affine[4, :, :-1], 0)
    np.testing.assert_array_equal(warp.bank.affine[4, :, -1], [.75, .5, 0, 0, 0, 0])
    np.testing.assert_array_equal(motion_step(warp.bank, jnp.ones(6), 4), warp.current[0])
    full = record(warp.bank, before, [-.5, -.5])
    assert full.valid and full.status[0] == S.FULL and full.born[0] == -1
    assert full.switch[0] == int(np.argmax(full.log_likelihood[0]))
    same_bank(full.bank, warp.bank)

    empty = record(bank, warp.current, [0., 0.])
    assert empty.valid and empty.status[0] == S.EMPTY_BIRTH and empty.born[0] == -1
    same_bank(empty.bank, bank)
    # At distance exactly .25, observation preparation zeros velocity while
    # acquisition chooses the velocity family. Its score ties old drift.
    edge = record(bank, before, [.25, 0.])
    assert edge.status[0] == S.VELOCITY_BIRTH and edge.born[0] == 4
    assert edge.log_likelihood[0, 0] == edge.log_likelihood[0, 4]
    assert edge.switch[0] == 0


def test_strict_gaussian_threshold_and_invalid_frame_rollback():
    cfg = MotionSettings(variance=1., logp_threshold=-1.)
    bank = initial_motion(capacity=6, dtype=jnp.float64)
    before = jnp.zeros((2, 6))
    tracked = jnp.ones(2, bool)
    boundary = before.at[:, 0].set(1.)
    hit = acquire_motion(bank, before, boundary, 1, tracked, settings=cfg)
    assert hit.valid and (hit.status == S.MATCHED).all()
    np.testing.assert_allclose(hit.log_likelihood[:, 0], -3*np.log(2*np.pi)-.5,
                               rtol=0, atol=2e-15)
    above = boundary.at[:, 0].add(.01)
    fit = acquire_motion(bank, before, above, 1, tracked, settings=cfg)
    np.testing.assert_array_equal(fit.status, [S.TELEPORT_BIRTH, S.MATCHED])
    broken = jax.jit(acquire_motion)(bank, before, above.at[1, 0].set(1e200), 1, tracked)
    assert not broken.valid and (broken.switch == -1).all() and (broken.born == -1).all()
    np.testing.assert_array_equal(broken.status, [S.ROLLED_BACK, S.INVALID])
    same_bank(broken.bank, bank)
    skipped = acquire_motion(bank, before, above * jnp.nan, 0, tracked, update=False)
    assert skipped.valid and (skipped.status == S.SKIPPED).all()
    same_bank(skipped.bank, bank)
    vacant = bank._replace(occupied=jnp.zeros_like(bank.occupied))
    no_expert = acquire_motion(vacant, before, before, 0, tracked)
    assert not no_expert.valid and (no_expert.status == S.INVALID).all()
    same_bank(no_expert.bank, vacant)
