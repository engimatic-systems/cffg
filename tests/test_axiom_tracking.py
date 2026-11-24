"""Independent slot history checks using injected spatial posteriors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.experiments.axiom_interaction import InteractionConfig, interaction_batch
from cffg.experiments.axiom_motion import acquire_motion, initial_motion
from cffg.experiments.axiom_spatial import Positions, SpatialReadout
from cffg.experiments.axiom_tracking import (
    DecodedSlots,
    TrackCarry,
    initial_tracks,
    track_frame,
)


def frame(xy, occupied, *, dtype=jnp.float64, mass=None):
    k = len(occupied)
    xy = jnp.asarray(xy, dtype)
    color = jnp.asarray([[i / 10, -0.2, 0.25] for i in range(k)], dtype)
    mean = jnp.column_stack((xy, color))
    cov = jnp.broadcast_to(jnp.diag(jnp.asarray([0.01, 0.04, 0.03, 0.02, 0.04], dtype)), (k, 5, 5))
    if mass is None:
        mass = [20] + [1] * (k - 1)
    return DecodedSlots(mean, cov, jnp.asarray(occupied, bool), jnp.array(mass, dtype), jnp.array(True))


def same(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_masks_motion_and_separate_evidence_clocks(dtype):
    start = frame([[0, 0], [0, 0], [0, 0]], [1, 1, 1], dtype=dtype)
    state = initial_tracks(start).state
    np.testing.assert_allclose(state.use, 0.01)
    advance = jax.jit(track_frame)
    # Slot 1 is static; slot 2 first teleports, then has small normal motion.
    shifted = start._replace(mean=start.mean.at[2, 0].set(0.4))
    first = advance(state, shifted)
    assert bool(first.valid)
    np.testing.assert_allclose(first.state.moving, [0, 0, 0.004], rtol=2e-6)
    np.testing.assert_array_equal(first.state.scene[:, 3:6], 0)
    np.testing.assert_array_equal(first.identity_eligible, [False, True, True])
    np.testing.assert_array_equal(first.dynamics_eligible, False)
    visible = shifted._replace(mean=shifted.mean.at[2, 0].add(0.05))
    second = advance(first.state, visible)
    np.testing.assert_allclose(second.state.scene[2, 3], 0.05, rtol=2e-6)
    np.testing.assert_array_equal(second.dynamics_eligible, [False, False, True])

    # A fresh measurement after one absent frame cannot train identity on a missing preceding observation.
    missing = visible._replace(
        occupied=visible.occupied.at[2].set(False),
        mean=visible.mean.at[2].set(jnp.nan),
        covariance=visible.covariance.at[2].set(jnp.nan),
    )
    third = advance(second.state, missing)
    assert bool(third.valid)
    np.testing.assert_allclose(third.state.scene[2, 2], 0.05)
    np.testing.assert_array_equal(third.state.scene[2, 6:], second.state.scene[2, 6:])
    np.testing.assert_array_equal(third.state.covariance[2], second.state.covariance[2])
    assert third.identity_eligible[2] and third.dynamics_eligible[2]
    fourth = advance(third.state, visible)
    assert not fourth.identity_eligible[2] and fourth.dynamics_eligible[2]
    np.testing.assert_allclose(fourth.state.scene[2, 2], 0)
    reset = advance(third.state, visible, update=False)
    same(reset.state, fourth.state)
    np.testing.assert_array_equal(reset.identity_eligible, False)
    np.testing.assert_array_equal(reset.dynamics_eligible, False)

    # Following the same scene, feed the two masks and the actual command into the existing consumers.
    bank = initial_motion(capacity=12, dtype=dtype)
    observed = acquire_motion(
        bank, first.state.scene[:, :6], second.state.scene[:, :6], 1, second.dynamics_eligible
    )
    out = interaction_batch(
        first.state.scene,
        jnp.arange(3),
        second.state.tracked,
        1,
        jnp.array([1.2, 1.2], dtype),
        config=InteractionConfig(actions=3, motions=12),
        switches=observed.switch,
        reward=-1,
    )
    assert bool(out.valid)
    np.testing.assert_allclose(out.records.mean[:, :2], first.state.scene[:, :2], rtol=2e-6)


def test_reset_restore_permutation_and_slot_reuse(tmp_path):
    # The background stays row zero. Rows 1 and 2 exchange labels with their entire histories.
    inputs = frame([[0, 0], [0.2, -0.1], [0.5, 0.4]], [1, 1, 1])
    previous = initial_tracks(inputs).state
    absent = inputs._replace(occupied=inputs.occupied.at[1].set(False))
    for _ in range(11):
        previous = track_frame(previous, absent).state
    assert previous.scene[1, 2] > 0.5 and not previous.tracked[1]
    new = inputs._replace(mean=inputs.mean.at[1].set(jnp.array([-0.8, 0.3, -0.4, 1, 0.5])))
    order = jnp.array([0, 2, 1])

    def permute(record):
        return jax.tree.map(lambda a: a[order] if a.ndim else a, record)

    moved = track_frame(previous, new)
    same(permute(moved), track_frame(permute(previous), permute(new)))
    np.testing.assert_array_equal(moved.state.scene[1, -3:], new.mean[1, -3:])
    assert moved.state.moving[1] > 0 and not moved.identity_eligible[1]
    archive = tmp_path / "tracks.npz"
    np.savez(archive, **previous._asdict())
    with np.load(archive, allow_pickle=False) as data:
        loaded = TrackCarry(**{name: jnp.asarray(data[name]) for name in TrackCarry._fields})
    same(track_frame(loaded, new), moved)


def test_reject_invalid_frame_and_preserve_state():
    first = frame([[0, 0], [0, 0]], [1, 1])
    prior = initial_tracks(first).state
    for bad in (
        first._replace(mass=jnp.array([1.0, -1.0])),
        first._replace(occupied=jnp.array([False, False])),
        first._replace(mass=jnp.array([1.0, 0.0])),
        first._replace(valid=jnp.array(False)),
        first._replace(covariance=first.covariance.at[0, 0, 1].set(1.0)),
        first._replace(covariance=first.covariance.at[0, 0, 0].set(-0.01)),
        first._replace(mean=first.mean.at[1, 0].set(jnp.nan)),
    ):
        result = track_frame(prior, bad)
        assert not result.valid and not initial_tracks(bad).state.initialized
        same(result.state, prior)
        np.testing.assert_array_equal(result.identity_eligible, False)
        np.testing.assert_array_equal(result.dynamics_eligible, False)
    with pytest.raises((TypeError, ValueError)):
        track_frame(prior, first._replace(occupied=jnp.array([1, 1])))
    with pytest.raises(ValueError):
        track_frame(prior, first._replace(mass=jnp.array([1.0])))
    assert not track_frame(initial_tracks(first._replace(valid=False)).state, first).valid


def test_soft_mass_vs_hard_occupancy_and_large_bank():
    k = 128
    first = frame([[0, 0]] * k, [1] * k)
    responsibility = jnp.full((2, k), 1 / k, jnp.float64)
    positions = Positions(first.mean[:, :2], first.covariance[:, :2, :2])
    readout = SpatialReadout(
        positions, responsibility, jnp.zeros(2), jnp.zeros(k, int).at[0].set(2), jnp.array(True)
    )
    decoded = DecodedSlots.from_readout(first.mean, first.covariance, readout)
    np.testing.assert_array_equal(decoded.occupied, np.arange(k) == 0)
    np.testing.assert_array_equal(decoded.mass, np.full(k, 2 / k))
    initial = initial_tracks(decoded)
    stepped = jax.jit(track_frame)(initial.state, decoded)
    assert stepped.valid and not stepped.state.foreground.any() and not stepped.identity_eligible.any()
    rejected = DecodedSlots.from_readout(first.mean, first.covariance, readout, accepted=False)
    assert not track_frame(initial.state, rejected).valid
