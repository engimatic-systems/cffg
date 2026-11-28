"""Online scene learning preserves the applied action clock and model-specific priors."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.experiments.axiom_banks import (
    BankSettings,
    identity_labels,
    identity_records,
    initial_banks,
    seeded_dormant,
)
from cffg.experiments.axiom_observation import SceneSettings, initialize_scene, observe_scene


def same(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("color_only", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_scene_identity_beliefs_from_saved_descriptors(tmp_path, color_only):
    cfg = BankSettings(identities=3, motions=4, interactions_per_motion=1, color_only=color_only)
    scene, ok = initialize_scene(jnp.zeros((2, 2, 3), jnp.uint8), jax.random.key(5),
                                 SceneSettings(2, slots=3, banks=cfg), dtype=jnp.float32)
    assert ok
    weights, valid = scene.identity_beliefs()
    assert valid
    np.testing.assert_array_equal(weights, np.full((3, 3), 1 / 3, np.float32))

    # Red channel scaled by 100. Row 2 is currently hidden and retains its last color.
    red = np.array([0, .5, 1.], np.float32)
    width = np.array([.3, .8, .2], np.float32)
    rows = scene.tracks.scene.at[:, -5:].set(0).at[:, -3].set(red / 100).at[:, -5].set(width)
    tracks = scene.tracks._replace(scene=rows, tracked=jnp.array([False, True, False]))
    bank = scene.identity.posterior
    k, d = bank.mu.shape
    centers = bank.mu.at[:].set(0).at[1, -3].set(1)
    if not color_only:
        centers = centers.at[1, 0].set(.75)
    bank = bank._replace(mu=centers, kappa=jnp.ones_like(bank.kappa),
                         psi=jnp.tile(jnp.eye(d, dtype=centers.dtype), (k, 1, 1)),
                         active=jnp.array([True, True, False]), alpha=bank.alpha.at[1].set(9))
    scene = scene._replace(tracks=tracks, identity=scene.identity._replace(posterior=bank))
    # Same unit scale, degrees and kappa: the likelihood odds depend only on distance.
    distance_difference = red**2 - (red - 1)**2
    if not color_only:
        distance_difference += width**2 - (width - .75)**2
    right = 1 / (1 + np.exp(-.5 * float(bank.nu[0]) * distance_difference))
    expected = np.column_stack((1 - right, right, np.zeros(3)))
    read = jax.jit(lambda s: s.identity_beliefs())
    weights, valid = read(scene)
    assert valid
    np.testing.assert_allclose(weights, expected, rtol=2e-6, atol=3e-7)
    labels, valid = identity_labels(bank, identity_records(rows, color_only=color_only))
    assert valid
    np.testing.assert_array_equal(labels, np.argmax(expected, axis=1))

    order = jnp.array([0, 2, 1])
    permuted = jax.tree.map(lambda a: a[order] if a.ndim else a, tracks)
    swapped, valid = read(scene._replace(tracks=permuted))
    assert valid
    np.testing.assert_array_equal(swapped, weights[order])
    leaves, structure = jax.tree.flatten(scene)
    path = tmp_path / "scene.npz"
    np.savez(path, *leaves)
    with np.load(path, allow_pickle=False) as saved:
        resumed = structure.unflatten([jnp.asarray(saved[f"arr_{i}"]) for i in range(len(leaves))])
    same(read(resumed), read(scene))
    broken = tracks._replace(scene=rows.at[1, -3].set(jnp.nan))
    for active in (bank.active, jnp.zeros_like(bank.active)):
        invalid = scene._replace(tracks=broken, identity=scene.identity._replace(
            posterior=bank._replace(active=active)))
        failed, valid = read(invalid)
        assert not valid
        np.testing.assert_array_equal(failed, 0)


def test_priors_seeded_dormant_rows_and_observed_identity_features():
    settings = BankSettings(identities=3, motions=4, interactions_per_motion=2, color_precision=2.0)
    i, r = initial_banks(3, settings=settings, dtype=jnp.float32)
    assert not i.posterior.active.any() and not r.posterior.active.any()
    np.testing.assert_allclose(i.prior.psi[0], np.diag([4.0, 4.0, 2.0, 2.0, 2.0]))
    np.testing.assert_allclose(r.prior.psi[1], np.eye(7) / 5625)
    np.testing.assert_allclose(r.prior.nu, 15)
    assert [x.shape for x in r.prior.categories] == [(8, n) for n in [4, 4, 2, 3, 3, 4]]
    for head, alpha in zip(r.prior.categories, (1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1e-4), strict=True):
        np.testing.assert_allclose(head, alpha)
    dormant = seeded_dormant(r.prior, jax.random.key(33))
    same(dormant, seeded_dormant(r.prior, jax.random.key(33)))
    assert (dormant.mu != r.prior.mu).any()
    assert (dormant.categories[0] < r.prior.categories[0]).any()
    np.testing.assert_array_equal(r.prior.mu, 0)
    x = jnp.linspace(-1, 1, 33).reshape(3, 11)
    np.testing.assert_allclose(
        identity_records(x).mean, x[:, -5:] * jnp.array([1, 1, 100, 100, 100])
    )


def test_observed_scene_is_atomic_and_skips_fictitious_transitions():
    settings = SceneSettings(
        3, slots=3, banks=BankSettings(identities=4, motions=4, interactions_per_motion=2)
    )
    rgb = jnp.zeros((8, 8, 3), jnp.uint8).at[1:3, 2:4, 0].set(200)
    initial, ok = initialize_scene(rgb, jax.random.key(5), settings, dtype=jnp.float32)
    assert ok
    step = jax.jit(partial(observe_scene, settings=settings))
    next_rgb = jnp.roll(rgb, 1, axis=0)
    first = step(initial, next_rgb, jnp.int32(2), jnp.float32(-1), jnp.int32(0))
    assert first.valid and int(first.state.tick) == 1
    hunt = first.state
    for tick, reward in [(1, 2), (2, -2)]:
        receipt = step(hunt, rgb, jnp.int32(0), jnp.float32(reward), jnp.int32(tick))
        assert receipt.valid and int(receipt.state.tick) == tick + 1
        hunt = receipt.state
    for action, reward, tick in [(2, -1, 0), (3, 0, 1), (0, 1.5, 1), (0, np.nan, 1), (0, np.inf, 1)]:
        failed = step(first.state, rgb, jnp.int32(action), jnp.float32(reward), jnp.int32(tick))
        assert not failed.valid
        same(failed.state, first.state)
        assert not failed.identity_writes.any() and not failed.interaction_writes.any()
    reset = step(
        first.state, rgb, jnp.int32(0), jnp.float32(0), jnp.int32(1), update=False, reset=True
    )
    assert reset.valid and int(reset.state.tick) == 1
    duplicate_reset = step(
        reset.state, rgb, jnp.int32(0), jnp.float32(0), jnp.int32(1), update=False, reset=True
    )
    assert not duplicate_reset.valid
    assert step(reset.state, rgb, jnp.int32(0), jnp.float32(0), jnp.int32(1)).valid
    for field in ("motion", "identity", "interaction"):
        same(getattr(first.state, field), getattr(reset.state, field))
