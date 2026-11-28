"""Online scene learning preserves the applied action clock and model-specific priors."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from cffg.experiments.axiom_banks import (
    BankSettings,
    identity_records,
    initial_banks,
    seeded_dormant,
)
from cffg.experiments.axiom_observation import SceneSettings, initialize_scene, observe_scene


def same(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


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
