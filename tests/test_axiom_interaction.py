"""Independent contacts and categorical observations on supplied slot scenes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.experiments.axiom_interaction import InteractionConfig, interaction_batch


def scene(dtype):
    # [x,y,unused,vx,vy,vu,width,height], row 0 background.
    return jnp.array([[0., 0., 0., 0., 0., 0., .2, .2],
                      [.2, .1, 0., .01, -.02, 0., .06, .08],
                      [.26, .15, 0., 0., 0., 0., .06, .08],
                      [.28, .1, .05, -.08, 0., 0., .06, .08]], dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_default_scene_actual_and_contextual_records(dtype):
    x = scene(dtype)
    labels = jnp.array([0, 1, 1, 2])
    tracks = jnp.array([False, True, False, True])
    cfg = InteractionConfig(actions=3, identities=3, motions=5, static_contacts=True)
    args = x, labels, tracks, 2, jnp.array([1.201, 1.204], dtype)
    switches = jnp.array([0, 3, 1, 2])
    train = interaction_batch(*args, config=cfg, switches=switches, reward=-1)
    pred = jax.jit(lambda *a: interaction_batch(*a, config=cfg))(*args)
    assert train.valid and pred.valid
    # Closer static row 2 yields to tracked row 3. Default rectangle path
    # permits a tracked neighbor with nonzero unused counter.
    assert train.neighbor[1] == pred.neighbor[1] == 3
    np.testing.assert_allclose(train.records.mean[1], [.2,.1,0,.1,-.2,-.08,0], atol=2e-8)
    np.testing.assert_array_equal([h[1] for h in train.records.categories], [1,2,1,2,0,3])
    np.testing.assert_array_equal([h[1] for h in pred.records.categories], [1,2,1,2,-1,-1])
    # When no dynamic neighbor remains, the nearest eligible static is same identity.
    only = tracks.at[3].set(False)
    train = interaction_batch(x, labels, only, 2, args[-1], config=cfg, switches=switches, reward=1)
    pred = interaction_batch(x, labels, only, 2, args[-1], config=cfg)
    assert train.neighbor[1] == pred.neighbor[1] == 2
    assert train.records.categories[1][1] == 3 and pred.records.categories[1][1] == 1


def test_fixed_radius_forward_contact_and_real_default_motion_veto():
    # Centers separated by .3; radii .04 each. Current ellipses do not intersect,
    # but the projected centers coincide. Static row 0 is excluded.
    x = jnp.array([[.4,0,0,0,0,0,.2,.2], [.2,0,0,.2,0,0,.04,.04],
                   [.5,0,0,-.1,0,0,.04,.04]])
    cfg = InteractionConfig(actions=2, fixed_radius=True, train_radius=1., predict_radius=1.)
    args = x, jnp.arange(3), jnp.array([False,True,True]), 0, jnp.zeros(2)
    pred = interaction_batch(*args, config=cfg)
    train = interaction_batch(*args, config=cfg, reward=0, switches=jnp.array([0,0,1]))
    assert pred.valid and train.valid and pred.records.mean.shape == (3,5)
    np.testing.assert_array_equal(pred.neighbor[1:], [2,1])
    np.testing.assert_array_equal(train.neighbor[1:], [-1,1])
    np.testing.assert_array_equal(train.records.mean[1], [.2,0,0,2,0])
    # The tuned path excludes invisible candidates.
    absent = interaction_batch(x.at[2,2].set(.05), *args[1:], config=cfg)
    assert absent.neighbor[1] == -1


def test_recorded_absent_distance_and_invalid_categories():
    x = scene(jnp.float32)[1:2]
    cfg = InteractionConfig(actions=2)
    args = x, jnp.array([1]), jnp.array([True]), 0, jnp.array([1.203,1.207], x.dtype)
    result = interaction_batch(*args, config=cfg)
    assert result.valid and result.neighbor[0] == -1
    np.testing.assert_array_equal(result.records.mean[0, -2:], args[-1])
    assert not interaction_batch(*args, config=cfg, reward=.5, switches=jnp.array([1])).valid
    with pytest.raises(ValueError, match="both"):
        interaction_batch(*args, config=cfg, reward=1)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_reward_bins_and_large_native_integers(dtype):
    cfg = InteractionConfig(actions=2)
    x = scene(dtype)[1:2]
    def encode(reward):
        evidence = interaction_batch(x, jnp.array([1]), jnp.array([True]), 0, jnp.ones(2, dtype),
                                     config=cfg, reward=reward, switches=jnp.array([1]))
        return evidence.records.categories[-2][0], evidence.valid

    bins, valid = jax.jit(jax.vmap(encode))(jnp.array([-512, -2, -1, 0, 1, 2, 512], dtype))
    assert valid.all()
    np.testing.assert_array_equal(bins, [-1, -1, 0, 1, 2, -1, -1])
    # Narrowing this observed reward to int32 first would turn it into zero.
    label, ok = encode(jnp.int64(2**32))
    assert ok and label == -1
