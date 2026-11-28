"""Independent finite-world and addressing oracles for sampled open-loop planning."""

import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.rollout_planning import sample_action_sequences, sample_trajectory, trajectory_key


def flip(chance, state, action, key):
    """Toggle a binary state with the action's probability."""
    new = jnp.logical_xor(state, jax.random.bernoulli(key, chance[action]))
    return new, new


def test_two_step_distribution_and_key_addressing():
    p = jnp.array([0.25, 0.75])
    actions = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 0]])
    root = jax.random.key(19)
    count = 4096
    result = sample_action_sequences(flip, p, False, actions, root, replicates=count)
    # Exactly one flip in two independent trials.
    np.testing.assert_allclose(result[0].mean(axis=1),
                               [.375, .625, .625, .375, .625], atol=.035)
    assert not np.array_equal(result[0][2], result[0][4])
    for i, r in [(0, 0), (1, 53), (4, 4021)]:
        scalar = sample_trajectory(flip, p, False, actions[i], trajectory_key(root, i, r))
        for batch, single in zip(result, scalar, strict=True):
            np.testing.assert_array_equal(batch[i, r], single)
    np.testing.assert_array_equal(p, [.25, .75])


def test_resume_and_chunking_preserve_each_sample():
    root = jax.random.key(7)
    actions = np.arange(15).reshape(5, 3) % 2
    probabilities = np.array([.125, .5])
    expected = sample_action_sequences(flip, probabilities, True, actions, root, replicates=5)
    checkpoint = io.BytesIO()
    np.savez(checkpoint, actions=actions, p=probabilities, state=True,
             key=jax.random.key_data(root))
    checkpoint.seek(0)
    with np.load(checkpoint) as data:
        resumed = jax.jit(lambda p, s, a, k: sample_action_sequences(
            flip, p, s, a, k, replicates=5, batch_size=2
        ))(data["p"], data["state"], data["actions"], jax.random.wrap_key_data(data["key"]))
    for actual, wanted in zip(resumed, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)


def test_sampler_rejects_empty_time_and_unsupported_key():
    with pytest.raises(ValueError, match="nonempty"):
        sample_action_sequences(flip, [.25], True, jnp.empty((1, 0)), jax.random.key(1))
    with pytest.raises(ValueError, match="Threefry"):
        sample_action_sequences(flip, [.25], True, jnp.zeros((1, 1)), jax.random.key(1, impl="rbg"))
