"""Forward simulation of fixed action sequences from a caller's model snapshot.

A step is a pure function of (snapshot, simulated state, action, key). It returns
(next simulated state, readout), both JAX array pytrees. The snapshot is shared
read-only across all steps. Callers select which per-step readouts to retain.
"""

import jax
import jax.numpy as jnp
from jax import lax, random


def trajectory_key(key, policy_index, replicate_index):
    """Address one trajectory independently of the number of policies or draws."""
    return random.fold_in(random.fold_in(key, policy_index), replicate_index)


def sample_trajectory(step, snapshot, initial_state, actions, key):
    """Return final state and time-indexed readouts for one action sequence."""
    snapshot = jax.tree.map(jnp.asarray, snapshot)
    actions = jnp.asarray(actions)
    if actions.ndim < 1 or actions.shape[0] == 0:
        raise ValueError("a trajectory needs at least one action")

    def advance(state, inputs):
        tick, action = inputs
        return step(snapshot, state, action, random.fold_in(key, tick))

    return lax.scan(advance, initial_state, (jnp.arange(actions.shape[0]), actions))


def sample_action_sequences(step, snapshot, initial_state, policies, key, *,
                            replicates=1, batch_size=None):
    """Simulate [policy, time, ...] actions. Result leaves start [policy, draw].

    Supply a Threefry key so vmapped draws agree with individually addressed
    scalar trajectories. Optional batch_size limits concurrent policies while
    preserving random streams. Call with a fresh key after each real observation.
    """
    policies = jnp.asarray(policies)
    if policies.ndim < 2 or min(policies.shape[:2]) == 0 or replicates < 1:
        raise ValueError("expected a nonempty policy and time axis and positive replicates")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if str(random.key_impl(key)) != "threefry2x32":
        raise ValueError("use Threefry keys for scalar and batched sampling parity")

    def simulate_policy(inputs):
        index, actions = inputs

        def simulate_draw(draw):
            stream = trajectory_key(key, index, draw)
            return sample_trajectory(step, snapshot, initial_state, actions, stream)

        return jax.vmap(simulate_draw)(jnp.arange(replicates))

    indexed = (jnp.arange(policies.shape[0]), policies)
    if batch_size is None:
        return jax.vmap(simulate_policy)(indexed)
    return lax.map(simulate_policy, indexed, batch_size=min(batch_size, policies.shape[0]))
