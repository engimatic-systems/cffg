"""The single-layer AXIOM learner driven by actual observations and applied actions."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg.component_bank_device import DeviceBank
from cffg.experiments.axiom_banks import (
    BankSettings,
    identity_labels,
    identity_records,
    initial_banks,
)
from cffg.experiments.axiom_interaction import (
    InteractionConfig,
    interaction_batch,
    learn_interactions,
)
from cffg.experiments.axiom_mixture import learn_frame
from cffg.experiments.axiom_motion import MotionBank, acquire_motion, initial_motion
from cffg.experiments.axiom_slots import (
    SpatialModel,
    frame_tokens,
    initial_model,
    initialize_frame,
    observe_frame,
    slot_descriptors,
)
from cffg.experiments.axiom_spatial import Positions
from cffg.experiments.axiom_tracking import (
    DecodedSlots,
    TrackCarry,
    TrackingSettings,
    initial_tracks,
    track_frame,
)


@dataclass(frozen=True)
class SceneSettings:
    actions: int
    slots: int = 32
    banks: BankSettings = BankSettings()
    tracking: TrackingSettings = TrackingSettings()
    train_radius: float = 0.075
    predict_radius: float = 0.075

    def __post_init__(self):
        if type(self.slots) is not int or self.slots < 1:
            raise ValueError("at least one spatial slot required")
        self.interaction_config()

    def interaction_config(self):
        return InteractionConfig(
            self.actions,
            self.banks.identities,
            self.banks.motions,
            self.banks.fixed_radius,
            self.train_radius,
            self.predict_radius,
        )


class Scene(NamedTuple):
    spatial: SpatialModel
    positions: Positions
    tracks: TrackCarry
    motion: MotionBank
    identity: DeviceBank
    interaction: DeviceBank
    absent_distance: jax.Array
    tick: jax.Array
    can_reset: jax.Array


class ObservationResult(NamedTuple):
    state: Scene
    status: jax.Array
    spatial_births: jax.Array
    identity_eligible: jax.Array
    identity_born: jax.Array
    identity_writes: jax.Array
    dynamics_eligible: jax.Array
    motion_class: jax.Array
    motion_born: jax.Array
    interaction_born: jax.Array
    interaction_writes: jax.Array
    valid: jax.Array


def _decode(model, out):
    return DecodedSlots.from_readout(*slot_descriptors(model, out.positions), out)


class SceneKeys(NamedTuple):
    appearance: jax.Array
    identity: jax.Array
    interaction: jax.Array
    geometry: jax.Array


def scene_keys(key):
    """Derive the layer creation keys from the runner root seed."""
    _, model = jax.random.split(key)
    model, appearance = jax.random.split(model)
    model, _ = jax.random.split(model)  # Seeded affine motions are deterministic.
    model, identity = jax.random.split(model)
    _, interaction = jax.random.split(model)
    return SceneKeys(appearance, identity, interaction, jax.random.fold_in(key, 0xCFF6))


def initialize_scene(rgb, key, settings: SceneSettings, *, dtype=jnp.float32):
    """Build the model from its first real frame; tick zero is the next applied action."""
    keys = scene_keys(key)
    identity, interaction = initial_banks(settings.actions, settings=settings.banks, dtype=dtype)
    model = initial_model(keys.appearance, slots=settings.slots, dtype=dtype)
    model, spatial = initialize_frame(model, frame_tokens(rgb, dtype))
    track = initial_tracks(_decode(model, spatial), settings=settings.tracking)
    motion = initial_motion(
        capacity=settings.banks.motions, dtype=dtype, settings=settings.tracking.motion
    )
    absent = 1.2 + 0.01 * jax.random.uniform(keys.geometry, (2,), dtype)
    return Scene(
        model,
        spatial.positions,
        track.state,
        motion,
        identity,
        interaction,
        absent,
        jnp.int32(0),
        jnp.bool_(False),
    ), spatial.valid & track.valid


def observe_scene(
    state: Scene, rgb, action, reward, tick, *, settings: SceneSettings, update=True, reset=False
):
    """Consume one observed transition. Invalid or out-of-order inputs leave the state intact.

    A reset observation follows the most recent real transition. Pass reset=True,
    update=False, action zero and reward zero to preserve the real action count.
    """
    pixels = frame_tokens(rgb, state.positions.mean.dtype)
    visible = observe_frame(state.spatial, pixels, state.positions)
    track = track_frame(
        state.tracks,
        _decode(visible.model, visible.readout),
        update=update,
        settings=settings.tracking,
    )
    previous = state.tracks.scene
    motion = acquire_motion(
        state.motion,
        previous[:, :6],
        track.state.scene[:, :6],
        action,
        track.dynamics_eligible,
        settings=settings.tracking.motion,
    )
    points = identity_records(previous, color_only=settings.banks.color_only)
    identity = learn_frame(state.identity, points, track.identity_eligible, threshold=-500.0)
    labels, labeled = jax.lax.cond(
        jnp.any(track.dynamics_eligible),
        lambda: identity_labels(identity.bank.posterior, points),
        lambda: (jnp.zeros(len(previous), jnp.int32), jnp.bool_(True)),
    )
    context = interaction_batch(
        previous,
        labels,
        track.state.tracked,
        action,
        state.absent_distance,
        config=settings.interaction_config(),
        switches=motion.switch,
        reward=reward,
    )
    relation = learn_interactions(state.interaction, context, track.dynamics_eligible)
    tick = jnp.asarray(tick)
    if tick.shape != () or not jnp.issubdtype(tick.dtype, jnp.integer):
        raise ValueError("tick must be a scalar integer")
    reset = jnp.asarray(reset)
    if reset.shape or reset.dtype != jnp.bool_:
        raise ValueError("reset must be a scalar boolean")
    clock_ok = ~reset | (state.can_reset & ~jnp.asarray(update) & (action == 0) & (reward == 0))
    valid = (
        clock_ok
        & (state.tick == tick)
        & (state.tick < jnp.iinfo(state.tick.dtype).max)
        & visible.readout.valid
        & track.valid
        & motion.valid
        & identity.valid
        & labeled
        & context.valid
        & relation.valid
    )
    candidate = Scene(
        visible.model,
        visible.readout.positions,
        track.state,
        motion.bank,
        identity.bank,
        relation.bank,
        state.absent_distance,
        state.tick + (~reset).astype(state.tick.dtype),
        ~reset,
    )
    committed = jax.tree.map(lambda new, old: jnp.where(valid, new, old), candidate, state)
    return ObservationResult(
        committed,
        visible.status,
        jnp.where(valid, visible.births, 0),
        valid & track.identity_eligible,
        jnp.where(valid, identity.born, -1),
        valid & identity.written,
        valid & track.dynamics_eligible,
        jnp.where(valid, motion.switch, -1),
        jnp.where(valid, motion.born, -1),
        jnp.where(valid, relation.born, -1),
        valid & relation.written,
        valid,
    )
