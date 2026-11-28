"""Persistent untuned AXIOM actor using full-budget, read-only scene rollouts."""

from copy import copy
from dataclasses import dataclass
from functools import partial
import hashlib
import json
from time import perf_counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cffg.experiments.axiom_banks import identity_labels, identity_records, seeded_dormant
from cffg.experiments.axiom_evidence import initialized_view
from cffg.experiments.axiom_observation import SceneSettings, initialize_scene, observe_scene, scene_keys
from cffg.experiments.axiom_planning import SearchSettings, initial_proposal, plan
from cffg.experiments.axiom_rollout import WorldStatus, freeze_world, rollout_step

from cffg import axiom_checkpoint
from cffg.experiments import axiom_reduction_schedule as reduction_schedule
from cffg.experiments.axiom_cached_plan import cached_plan


@partial(jax.jit, static_argnames=("color_only",))
def _labels(bank, scene, *, color_only=False):
    return identity_labels(bank, identity_records(scene, color_only=color_only))


_context = jax.jit(initialized_view)
_observe = jax.jit(observe_scene, static_argnames=("settings",))
_virgin = jax.jit(seeded_dormant)


@partial(jax.jit, static_argnames=("config", "batch_size"))
def _plan(snapshot, scene, proposal, subkey, *, config, batch_size):
    return plan(partial(rollout_step, config=config), snapshot, scene, proposal, subkey,
                settings=SearchSettings(batch_size=batch_size))


@jax.jit
def _empty_plan(proposal, key):
    # With no tracked slots every sampled action sequence has identically
    # zero reward and novelty. Run the same proposal generator, ties and refit.
    def stationary(_, value, action, step_key):
        return value, (jnp.float32(0), jnp.float32(0))
    return plan(stationary, (), jnp.int32(0), proposal, key)


def _chunk(components, slots):
    # The largest intermediates carry policy, replicate, slot, component and
    # feature axes. Limit their product and round down for executable reuse.
    maximum = max(1, min(64, 65536 // (components * slots)))
    return 1 << (maximum.bit_length() - 1)


def _count(value):
    return int(np.asarray(value).sum())


def _plan_digest(tree):
    rows = []
    for path, value in jax.tree.flatten_with_path(tree)[0]:
        array = np.asarray(value)
        rows.append((jax.tree_util.keystr(path), array.dtype.str, array.shape,
                     hashlib.sha256(array.tobytes()).hexdigest()))
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


@dataclass(frozen=True)
class PlannedAction:
    """A pending action bound to the observation clock and planner key."""

    tick: int
    action: int
    key: tuple[int, ...]


class AxiomAgent:
    """One real game and seed. Repeated choose calls return the same pending command."""

    def __init__(self, initial_rgb: np.ndarray, game: str, actions: int, seed: int, *,
                 settings: SceneSettings | None = None, audit_plan: bool = False):
        rgb = np.asarray(initial_rgb)
        if rgb.shape != (210, 160, 3) or rgb.dtype != np.uint8:
            raise ValueError("initial frame must be a native Gameworld uint8 RGB")
        self.audit_plan = bool(audit_plan)
        self.game = str(game)
        self.actions = int(actions) if isinstance(actions, (int, np.integer)) else len(actions)
        self.seed = int(seed)
        self.initial_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
        self.settings = SceneSettings(self.actions) if settings is None else settings
        if self.settings.actions != self.actions:
            raise ValueError("scene settings and action alphabet differ")
        self.config = self.settings.interaction_config()
        root = jax.random.PRNGKey(self.seed)
        self.scene, ok = jax.block_until_ready(initialize_scene(rgb, root, self.settings))
        if not bool(ok):
            raise ArithmeticError("initial scene is invalid")
        streams = scene_keys(root)
        self.virgin_identity = _virgin(self.scene.identity.prior, streams.identity)
        self.virgin_interaction = _virgin(self.scene.interaction.prior, streams.interaction)
        self.proposal = initial_proposal(self.actions, horizon=32)
        self.key = jax.random.fold_in(root, 0x504C414E)
        self.reduction = reduction_schedule.start(self.scene.interaction, jax.random.fold_in(root, 0x424D52))
        self.pending = None
        self.last_decision = None
        jax.block_until_ready((self.virgin_identity, self.virgin_interaction))

    @property
    def tick(self):
        return int(self.scene.tick)

    def _pending_plan(self):
        if self.pending is None:
            return None
        words = np.asarray(jax.random.key_data(self.key)).ravel()
        return PlannedAction(self.tick, self.pending, tuple(map(int, words)))

    def propose(self) -> PlannedAction:
        """Repeated calls at the same pending boundary return the same decision."""
        self.choose()
        return self._pending_plan()

    def abandon(self, plan: PlannedAction) -> None:
        """Discard a pending command. The next search uses the advanced proposal and key."""
        if not isinstance(plan, PlannedAction) or plan != self._pending_plan():
            raise ValueError("stale or missing pending decision")
        self.pending = self.last_decision = None

    def observe_applied(self, plan: PlannedAction, rgb: np.ndarray, applied: int, reward: float,
                        *, reset_rgb: np.ndarray | None = None) -> dict:
        """Consume the actual feedback once and optionally install the next reset frame."""
        if not isinstance(plan, PlannedAction) or plan != self._pending_plan():
            raise ValueError("stale or missing pending decision")
        for frame in (rgb,) if reset_rgb is None else (rgb, reset_rgb):
            if np.shape(frame) != (210, 160, 3) or np.asarray(frame).dtype != np.uint8:
                raise ValueError("observed and reset frames must be 210 by 160 uint8 RGB")
        # The staged transitions return new immutable scene, proposal and reduction values.
        staged = copy(self)
        receipt = staged._observe(rgb, applied, reward, False)
        if reset_rgb is not None:
            receipt["reset_observation"] = staged.observe(reset_rgb, 0, 0, reset=True)
        self.scene, self.reduction, self.pending = staged.scene, staged.reduction, staged.pending
        return receipt

    def checkpoint(self, path: str | Path, metadata: dict, *, exact: bool = True) -> None:
        """Save an exact continuation by default or an explicitly rounded compact one."""
        axiom_checkpoint.save_agent(path, self, metadata, exact=exact)

    def restore(self, path: str | Path, metadata: dict) -> None:
        """Restore on this factory after checking the independent run and trace identity."""
        axiom_checkpoint.restore(path, self, metadata)

    def choose(self):
        if self.pending is not None:
            return self.pending
        start = perf_counter()
        next_key, subkey = jax.random.split(self.key)
        tracks = self.scene.tracks
        n_tracked = _count(tracks.tracked)
        n_active = _count(self.scene.interaction.posterior.active)
        cache_stats = {}
        chunk = 512
        if not n_tracked:
            prepared = perf_counter()
            result = jax.block_until_ready(_empty_plan(self.proposal, subkey))
        else:
            identity = _context(self.scene.identity.posterior, self.scene.identity.generations,
                                self.virgin_identity)
            labels, valid = _labels(identity, tracks.scene, color_only=self.settings.banks.color_only)
            if not bool(valid):
                raise ArithmeticError("invalid identity evidence before planning")
            interaction = _context(self.scene.interaction.posterior, self.scene.interaction.generations,
                                   self.virgin_interaction)
            model = jax.block_until_ready(freeze_world(interaction, self.scene.motion, labels, tracks.tracked,
                                            self.scene.absent_distance, config=self.config))
            if not bool(model.valid):
                raise ArithmeticError(f"invalid frozen scene: {WorldStatus(int(model.status)).name}")
            prepared = perf_counter()
            if self.config.fixed_radius or self.config.static_contacts:
                chunk = _chunk(model.component_rows.size, model.focused_slots.size)
                result = _plan(model, tracks.scene, self.proposal, subkey, config=self.config, batch_size=chunk)
            else:
                result, cache_stats = cached_plan(model, tracks.scene, self.proposal, subkey,
                                                  config=self.config, settings=SearchSettings())
            result = jax.block_until_ready(result)
        finished = perf_counter()
        if not bool(result.valid):
            raise ArithmeticError(f"invalid full-budget decision at tick {self.tick}")
        self.pending = int(result.action)
        self.proposal = result.proposal
        self.key = next_key
        self.last_decision = dict(tick=self.tick, action=self.pending, tracked=n_tracked, active=n_active,
                                  policy_batch_size=chunk, horizon=32, candidates=512, replicates=3,
                                  total_score_range=float(np.ptp(np.asarray(result.total_return))),
                                  predicted_reward_range=float(np.ptp(np.asarray(result.reward))),
                                  novelty_range=float(np.ptp(np.asarray(result.novelty))),
                                  snapshot_s=prepared-start, plan_s=finished-prepared, decision_s=finished-start,
                                  **cache_stats)
        if self.audit_plan:
            self.last_decision["plan_sha256"] = _plan_digest((result, next_key))
        return self.pending

    def observe(self, rgb, applied, reward, reset=False):
        """Publish one valid observed transition, or one learning-disabled terminal reset."""
        if not reset and (self.pending is None or applied != self.pending):
            raise ValueError("applied command differs from the pending planned action")
        return self._observe(rgb, applied, reward, reset)

    def _observe(self, rgb, applied, reward, reset):
        action, rew = int(applied), float(reward)
        if action != applied or not 0 <= action < self.actions or not np.isfinite(rew) or not rew.is_integer():
            raise ValueError("invalid applied command or native reward")
        if reset:
            if self.pending is not None or action != 0 or rew != 0 or not bool(self.scene.can_reset):
                raise ValueError("a reset follows a completed real action and uses zero command and reward")
        tick = self.tick
        start = perf_counter()
        result = jax.block_until_ready(_observe(self.scene, rgb, jnp.int32(action), jnp.float32(rew),
             jnp.int32(tick), settings=self.settings, update=jnp.bool_(not reset), reset=jnp.bool_(reset)))
        observed = perf_counter()
        if not bool(result.valid):
            raise ArithmeticError(f"invalid observation at tick {tick}, spatial status {int(result.status)}")
        scene, history, reduction = result.state, self.reduction, None
        if not reset:
            bank, history, reduction = reduction_schedule.after_action(
                scene.interaction, history, step=tick+1, dormant_initial=self.virgin_interaction)
            scene = scene._replace(interaction=bank)
        self.scene, self.reduction, self.pending = scene, history, None
        return dict(valid=True, tick_before=tick, tick_after=self.tick, reset=bool(reset), update=not reset,
                    applied=action, reward=rew, spatial_status=int(result.status),
                    tracked=_count(scene.tracks.tracked), active_identity=_count(scene.identity.posterior.active),
                    active_motion=_count(scene.motion.occupied), active_interaction=_count(scene.interaction.posterior.active),
                    identity_writes=_count(result.identity_writes), interaction_writes=_count(result.interaction_writes),
                    observe_s=observed-start, reduction_performed=reduction is not None, reduction=reduction)

    def _leaves(self):
        return jax.tree.flatten_with_path((self.scene, self.proposal, self.key))

    def _identity(self):
        return json.dumps([1, self.game, self.actions, self.seed, self.initial_sha256])

    def state_dict(self):
        """NPZ-safe exact numerical payload. Timing is excluded from checkpoint identity."""
        leaves, _ = self._leaves()
        state = {"scene" + jax.tree_util.keystr(path): np.array(value, copy=True) for path, value in leaves}
        state["identity"] = np.asarray(self._identity())
        digest = self.last_decision.get("plan_sha256", "") if self.pending is not None else ""
        state["pending_plan_sha256"] = np.array(digest, dtype="<U64")
        state["pending"] = np.int32(-1 if self.pending is None else self.pending)
        state.update({"reduction/" + k: v for k, v in reduction_schedule.archive(self.reduction).items()})
        return state

    def load_state_dict(self, state):
        """Restore atomically on a fresh actor with the same initial RGB, game, action count and seed."""
        required = {"identity", "pending", "pending_plan_sha256"}
        if not required.issubset(state):
            raise ValueError(f"missing actor fields: {sorted(required - state.keys())}")
        axiom_checkpoint._check_reduction_layout(self, state)
        if str(np.asarray(state["identity"]).item()) != self._identity():
            raise ValueError("actor checkpoint belongs to a different initial run")
        pending = np.asarray(state["pending"])
        if pending.shape != () or pending.dtype != np.int32 or not -1 <= pending < self.actions:
            raise ValueError("actor checkpoint has an invalid pending command")
        reduction = {k[10:]: state[k] for k in state if k.startswith("reduction/")}
        leaves, layout = self._leaves()
        keys = ["scene" + jax.tree_util.keystr(path) for path, _ in leaves]
        if set(state) != {"identity", "pending", "pending_plan_sha256", *keys, *("reduction/" + k for k in reduction)}:
            raise ValueError("actor checkpoint inventory differs")
        values = []
        for name, (_, template) in zip(keys, leaves, strict=True):
            value = np.asarray(state[name])
            if value.dtype != template.dtype or value.shape != template.shape:
                raise ValueError(f"actor checkpoint layout mismatch: {name}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"nonfinite actor checkpoint field {name}")
            values.append(jnp.asarray(value))
        scene, proposal, key = jax.block_until_ready(layout.unflatten(values))
        axiom_checkpoint.validate_restored_scene(self, scene, proposal)
        digest = np.asarray(state["pending_plan_sha256"])
        if digest.shape or digest.dtype != np.dtype("<U64"):
            raise ValueError("invalid pending plan digest layout")
        digest = str(digest.item())
        if digest and (pending == -1 or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid pending plan digest")
        if self.audit_plan and pending != -1 and not digest:
            raise ValueError("pending plan was not audited in the saved checkpoint")
        history = reduction_schedule.restore(reduction, scene.interaction, step=int(scene.tick))
        self.scene, self.proposal, self.key, self.reduction = scene, proposal, key, history
        self.pending = None if pending == -1 else int(pending)
        self.last_decision = None if self.pending is None else {
            "tick": self.tick, "action": self.pending, "restored_pending": True}
        if digest:
            self.last_decision["plan_sha256"] = digest
