"""State initialization, observation binding and fresh-trial goal priors."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from cffg import kernels
from cffg._immutable import FrozenMap
from cffg.compiled import NEG, Arena
from cffg.changing_learning import initial_count_errors
from cffg.runtime_types import State, _replace


@jax.jit
def _scatter_observations(updates):
    # Missing numeric observations have index -1. Drop those writes to retain
    # cached values while binding an arbitrary mask with one compiled shape.
    return {
        name: buffer.at[indices].set(values, mode="drop", wrap_negative_indices=False)
        for name, (buffer, indices, values) in updates.items()
    }


class _ObservationWrites:
    """Collect validated segments in host memory until the entire bind succeeds."""

    def __init__(self):
        self.parts = {}

    def add(self, name, indices, values):
        self.parts.setdefault(name, []).append((indices, values))

    def numeric(self, values_name, mask_name, row, indices, value):
        self.add(mask_name, (row,), (value is not None,))
        self.add(values_name, indices, value)

    def apply(self, state):
        updates = {}
        for name, parts in self.parts.items():
            buffer = getattr(state, name)
            count = sum(len(indices) for indices, _ in parts)
            # Reuse compiled executables across different partial bind sizes.
            # The padding address is dropped and cannot overwrite a cached value.
            capacity = min(buffer.size, max(16, 1 << (count - 1).bit_length()))
            addresses = np.full(capacity, -1, dtype=np.intp)
            payload = np.zeros(capacity, dtype=buffer.dtype)
            start = 0
            for indices, values in parts:
                end = start + len(indices)
                if values is not None:
                    if isinstance(indices, range):
                        addresses[start:end] = np.arange(indices.start, indices.stop)
                    else:
                        addresses[start:end] = indices
                    payload[start:end] = values
                start = end
            updates[name] = (buffer, addresses, payload)
        return _replace(state, **_scatter_observations(updates)) if updates else state


@dataclass(frozen=True)
class StateLifecycle:
    """Host operations over immutable layout and current numerical inputs."""

    arena: Arena
    categorical_segments: Mapping[str, range] = field(init=False, repr=False)
    gaussian_segments: Mapping[str, range] = field(init=False, repr=False)
    gaussian_validators: Mapping[str, Callable] = field(init=False, repr=False)

    changing_count_errors: tuple[tuple[str, str, str], ...] = field(init=False, repr=False)

    def __post_init__(self):
        a = self.arena
        object.__setattr__(self, "changing_count_errors", initial_count_errors(a))
        message_starts = np.asarray(a.message_starts)
        gaussian_starts = np.asarray(a.g_obs_starts)
        object.__setattr__(self, "categorical_segments", FrozenMap({
            port: range(message_starts[slot], message_starts[slot + 1])
            for port, slot in a.data_slots.items()
        }))
        object.__setattr__(self, "gaussian_segments", FrozenMap({
            port: range(gaussian_starts[row], gaussian_starts[row + 1])
            for port, (row, _) in a.gauss_data.items()
        }))
        validators = {
            int(row): group.kernel.binding.conditioning.validate
            for group in self.arena.gauss_groups
            if group.obs_rows is not None
            for row in group.obs_rows
        }
        if a.gaussian_taps:
            from cffg.gaussian_taps import validate_point
            validators.update(dict.fromkeys(a.gaussian_taps.values(), validate_point))
        object.__setattr__(
            self,
            "gaussian_validators",
            FrozenMap(
                {
                    port: validators[row]
                    for port, (row, _) in self.arena.gauss_data.items()
                }
            ),
        )

    def initial_parameters(self) -> dict[str, jax.Array]:
        """Convert defaults using the precision setting active at this call."""
        params = {name: jnp.asarray(p) for name, p in self.arena.log_params.items()}
        for key, dtype, error in self.changing_count_errors:
            if params[key].dtype.str == dtype:
                raise ValueError(error)
        return params

    def init_state(self, params: dict[str, jax.Array]) -> State:
        """Install supplied initial parameters and representation-specific arrays."""
        a = self.arena
        return State(
            messages=self.uniform_messages(),
            params=params,
            g_eta=jnp.zeros(a.n_g_eta_values),  # zeros = vacuous
            g_lam=jnp.zeros(a.n_g_lam_values),
            g_obs=jnp.zeros(a.n_g_obs_values),
            g_mask=jnp.zeros(a.n_obs_rows),
            ef_eta=jnp.zeros((a.n_efslots, 2)),  # zeros = vacuous
            ef_obs=jnp.zeros(a.n_ef_obs),
            ef_mask=jnp.zeros(a.n_ef_obs),
        )

    def uniform_messages(self) -> jax.Array:
        rows = np.empty(self.arena.n_message_values)
        for e in self.arena.edges.values():
            if e.dim or e.ef:
                continue  # gaussian/EF slots live in their own arenas
            initial = 0.0 if e.param_shape else -np.log(e.card)
            rows[self.arena.message_indices(e.slot0, e.card)] = initial
            rows[self.arena.message_indices(e.slot1, e.card)] = initial
        return jnp.asarray(rows)

    def bind_data(self, state: State, observations: dict) -> State:
        """Validate the partial host update, then scatter just its named observations."""
        writes = _ObservationWrites()
        for port, value in observations.items():
            if port in self.arena.ef_data:
                row, fam = self.arena.ef_data[port]
                if value is not None:
                    validator = kernels.observation_validator(fam)
                    try:
                        value = float(value) if validator is None else validator(value)
                    except ValueError as err:
                        raise ValueError(f"port {port!r}: {err}") from None
                writes.numeric("ef_obs", "ef_mask", row, (row,),
                               None if value is None else (value,))
            elif port in self.arena.gauss_data:
                row, dim = self.arena.gauss_data[port]
                if value is not None:
                    try:
                        value = np.asarray(self.gaussian_validators[port](value, dim),
                                           dtype=np.float64)
                        if self.arena.gaussian_taps and row in self.arena.gaussian_taps.values():
                            if not np.all(np.abs(value) <= np.finfo(state.g_obs.dtype).max):
                                raise ValueError("tap value must be finite in the state dtype")
                        if value.shape != (dim,):
                            raise ValueError(
                                f"observation validator returned shape {value.shape}, expected {(dim,)}"
                            )
                    except ValueError as err:
                        raise ValueError(f"port {port!r}: {err}") from None
                writes.numeric("g_obs", "g_mask", row, self.gaussian_segments[port], value)
            elif port in self.categorical_segments:
                indices = self.categorical_segments[port]
                card = len(indices)
                if value is None:
                    message = np.full(card, -np.log(card))
                else:
                    if not 0 <= value < card:
                        raise ValueError(f"port {port!r}: value {value} out of range [0, {card})")
                    message = np.full(card, NEG)
                    message[value] = 0.0
                writes.add("messages", indices, message)
            else:
                raise KeyError(
                    f"no DATA port {port!r}; have "
                    f"{sorted([*self.arena.data_slots, *self.arena.gauss_data, *self.arena.ef_data])}"
                )
        return writes.apply(state)

    def with_goal_priors(self, state: State, priors: dict[str, np.ndarray]) -> State:
        """Install trial priors on a fresh state without accumulating sweeps.

        Counts already present on a goal parameter edge prove that the state
        has participated in inference. Callers must start the next trial from
        ``init_state`` rather than add a posterior to an old count message.
        """
        known = {
            blob for group in self.arena.gfe_groups for blob in group.goal_prior_blobs
        }
        unknown = sorted(set(priors) - known)
        if unknown:
            raise KeyError(
                f"unknown learned-goal prior blobs {unknown}; have {sorted(known)}"
            )
        params = dict(state.params)
        for group in self.arena.gfe_groups:
            if not group.goal_learned:
                continue
            existing = np.asarray(
                state.messages[
                    self.arena.message_indices(group.goal_count_scatter, group.cards[0])
                ]
            )
            if np.any(existing != 0):
                raise ValueError(
                    "goal priors can only be installed on a fresh inference "
                    "state; within-trial count messages are replaced per sweep"
                )
            alpha0 = params[group.name + ":c_alpha0"]
            for row, blob in enumerate(group.goal_prior_blobs):
                if blob not in priors:
                    continue
                value = np.asarray(priors[blob])
                if (
                    value.shape != (group.cards[0],)
                    or not np.all(np.isfinite(value))
                    or not np.all(value > 0)
                ):
                    raise ValueError(
                        f"goal prior {blob!r}: expected finite positive shape "
                        f"{(group.cards[0],)}, got {value.shape}"
                    )
                alpha0 = alpha0.at[row].set(jnp.asarray(value, dtype=alpha0.dtype))
            params[group.name + ":c_alpha0"] = alpha0
        return _replace(state, params=params)
