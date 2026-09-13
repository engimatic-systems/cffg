"""State initialization, observation binding and fresh-trial goal priors."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from cffg import kernels
from cffg._immutable import FrozenMap
from cffg.compiled import NEG, Arena
from cffg.runtime_types import State, _replace


@dataclass(frozen=True)
class StateLifecycle:
    """Host operations over immutable layout and current numerical inputs."""

    arena: Arena
    gaussian_validators: Mapping[str, Callable] = field(init=False, repr=False)

    def __post_init__(self):
        validators = {
            int(row): group.kernel.binding.conditioning.validate
            for group in self.arena.gauss_groups
            if group.obs_rows is not None
            for row in group.obs_rows
        }
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
        return {name: jnp.asarray(p) for name, p in self.arena.log_params.items()}

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
        """Bind DATA ports: discrete ports take an int (None = missing =
        uniform); Gaussian ports take a length-dim vector, EF ports a scalar
        count/flip (None = missing = the likelihood node drops out)."""
        msgs, g_obs, g_mask = state.messages, state.g_obs, state.g_mask
        ef_obs, ef_mask = state.ef_obs, state.ef_mask
        for port, value in observations.items():
            if port in self.arena.ef_data:
                row, fam = self.arena.ef_data[port]
                if value is None:
                    ef_mask = ef_mask.at[row].set(0.0)
                    continue
                validator = kernels.observation_validator(fam)
                try:
                    val = float(value) if validator is None else validator(value)
                except ValueError as err:
                    raise ValueError(f"port {port!r}: {err}") from None
                ef_obs = ef_obs.at[row].set(val)
                ef_mask = ef_mask.at[row].set(1.0)
                continue
            if port in self.arena.gauss_data:
                row, dim = self.arena.gauss_data[port]
                if value is None:
                    g_mask = g_mask.at[row].set(0.0)
                else:
                    try:
                        val = np.asarray(
                            self.gaussian_validators[port](value, dim), dtype=np.float64
                        )
                        if val.shape != (dim,):
                            raise ValueError(
                                f"observation validator returned shape {val.shape}, expected {(dim,)}"
                            )
                    except ValueError as err:
                        raise ValueError(f"port {port!r}: {err}") from None
                    idx = self.arena.g_obs_indices(row, dim)
                    g_obs = g_obs.at[idx].set(jnp.asarray(val, dtype=g_obs.dtype))
                    g_mask = g_mask.at[row].set(1.0)
                continue
            if port not in self.arena.data_slots:
                raise KeyError(
                    f"no DATA port {port!r}; have "
                    f"{sorted([*self.arena.data_slots, *self.arena.gauss_data, *self.arena.ef_data])}"
                )
            slot = self.arena.data_slots[port]
            card = next(
                e.card
                for e in self.arena.edges.values()
                if (
                    not e.dim
                    and not e.ef
                    and not e.param_shape
                    and slot in (e.slot0, e.slot1)
                )
            )
            row = np.full(card, NEG)
            if value is None:
                row[:card] = -np.log(card)
            else:
                if not 0 <= value < card:
                    raise ValueError(
                        f"port {port!r}: value {value} out of range [0, {card})"
                    )
                row[value] = 0.0
            idx = self.arena.message_indices(slot, card)
            msgs = msgs.at[idx].set(jnp.asarray(row, dtype=msgs.dtype))
        return _replace(
            state,
            messages=msgs,
            g_obs=g_obs,
            g_mask=g_mask,
            ef_obs=ef_obs,
            ef_mask=ef_mask,
        )

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
