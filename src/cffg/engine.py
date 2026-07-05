"""The inference VM: flooding message passing over a compiled arena.

State is pure data (an Equinox pytree of the message array plus per-group
log-parameters); the step is a pure function of it. Index tables from the
Arena are captured as numpy constants inside the jitted step, so instances
sharing a structure batch under `eqx.filter_vmap` (colony contract).

Free energy: `bfe` evaluates F = Σ_a F[q_a] + Σ_{interior edges} H[q_i]
exactly — node terms from the rule registry, plus the Bethe edge-entropy
correction (interior edges have degree 2, hence counting number d_i − 1 = 1).
On tree graphs at the BP fixed point, F = −log Z.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cffg.compiler import NEG, Arena
from cffg.rules import (
    BP_UPDATES,
    NODE_ENERGIES,
    VMP_ENERGIES,
    VMP_UPDATES,
    gfe_energy,
    gfe_update,
)

_TINY = 1e-12


class State(eqx.Module):
    messages: jax.Array  # [n_slots, k_max] log-space
    params: dict[str, jax.Array]  # group name -> [M, ...] log-factors


class Engine:
    """Compiled, executable model. Construction is cheap; `step` is jitted."""

    def __init__(self, arena: Arena):
        self.arena = arena
        self._step = eqx.filter_jit(_make_step(arena))
        self._bfe = eqx.filter_jit(_make_bfe(arena))

    # -- state construction ---------------------------------------------

    def init_state(self) -> State:
        return State(messages=self._uniform_messages(), params={
            name: jnp.asarray(logf) for name, logf in self.arena.log_params.items()
        })

    def _uniform_messages(self) -> jax.Array:
        rows = np.full((self.arena.n_slots, self.arena.k_max), NEG)
        for e in self.arena.edges.values():
            rows[e.slot0, : e.card] = -np.log(e.card)
            rows[e.slot1, : e.card] = -np.log(e.card)
        return jnp.asarray(rows)

    def bind_data(self, state: State, observations: dict[str, int | None]) -> State:
        """Clamp DATA ports to observed values (None = missing = uniform)."""
        msgs = state.messages
        for port, value in observations.items():
            if port not in self.arena.data_slots:
                raise KeyError(f"no DATA port {port!r}; have {sorted(self.arena.data_slots)}")
            slot = self.arena.data_slots[port]
            card = next(
                e.card for e in self.arena.edges.values() if slot in (e.slot0, e.slot1)
            )
            row = np.full(self.arena.k_max, NEG)
            if value is None:
                row[:card] = -np.log(card)
            else:
                if not 0 <= value < card:
                    raise ValueError(f"port {port!r}: value {value} out of range [0, {card})")
                row[value] = 0.0
            msgs = msgs.at[slot].set(jnp.asarray(row, dtype=msgs.dtype))
        return State(messages=msgs, params=state.params)

    # -- execution --------------------------------------------------------

    def step(self, state: State) -> State:
        return State(messages=self._step(state.messages, state.params), params=state.params)

    def run(self, state: State, n_iters: int, *, tol: float | None = None) -> tuple[State, list[float]]:
        """Flood n_iters sweeps (or until max message delta < tol); returns the
        final state and the per-sweep delta trace — the 'coherence' signal."""
        deltas: list[float] = []
        for _ in range(n_iters):
            nxt = self.step(state)
            delta = float(jnp.max(jnp.abs(nxt.messages - state.messages)))
            deltas.append(delta)
            state = nxt
            if tol is not None and delta < tol:
                break
        return state, deltas

    # -- readouts ---------------------------------------------------------

    def marginals(self, state: State) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        msgs = np.asarray(state.messages)
        for e in self.arena.edges.values():
            lq = msgs[e.slot0, : e.card] + msgs[e.slot1, : e.card]
            lq -= lq.max()
            q = np.exp(lq)
            out[e.name] = q / q.sum()
        return out

    def bfe(self, state: State) -> float:
        return float(self._bfe(state.messages, state.params))


def _make_step(arena: Arena):
    groups = arena.groups
    k_max = arena.k_max

    def step(messages: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
        new = messages
        for g in groups:
            logf = params.get(g.name)
            if g.mode == "vmp":
                inc2 = messages[g.gather_both]  # [M, R, 2, k_max]
                out = VMP_UPDATES[g.type](inc2, logf, g.cards, k_max)
            else:
                inc = messages[g.gather]  # [M, R, k_max] (synchronous: reads old)
                out = BP_UPDATES[g.type](inc, logf, g.cards, k_max)
            new = new.at[g.scatter.reshape(-1)].set(out.reshape(-1, k_max))
        for g in arena.gfe_groups:
            _, ki = g.cards
            log_d = messages[g.gather_z][:, :ki]
            mu2, log_qx = gfe_update(log_d, params[g.name + ":A"], params[g.name + ":c"], k_max)
            new = new.at[g.scatter_z].set(mu2)
            new = new.at[g.scatter_x].set(log_qx)
        return new

    return step


def _make_bfe(arena: Arena):
    groups = arena.groups
    pairs = arena.interior_slot_pairs

    def bfe(messages: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
        total = jnp.asarray(0.0, dtype=messages.dtype)
        for g in groups:
            logf = params.get(g.name)
            if logf is None:
                logf = jnp.zeros(())  # equality handles its own zero log-f
            if g.mode == "vmp":
                total = total + VMP_ENERGIES[g.type](messages[g.gather_both], logf, g.cards).sum()
            else:
                total = total + NODE_ENERGIES[g.type](messages[g.gather], logf, g.cards).sum()
        for g in arena.gfe_groups:
            _, ki = g.cards
            log_d = messages[g.gather_z][:, :ki]
            total = total + gfe_energy(log_d, params[g.name + ":A"], params[g.name + ":c"]).sum()
        if len(pairs):
            lq = messages[pairs[:, 0]] + messages[pairs[:, 1]]  # [E_int, k_max]
            lq = lq - logsumexp(lq, axis=-1, keepdims=True)
            q = jnp.exp(lq)
            entropy = -jnp.where(q > _TINY, q * lq, 0.0).sum()
            total = total + entropy
        return total

    return bfe
