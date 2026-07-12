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

from cffg import gauss
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
    """Pure-data instance state. The Gaussian arrays are zero-size on
    discrete-only models — same pytree structure either way, so colonies mix
    freely. g_obs/g_mask are the *conditioned* Gaussian observations (see
    cffg.gauss: conditioning, not clamping) with mask 0 = missing."""

    messages: jax.Array  # [n_slots, k_max] log-space
    params: dict[str, jax.Array]  # group name -> [M, ...] log-factors
    g_eta: jax.Array  # [n_gslots, d_max] information vectors
    g_lam: jax.Array  # [n_gslots, d_max, d_max] precisions
    g_obs: jax.Array  # [n_obs_rows, d_max] observation values
    g_mask: jax.Array  # [n_obs_rows] 1.0 = bound, 0.0 = missing


def _replace(state: State, **kw) -> State:
    return State(**{f: kw.get(f, getattr(state, f)) for f in (
        "messages", "params", "g_eta", "g_lam", "g_obs", "g_mask"
    )})


class Engine:
    """Compiled, executable model. Construction is cheap; `step` is jitted."""

    def __init__(self, arena: Arena):
        self.arena = arena
        self._step = eqx.filter_jit(_make_step(arena))
        self._bfe = eqx.filter_jit(_make_bfe(arena))

    # -- state construction ---------------------------------------------

    def init_state(self) -> State:
        a = self.arena
        return State(
            messages=self._uniform_messages(),
            params={name: jnp.asarray(p) for name, p in a.log_params.items()},
            g_eta=jnp.zeros((a.n_gslots, a.d_max)),  # zeros = vacuous
            g_lam=jnp.zeros((a.n_gslots, a.d_max, a.d_max)),
            g_obs=jnp.zeros((a.n_obs_rows, a.d_max)),
            g_mask=jnp.zeros(a.n_obs_rows),
        )

    def _uniform_messages(self) -> jax.Array:
        rows = np.full((self.arena.n_slots, self.arena.k_max), NEG)
        for e in self.arena.edges.values():
            if e.dim:
                continue  # gaussian slots live in the (eta, Lambda) arena
            rows[e.slot0, : e.card] = -np.log(e.card)
            rows[e.slot1, : e.card] = -np.log(e.card)
        return jnp.asarray(rows)

    def bind_data(self, state: State, observations: dict) -> State:
        """Bind DATA ports: discrete ports take an int (None = missing =
        uniform); Gaussian ports take a length-dim vector (None = missing =
        the likelihood node drops out)."""
        msgs, g_obs, g_mask = state.messages, state.g_obs, state.g_mask
        for port, value in observations.items():
            if port in self.arena.gauss_data:
                row, dim = self.arena.gauss_data[port]
                if value is None:
                    g_mask = g_mask.at[row].set(0.0)
                else:
                    val = np.asarray(value, dtype=np.float64).reshape(-1)
                    if val.shape[0] != dim:
                        raise ValueError(
                            f"port {port!r}: observation has dim {val.shape[0]}, edge has {dim}"
                        )
                    vec = np.zeros(self.arena.d_max)
                    vec[:dim] = val
                    g_obs = g_obs.at[row].set(jnp.asarray(vec, dtype=g_obs.dtype))
                    g_mask = g_mask.at[row].set(1.0)
                continue
            if port not in self.arena.data_slots:
                raise KeyError(
                    f"no DATA port {port!r}; have "
                    f"{sorted([*self.arena.data_slots, *self.arena.gauss_data])}"
                )
            slot = self.arena.data_slots[port]
            card = next(
                e.card
                for e in self.arena.edges.values()
                if not e.dim and slot in (e.slot0, e.slot1)
            )
            row = np.full(self.arena.k_max, NEG)
            if value is None:
                row[:card] = -np.log(card)
            else:
                if not 0 <= value < card:
                    raise ValueError(f"port {port!r}: value {value} out of range [0, {card})")
                row[value] = 0.0
            msgs = msgs.at[slot].set(jnp.asarray(row, dtype=msgs.dtype))
        return _replace(state, messages=msgs, g_obs=g_obs, g_mask=g_mask)

    # -- execution --------------------------------------------------------

    def step(self, state: State, damp: float = 0.0, harden: bool = True) -> State:
        msgs, g_eta, g_lam = self._step(
            state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
            state.params, damp, harden,
        )
        return _replace(state, messages=msgs, g_eta=g_eta, g_lam=g_lam)

    def run(
        self,
        state: State,
        n_iters: int,
        *,
        tol: float | None = None,
        damp: float = 0.0,
        harden_last: int | None = None,
    ) -> tuple[State, list[float]]:
        """Flood n_iters sweeps (or until max message delta < tol); returns the
        final state and the per-sweep delta trace — the 'coherence' signal.

        `damp` (0..1) blends in the previous messages; use ~0.5 for models with
        multiple GFE composites, whose outer coordination can limit-cycle.

        δ form-constraints must not fire while beliefs are still garbage — a
        premature pointmass locks in its own arbitrary argmax. With
        `harden_last=k` (default: k = n_iters // 2 when δ edges exist), the
        first phase runs soft to coherence, then δ edges harden for the final
        k sweeps so decisions commit sequentially."""
        if harden_last is None:
            harden_last = n_iters // 2 if len(self.arena.delta_pairs) else 0
        deltas: list[float] = []

        def phase(n: int, harden: bool, st: State) -> State:
            for _ in range(n):
                nxt = self.step(st, damp, harden)
                delta = max(
                    (
                        float(jnp.max(jnp.abs(getattr(nxt, f) - getattr(st, f))))
                        for f in ("messages", "g_eta", "g_lam")
                        if getattr(st, f).size
                    ),
                    default=0.0,
                )
                deltas.append(delta)
                st = nxt
                # A non-finite delta is divergence, never convergence: it must
                # not satisfy tol (cff-i45j: an -inf delta once returned a
                # transient sweep-4 state as "converged").
                if tol is not None and np.isfinite(delta) and delta < tol:
                    break
            return st

        state = phase(n_iters - harden_last, harden=False, st=state)
        if harden_last:
            state = phase(harden_last, harden=True, st=state)
        return state, deltas

    # -- readouts ---------------------------------------------------------

    def marginals(self, state: State) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        msgs = np.asarray(state.messages)
        for e in self.arena.edges.values():
            if e.dim:
                continue  # gaussian_marginals
            lq = msgs[e.slot0, : e.card] + msgs[e.slot1, : e.card]
            lq -= lq.max()
            q = np.exp(lq)
            out[e.name] = q / q.sum()
        return out

    def node_beliefs(self, state: State) -> dict[str, np.ndarray]:
        """Factor name -> the node belief q_a, the object conjugate updates
        integrate against (cff-cd1i): e.g. a transition_mixture's q_a(z', z, u)
        is the smoothed pairwise marginal whose slices are Dirichlet count
        increments for B-learning.

        BP groups: exact q_a ∝ f_a · Π incoming, shaped [*cards]. VMP groups:
        the mean-field belief (outer product of edge marginals) — the correct
        expectation measure under that bead. Equality nodes report the shared
        marginal [k] (their joint lives on the diagonal). GFE composites and
        Gaussian groups are not included: composites own their update path
        (agent A-learning), Gaussian sufficient stats land with cff-7c5o."""
        out: dict[str, np.ndarray] = {}
        msgs = np.asarray(state.messages)

        def norm_exp(lb: np.ndarray) -> np.ndarray:
            m = lb.shape[0]
            flat = lb.reshape(m, -1)
            flat = flat - flat.max(axis=1, keepdims=True)
            q = np.exp(flat)
            return (q / q.sum(axis=1, keepdims=True)).reshape(lb.shape)

        for g in self.arena.groups:
            inc = msgs[g.gather]  # [M, R, k_max]
            if g.type == "equality":
                k = g.cards[0]
                q = norm_exp(inc[:, :, :k].sum(axis=1))
            elif g.mode == "vmp":
                both = msgs[g.gather_both]  # [M, R, 2, k_max]
                q = None
                for r, k in enumerate(g.cards):
                    lq = both[:, r, 0, :k] + both[:, r, 1, :k]
                    edge_q = norm_exp(lq).reshape(len(g.members), *([1] * r), k)
                    q = edge_q if q is None else q[..., None] * edge_q
            else:
                lb = np.asarray(state.params[g.name]).copy()  # [M, *cards]
                for r, k in enumerate(g.cards):
                    shape = [len(g.members)] + [1] * len(g.cards)
                    shape[1 + r] = k
                    lb = lb + inc[:, r, :k].reshape(shape)
                q = norm_exp(lb)
            for i, name in enumerate(g.members):
                out[name] = q[i]
        return out

    def gaussian_marginals(self, state: State) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Edge name -> (mean, cov). Requires a proper combined precision —
        run to coherence (and bind data) first; improper beliefs surface as
        huge covariances, deliberately loud."""
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        eta, lam = np.asarray(state.g_eta), np.asarray(state.g_lam)
        for e in self.arena.edges.values():
            if not e.dim:
                continue
            d = e.dim
            eta_q = eta[e.slot0, :d] + eta[e.slot1, :d]
            lam_q = lam[e.slot0, :d, :d] + lam[e.slot1, :d, :d]
            mean, cov = gauss.moments(eta_q, lam_q)
            out[e.name] = (np.asarray(mean), np.asarray(cov))
        return out

    def bfe(self, state: State) -> float:
        return float(
            self._bfe(
                state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
                state.params,
            )
        )


def _pad_gauss(eta: jax.Array, lam: jax.Array, d_max: int) -> tuple[jax.Array, jax.Array]:
    """Zero-pad [M, d]/[M, d, d] rule outputs back to arena width."""
    p = d_max - eta.shape[-1]
    return (
        jnp.pad(eta, ((0, 0), (0, p))),
        jnp.pad(lam, ((0, 0), (0, p), (0, p))),
    )


def _gauss_sweep(arena, g_eta, g_lam, g_obs, g_mask, params):
    """One synchronous flooding sweep over the Gaussian groups; returns the
    fully rewritten (eta, Lambda) arrays. Dims are static per group, so every
    rule call sees exact [:d] slices of the padded arena."""
    d_max = arena.d_max
    new_eta, new_lam = g_eta, g_lam
    for g in arena.gauss_groups:
        inc_e = g_eta[g.gather]  # [M, R, d_max]
        inc_l = g_lam[g.gather]  # [M, R, d_max, d_max]
        outs: list[tuple[jax.Array, jax.Array]] = []  # per role, [M, d]/[M, d, d]
        if g.type == "gaussian":
            (d,) = g.dims
            w, m = params[g.name + ":W"], params[g.name + ":m"]
            outs.append(gauss.prior_message(jnp.einsum("mij,mj->mi", w, m), w))
        elif g.type == "equality":
            d = g.dims[0]
            to_e, to_l = gauss.equality_messages(inc_e[..., :d], inc_l[..., :d, :d])
            outs = [(to_e[:, r], to_l[:, r]) for r in range(len(g.roles))]
        else:  # gaussian_linear / gaussian_linear_obs; roles = (out, in)
            do, di = g.dims
            a, b, w = (params[g.name + k] for k in (":A", ":b", ":W"))
            e_in, l_in = inc_e[:, 1, :di], inc_l[:, 1, :di, :di]
            outs.append(gauss.linear_forward(e_in, l_in, a, b, w))
            if g.type == "gaussian_linear_obs":
                y, mk = g_obs[g.obs_rows][:, :do], g_mask[g.obs_rows]
                outs.append(gauss.linear_observed_backward(y, mk, a, b, w))
            else:
                e_out, l_out = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
                outs.append(gauss.linear_backward(e_out, l_out, a, b, w))
        for r, (oe, ol) in enumerate(outs):
            oe, ol = _pad_gauss(oe, ol, d_max)
            new_eta = new_eta.at[g.scatter[:, r]].set(oe)
            new_lam = new_lam.at[g.scatter[:, r]].set(ol)
    return new_eta, new_lam


def _make_step(arena: Arena):
    groups = arena.groups
    k_max = arena.k_max

    def step(
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        params: dict[str, jax.Array],
        damp: float = 0.0,
        harden: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
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
            a_key = g.name + (":counts" if g.learned else ":A")
            mu2, log_qx = gfe_update(
                log_d, params[a_key], params[g.name + ":c"], k_max, learned=g.learned
            )
            new = new.at[g.scatter_z].set(mu2)
            new = new.at[g.scatter_x].set(log_qx)
        # Gaussian sweep: information-form damping below is the log-space
        # damping of the Gaussian family (log of a Gaussian is its (eta,
        # Lambda) quadratic), so one damp knob covers both arenas.
        new_eta, new_lam = g_eta, g_lam
        if arena.gauss_groups:
            new_eta, new_lam = _gauss_sweep(arena, g_eta, g_lam, g_obs, g_mask, params)
            new_eta = damp * g_eta + (1.0 - damp) * new_eta
            new_lam = damp * g_lam + (1.0 - damp) * new_lam
        # Log-space damping: stabilizes the outer coordination between GFE
        # composites (their local solves are exact, but synchronous flooding
        # between them can limit-cycle — the outer face of Part II §4.5).
        new = damp * messages + (1.0 - damp) * new
        # Floor at NEG: exact log-zeros compound additively through chains
        # (NEG + NEG + ...), and unchecked they overflow to -inf and NaN once
        # δ-hardened controls make states truly impossible. exp(NEG) == 0
        # either way, so clamping changes nothing semantically.
        new = jnp.maximum(new, NEG)
        if harden and len(arena.delta_pairs):
            # δ form constraint: project each constrained edge onto a pointmass
            # at the argmax of its current marginal (both slots one-hot).
            p0, p1 = arena.delta_pairs[:, 0], arena.delta_pairs[:, 1]
            lq = new[p0] + new[p1]
            hot = jnp.where(
                jnp.arange(k_max)[None, :] == jnp.argmax(lq, axis=-1)[:, None], 0.0, NEG
            ).astype(new.dtype)
            new = new.at[p0].set(hot).at[p1].set(hot)
        return new, new_eta, new_lam

    return step


def _gauss_bfe(arena, g_eta, g_lam, g_obs, g_mask, params):
    """Gaussian node energies + interior-edge differential entropies. Exact
    closed forms (cffg.gauss docstring); with conditioned observations the
    tree identity BFE = −log p(y) holds to fixed-point/ridge precision."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for g in arena.gauss_groups:
        inc_e, inc_l = g_eta[g.gather], g_lam[g.gather]
        if g.type == "gaussian":
            (d,) = g.dims
            total = total + gauss.prior_energy(
                inc_e[:, 0, :d], inc_l[:, 0, :d, :d],
                params[g.name + ":m"], params[g.name + ":W"],
            ).sum()
        elif g.type == "equality":
            d = g.dims[0]
            total = total + gauss.equality_energy(inc_e[..., :d], inc_l[..., :d, :d]).sum()
        else:
            do, di = g.dims
            a, b, w = (params[g.name + k] for k in (":A", ":b", ":W"))
            e_i, l_i = inc_e[:, 1, :di], inc_l[:, 1, :di, :di]
            if g.type == "gaussian_linear_obs":
                y, mk = g_obs[g.obs_rows][:, :do], g_mask[g.obs_rows]
                total = total + gauss.linear_observed_energy(e_i, l_i, y, mk, a, b, w).sum()
            else:
                e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
                total = total + gauss.linear_energy(e_o, l_o, e_i, l_i, a, b, w).sum()
    for d, pairs in arena.gauss_interior:
        l0 = g_lam[pairs[:, 0], :d, :d]
        l1 = g_lam[pairs[:, 1], :d, :d]
        total = total + gauss.edge_entropy(l0, l1).sum()
    return total


def _make_bfe(arena: Arena):
    groups = arena.groups
    pairs = arena.interior_slot_pairs

    def bfe(
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        params: dict[str, jax.Array],
    ) -> jax.Array:
        total = jnp.asarray(0.0, dtype=messages.dtype)
        if arena.gauss_groups:
            total = total + _gauss_bfe(arena, g_eta, g_lam, g_obs, g_mask, params)
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
            a_key = g.name + (":counts" if g.learned else ":A")
            total = total + gfe_energy(
                log_d, params[a_key], params[g.name + ":c"], learned=g.learned
            ).sum()
        if len(pairs):
            lq = messages[pairs[:, 0]] + messages[pairs[:, 1]]  # [E_int, k_max]
            lq = lq - logsumexp(lq, axis=-1, keepdims=True)
            q = jnp.exp(lq)
            entropy = -jnp.where(q > _TINY, q * lq, 0.0).sum()
            total = total + entropy
        return total

    return bfe
