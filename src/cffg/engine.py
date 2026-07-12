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

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cffg import ef, efrules, gauss, mixg
from cffg.compiler import NEG, Arena
from cffg.rules import (
    BP_UPDATES,
    NODE_ENERGIES,
    VMP_ENERGIES,
    VMP_UPDATES,
    gfe_energy,
    gfe_tm_energy,
    gfe_tm_update,
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
    ef_eta: jax.Array  # [n_efslots, 2] EF natural parameters (cff-8s6a)
    ef_obs: jax.Array  # [n_ef_obs] conditioned counts/flips
    ef_mask: jax.Array  # [n_ef_obs] 1.0 = bound, 0.0 = missing


def _replace(state: State, **kw) -> State:
    return State(**{f: kw.get(f, getattr(state, f)) for f in (
        "messages", "params", "g_eta", "g_lam", "g_obs", "g_mask",
        "ef_eta", "ef_obs", "ef_mask",
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
            ef_eta=jnp.zeros((a.n_efslots, 2)),  # zeros = vacuous
            ef_obs=jnp.zeros(a.n_ef_obs),
            ef_mask=jnp.zeros(a.n_ef_obs),
        )

    def _uniform_messages(self) -> jax.Array:
        rows = np.full((self.arena.n_slots, self.arena.k_max), NEG)
        for e in self.arena.edges.values():
            if e.dim or e.ef:
                continue  # gaussian/EF slots live in their own arenas
            rows[e.slot0, : e.card] = -np.log(e.card)
            rows[e.slot1, : e.card] = -np.log(e.card)
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
                val = float(value)
                if fam == "poisson" and not (val >= 0 and val == int(val)):
                    raise ValueError(f"port {port!r}: Poisson counts are integers >= 0, got {value}")
                if fam == "bernoulli" and val not in (0.0, 1.0):
                    raise ValueError(f"port {port!r}: Bernoulli flips are 0 or 1, got {value}")
                ef_obs = ef_obs.at[row].set(val)
                ef_mask = ef_mask.at[row].set(1.0)
                continue
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
                    f"{sorted([*self.arena.data_slots, *self.arena.gauss_data, *self.arena.ef_data])}"
                )
            slot = self.arena.data_slots[port]
            card = next(
                e.card
                for e in self.arena.edges.values()
                if not e.dim and not e.ef and slot in (e.slot0, e.slot1)
            )
            row = np.full(self.arena.k_max, NEG)
            if value is None:
                row[:card] = -np.log(card)
            else:
                if not 0 <= value < card:
                    raise ValueError(f"port {port!r}: value {value} out of range [0, {card})")
                row[value] = 0.0
            msgs = msgs.at[slot].set(jnp.asarray(row, dtype=msgs.dtype))
        return _replace(
            state, messages=msgs, g_obs=g_obs, g_mask=g_mask, ef_obs=ef_obs, ef_mask=ef_mask
        )

    # -- execution --------------------------------------------------------

    def step(self, state: State, damp: float = 0.0, harden: bool = True) -> State:
        msgs, g_eta, g_lam, ef_eta = self._step(
            state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
            state.ef_eta, state.ef_obs, state.ef_mask, state.params, damp, harden,
        )
        return _replace(state, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta)

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
                        for f in ("messages", "g_eta", "g_lam", "ef_eta")
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
            if e.dim or e.ef:
                continue  # gaussian_marginals / ef_marginals
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

    def gaussian_node_stats(self, state: State) -> dict[str, dict[str, np.ndarray]]:
        """Factor name -> expected sufficient statistics of the node belief
        (cff-7c5o, D4): the continuous analog of node_beliefs, feeding the
        NIW/MNIW conjugate updates in cffg.conj.

        gaussian:            {"mean" E[x], "second" E[xxᵀ]}
        gaussian_linear:     {"Syy" E[yyᵀ], "Syx" E[yx̃ᵀ], "Sxx" E[x̃x̃ᵀ]}
                             with x̃ = [x; 1] (offset column absorbed)
        gaussian_linear_obs: same keys with y fixed at the bound observation,
                             plus "mask" (0.0 = missing: skip the occurrence).
        Requires proper beliefs (run to coherence first); equality and hybrid
        nodes are not learnable and are skipped."""
        out: dict[str, dict[str, np.ndarray]] = {}
        g_eta, g_lam = np.asarray(state.g_eta), np.asarray(state.g_lam)
        g_obs, g_mask = np.asarray(state.g_obs), np.asarray(state.g_mask)
        for g in self.arena.gauss_groups:
            if g.type == "equality":
                continue
            inc_e, inc_l = g_eta[g.gather], g_lam[g.gather]
            for i, name in enumerate(g.members):
                w = np.asarray(state.params[g.name + ":W"][i])
                if g.type == "gaussian":
                    (d,) = g.dims
                    m_blob = np.asarray(state.params[g.name + ":m"][i])
                    lam_q = w + inc_l[i, 0, :d, :d]
                    sig_q = np.linalg.inv(lam_q)
                    m_q = sig_q @ (w @ m_blob + inc_e[i, 0, :d])
                    out[name] = {"mean": m_q, "second": sig_q + np.outer(m_q, m_q)}
                elif g.type == "gaussian_linear":
                    do, di = g.dims
                    a = np.asarray(state.params[g.name + ":A"][i])
                    b = np.asarray(state.params[g.name + ":b"][i])
                    e_o, l_o = inc_e[i, 0, :do], inc_l[i, 0, :do, :do]
                    e_i, l_i = inc_e[i, 1, :di], inc_l[i, 1, :di, :di]
                    lam_j = np.block([[w + l_o, -w @ a], [-a.T @ w, a.T @ w @ a + l_i]])
                    eta_j = np.concatenate([w @ b + e_o, -a.T @ w @ b + e_i])
                    sig_j = np.linalg.inv(lam_j)
                    m_j = sig_j @ eta_j
                    m_y, m_x = m_j[:do], m_j[do:]
                    s_yy = sig_j[:do, :do] + np.outer(m_y, m_y)
                    s_yx = sig_j[:do, do:] + np.outer(m_y, m_x)
                    s_xx = sig_j[do:, do:] + np.outer(m_x, m_x)
                    out[name] = {
                        "Syy": s_yy,
                        "Syx": np.concatenate([s_yx, m_y[:, None]], axis=1),
                        "Sxx": np.block([[s_xx, m_x[:, None]], [m_x[None, :], np.ones((1, 1))]]),
                    }
                else:  # gaussian_linear_obs: y is data; q(x) is conditioned
                    do, di = g.dims
                    a = np.asarray(state.params[g.name + ":A"][i])
                    b = np.asarray(state.params[g.name + ":b"][i])
                    e_i, l_i = inc_e[i, 1, :di], inc_l[i, 1, :di, :di]
                    row = int(g.obs_rows[i])
                    y, mk = g_obs[row, :do], g_mask[row]
                    lam_q = mk * (a.T @ w @ a) + l_i
                    sig_q = np.linalg.inv(lam_q)
                    m_x = sig_q @ (mk * (a.T @ (w @ (y - b))) + e_i)
                    s_xx = sig_q + np.outer(m_x, m_x)
                    out[name] = {
                        "Syy": np.outer(y, y),
                        "Syx": np.concatenate(
                            [np.outer(y, m_x), y[:, None]], axis=1
                        ),
                        "Sxx": np.block(
                            [[s_xx, m_x[:, None]], [m_x[None, :], np.ones((1, 1))]]
                        ),
                        "mask": np.asarray(mk),
                    }
        return out

    def ef_marginals(self, state: State) -> dict[str, "EfMarginal"]:
        """Edge name -> EF marginal in standard parameters. Observation-family
        edges (poisson/bernoulli DATA ports) carry no messages and are
        skipped. An improper combined belief (e.g. before any sweep) reports
        proper=False with NaN moments — check it before trusting `mean`."""
        out: dict[str, EfMarginal] = {}
        eta = np.asarray(state.ef_eta)
        for e in self.arena.edges.values():
            if not e.ef or e.ef not in ef.FAMILIES:
                continue
            fam = ef.FAMILIES[e.ef]
            eta_q = eta[e.slot0] + eta[e.slot1]
            proper = bool(fam.is_proper(eta_q))
            out[e.name] = EfMarginal(
                family=e.ef,
                params=tuple(float(p) for p in fam.from_natural(eta_q)),
                mean=float(fam.mean(eta_q)) if proper else float("nan"),
                proper=proper,
            )
        return out

    def bfe(self, state: State) -> float:
        return float(
            self._bfe(
                state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
                state.ef_eta, state.ef_obs, state.ef_mask, state.params,
            )
        )


@dataclass(frozen=True)
class EfMarginal:
    """A scalar EF edge belief: standard parameters, e.g. (shape, rate) for
    gamma, (a, b) for beta."""

    family: str
    params: tuple[float, ...]
    mean: float
    proper: bool


def _pad_gauss(eta: jax.Array, lam: jax.Array, d_max: int) -> tuple[jax.Array, jax.Array]:
    """Zero-pad [M, d]/[M, d, d] rule outputs back to arena width."""
    p = d_max - eta.shape[-1]
    return (
        jnp.pad(eta, ((0, 0), (0, p))),
        jnp.pad(lam, ((0, 0), (0, p), (0, p))),
    )


def _pad_disc(rows: jax.Array, k_max: int) -> jax.Array:
    """NEG-pad a [M, K] block of log-messages out to the discrete arena width
    (log-space zero past the selector's own cardinality)."""
    return jnp.pad(rows, ((0, 0), (0, k_max - rows.shape[-1])), constant_values=NEG)


def _mix_sweep(arena, messages, g_eta, g_lam, params, k_max, d_max):
    """One synchronous flooding sweep over the hybrid Gaussian-mixture groups
    (cff-pksr). Reads only the pre-sweep arrays; returns the writes for both
    arenas so the caller can fold them into each arena's damping phase:
    `sel_writes` [(s_scatter, [M, k_max])] to the discrete messages, and
    `gauss_writes` [(g_scatter, [M, d_max], [M, d_max, d_max])] to (eta, Lam).
    Dims and K are static per group, so every rule sees exact slices."""
    sel_writes: list[tuple[jax.Array, jax.Array]] = []
    gauss_writes: list[tuple[jax.Array, jax.Array, jax.Array]] = []
    for g in arena.mix_groups:
        mu_e = messages[g.s_gather][:, : g.k_sel]  # [M, K] incoming selector msg
        inc_e = g_eta[g.g_gather]  # [M, R_g, d_max]
        inc_l = g_lam[g.g_gather]  # [M, R_g, d_max, d_max]
        w = params[g.name + ":W"]
        if g.type == "gaussian_mixture_linear":
            do, di = g.dims
            a, b = params[g.name + ":A"], params[g.name + ":b"]
            e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
            e_i, l_i = inc_e[:, 1, :di], inc_l[:, 1, :di, :di]
            g_outs = [
                mixg.mixlin_forward(e_i, l_i, mu_e, a, b, w),  # -> out
                mixg.mixlin_backward(e_o, l_o, mu_e, a, b, w),  # -> in
            ]
            to_sel = mixg.mixlin_to_selector(e_o, l_o, e_i, l_i, a, b, w)  # [M, K]
        else:  # gaussian_mixture
            (do,) = g.dims
            m = params[g.name + ":m"]
            e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
            g_outs = [mixg.mix_forward(mu_e, m, w)]  # -> out
            to_sel = mixg.mix_to_selector(e_o, l_o, m, w)  # [M, K]
        for r, (oe, ol) in enumerate(g_outs):
            oe, ol = _pad_gauss(oe, ol, d_max)
            gauss_writes.append((g.g_scatter[:, r], oe, ol))
        sel_writes.append((g.s_scatter, _pad_disc(to_sel, k_max)))
    return sel_writes, gauss_writes


def _prec_sweep(arena, messages_unused, g_eta, g_lam, ef_eta, params, d_max):
    """One synchronous flooding sweep over the gaussian_prec groups (cff-7c5o).
    Straddles the Gaussian and EF arenas; reads only the pre-sweep arrays and
    returns the writes for both so they fold into each arena's damping phase:
    `gauss_writes` [(x_scatter, [M, d_max], [M, d_max, d_max])] to (eta, Lam),
    `ef_writes` [(tau_scatter, [M, 2])] to the natural parameters. Both VMP
    messages read marginals (both slots summed), the mean-field convention."""
    gauss_writes: list[tuple[jax.Array, jax.Array, jax.Array]] = []
    ef_writes: list[tuple[jax.Array, jax.Array]] = []
    for g in arena.prec_groups:
        tau_q = ef_eta[g.tau_gather_both].sum(axis=1)  # [M, 2] τ-edge marginal
        x_eta_q = g_eta[g.x_gather_both].sum(axis=1)[:, :1]  # [M, 1]
        x_lam_q = g_lam[g.x_gather_both].sum(axis=1)[:, :1, :1]  # [M, 1, 1]
        m = params[g.name + ":m"]  # [M, 1]
        oe, ol = gauss.prec_to_x(tau_q, m)  # (eta [M,1], lam [M,1,1])
        oe, ol = _pad_gauss(oe, ol, d_max)
        gauss_writes.append((g.x_scatter, oe, ol))
        ef_writes.append((g.tau_scatter, gauss.prec_to_tau(x_eta_q, x_lam_q, m)))  # [M, 2]
    return gauss_writes, ef_writes


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


def _ef_sweep(arena, ef_eta, ef_obs, ef_mask, params):
    """One synchronous flooding sweep over the EF groups: prior and observed
    messages are constants (natural-parameter increments), equality is
    sum-minus-self — conjugate BP in its entirety (DERIVATIONS.md D2)."""
    new = ef_eta
    for g in arena.ef_groups:
        if g.type in ("gamma", "beta"):
            out = params[g.name + ":eta"].astype(new.dtype)
            new = new.at[g.scatter[:, 0]].set(out)
        elif g.type == "poisson_obs":
            x, mk = ef_obs[g.obs_rows], ef_mask[g.obs_rows]
            new = new.at[g.scatter[:, 0]].set(efrules.poisson_likelihood_message(x, mk))
        elif g.type == "bernoulli_obs":
            y, mk = ef_obs[g.obs_rows], ef_mask[g.obs_rows]
            new = new.at[g.scatter[:, 0]].set(efrules.bernoulli_likelihood_message(y, mk))
        else:  # equality
            out = efrules.equality_messages(ef_eta[g.gather])  # [M, 3, 2]
            new = new.at[g.scatter.reshape(-1)].set(out.reshape(-1, 2))
    return new


def _make_step(arena: Arena):
    groups = arena.groups
    k_max = arena.k_max

    def step(
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        ef_eta: jax.Array,
        ef_obs: jax.Array,
        ef_mask: jax.Array,
        params: dict[str, jax.Array],
        damp: float = 0.0,
        harden: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
        for g in arena.gfe_tm_groups:
            ko, ki, ku = g.cards
            md = messages[g.gather[:, 0]][:, :ko]
            mi = messages[g.gather[:, 1]][:, :ki]
            me = messages[g.gather[:, 2]][:, :ku]
            b_key = g.name + (":counts" if g.learned else ":B")
            to_out, to_in, to_sel = gfe_tm_update(
                mi, me, md, params[b_key], k_max, learned=g.learned
            )
            new = new.at[g.scatter[:, 0]].set(to_out)
            new = new.at[g.scatter[:, 1]].set(to_in)
            new = new.at[g.scatter[:, 2]].set(to_sel)
        # Hybrid Gaussian-mixture sweep (cff-pksr): straddles both arenas.
        # Computed here from the pre-sweep arrays so its writes fold into each
        # arena's damping phase — Gaussian writes below, selector writes with
        # the discrete blend (synchronous flooding: gather from OLD arrays).
        mix_sel, mix_gauss = (
            _mix_sweep(arena, messages, g_eta, g_lam, params, k_max, arena.d_max)
            if arena.mix_groups
            else ([], [])
        )
        # gaussian_prec sweep (cff-7c5o): also straddles two arenas — its out
        # edge writes the Gaussian arena, its prec edge the EF arena. Computed
        # here from the pre-sweep arrays so both writes fold into the matching
        # arena's damping phase (synchronous flooding: gather from OLD arrays).
        prec_gauss, prec_ef = (
            _prec_sweep(arena, messages, g_eta, g_lam, ef_eta, params, arena.d_max)
            if arena.prec_groups
            else ([], [])
        )
        # Gaussian sweep: information-form damping below is the log-space
        # damping of the Gaussian family (log of a Gaussian is its (eta,
        # Lambda) quadratic), so one damp knob covers both arenas.
        new_eta, new_lam = g_eta, g_lam
        if arena.gauss_groups:
            new_eta, new_lam = _gauss_sweep(arena, g_eta, g_lam, g_obs, g_mask, params)
        for scatter, oe, ol in [*mix_gauss, *prec_gauss]:
            new_eta = new_eta.at[scatter].set(oe)
            new_lam = new_lam.at[scatter].set(ol)
        if arena.gauss_groups or arena.mix_groups or arena.prec_groups:
            new_eta = damp * g_eta + (1.0 - damp) * new_eta
            new_lam = damp * g_lam + (1.0 - damp) * new_lam
        # EF sweep: natural parameters are the log-space of their family, so
        # the same damp knob is log-space damping here too.
        new_ef = ef_eta
        if arena.ef_groups or arena.prec_groups:
            if arena.ef_groups:
                new_ef = _ef_sweep(arena, ef_eta, ef_obs, ef_mask, params)
            for scatter, out in prec_ef:
                new_ef = new_ef.at[scatter].set(out)
            new_ef = damp * ef_eta + (1.0 - damp) * new_ef
        # Hybrid selector writes join the discrete arena before its damping.
        for scatter, rows in mix_sel:
            new = new.at[scatter].set(rows)
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
        return new, new_eta, new_lam, new_ef

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


def _mix_bfe(arena, messages, g_eta, g_lam, params):
    """Hybrid Gaussian-mixture node energies (cff-pksr): exact for the held
    labelled-mixture beliefs (mixg energies). The Gaussian and selector edge
    entropies need no code here — the out/in edges land in gauss_interior and
    the selector edge in interior_slot_pairs, both factor-agnostic tables."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for g in arena.mix_groups:
        mu_e = messages[g.s_gather][:, : g.k_sel]
        inc_e, inc_l = g_eta[g.g_gather], g_lam[g.g_gather]
        w = params[g.name + ":W"]
        if g.type == "gaussian_mixture_linear":
            do, di = g.dims
            a, b = params[g.name + ":A"], params[g.name + ":b"]
            e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
            e_i, l_i = inc_e[:, 1, :di], inc_l[:, 1, :di, :di]
            total = total + mixg.mixlin_energy(e_o, l_o, e_i, l_i, mu_e, a, b, w).sum()
        else:
            (do,) = g.dims
            m = params[g.name + ":m"]
            e_o, l_o = inc_e[:, 0, :do], inc_l[:, 0, :do, :do]
            total = total + mixg.mix_energy(e_o, l_o, mu_e, m, w).sum()
    return total


def _prec_bfe(arena, g_eta, g_lam, ef_eta, params):
    """gaussian_prec node energies (cff-7c5o): the mean-field VFE contribution
    −H[q_x] − H[q_τ] − E_q[log f]. gauss.prec_energy supplies −E_q[log f] (with
    the digamma correction that makes the mean-field VFE exact); the two edge
    neg-entropies are added here, exactly as the discrete VMP energies add one
    _neg_entropy per edge (cffg.rules.vmp_energy_matrix_cat). They net against
    the neighbours' −H and the interior +H (gauss_interior / ef_interior) to
    the single mean-field −H[q_i] per variable. Improper marginals ⇒ +inf."""
    total = jnp.asarray(0.0, dtype=g_eta.dtype)
    for g in arena.prec_groups:
        tau_q = ef_eta[g.tau_gather_both].sum(axis=1)  # [M, 2]
        x_eta_q = g_eta[g.x_gather_both].sum(axis=1)[:, :1]  # [M, 1]
        x_lam_q = g_lam[g.x_gather_both].sum(axis=1)[:, :1, :1]  # [M, 1, 1]
        m = params[g.name + ":m"]
        e = gauss.prec_energy(tau_q, x_eta_q, x_lam_q, m)  # −E_q[log f], +inf if improper
        neg_h_x = gauss._neg_entropy(x_lam_q)  # −H[q_x]
        neg_h_tau = -efrules.GAMMA.entropy(tau_q)  # −H[q_τ]
        ok = (x_lam_q[..., 0, 0] > 0) & efrules.GAMMA.is_proper(tau_q)
        total = total + jnp.where(ok, neg_h_x + neg_h_tau + e, jnp.inf).sum()
    return total


def _ef_bfe(arena, ef_eta, ef_obs, ef_mask, params):
    """EF node energies + interior-edge entropies, all closed form
    (cffg.efrules). Improper beliefs anywhere make the total +inf — loud.
    With conditioned observations the tree identity BFE = −log p(data)
    holds exactly (no ridge: nothing here is iterative or solved)."""
    total = jnp.asarray(0.0, dtype=ef_eta.dtype)
    for g in arena.ef_groups:
        fam = efrules.JFAMILIES[g.family]
        inc = ef_eta[g.gather]  # [M, R, 2]
        if g.type in ("gamma", "beta"):
            total = total + efrules.prior_energy(
                fam, inc[:, 0], params[g.name + ":eta"].astype(ef_eta.dtype)
            ).sum()
        elif g.type == "poisson_obs":
            x, mk = ef_obs[g.obs_rows], ef_mask[g.obs_rows]
            total = total + efrules.poisson_obs_energy(inc[:, 0], x, mk).sum()
        elif g.type == "bernoulli_obs":
            y, mk = ef_obs[g.obs_rows], ef_mask[g.obs_rows]
            total = total + efrules.bernoulli_obs_energy(inc[:, 0], y, mk).sum()
        else:  # equality
            total = total + efrules.equality_energy(fam, inc).sum()
    for fam_name, pairs in arena.ef_interior:
        fam = efrules.JFAMILIES[fam_name]
        total = total + efrules.edge_entropy(
            fam, ef_eta[pairs[:, 0]], ef_eta[pairs[:, 1]]
        ).sum()
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
        ef_eta: jax.Array,
        ef_obs: jax.Array,
        ef_mask: jax.Array,
        params: dict[str, jax.Array],
    ) -> jax.Array:
        total = jnp.asarray(0.0, dtype=messages.dtype)
        # prec_groups included so the interior-edge entropy tables run even for
        # a prec-only model: _gauss_bfe owns gauss_interior (the out edges),
        # _ef_bfe owns ef_interior (the prec edges), both EdgeInfo-driven.
        if arena.gauss_groups or arena.prec_groups:
            total = total + _gauss_bfe(arena, g_eta, g_lam, g_obs, g_mask, params)
        if arena.mix_groups:
            total = total + _mix_bfe(arena, messages, g_eta, g_lam, params)
        if arena.ef_groups or arena.prec_groups:
            total = total + _ef_bfe(arena, ef_eta, ef_obs, ef_mask, params)
        if arena.prec_groups:
            total = total + _prec_bfe(arena, g_eta, g_lam, ef_eta, params)
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
        for g in arena.gfe_tm_groups:
            ko, ki, ku = g.cards
            md = messages[g.gather[:, 0]][:, :ko]
            mi = messages[g.gather[:, 1]][:, :ki]
            me = messages[g.gather[:, 2]][:, :ku]
            b_key = g.name + (":counts" if g.learned else ":B")
            total = total + gfe_tm_energy(
                mi, me, md, params[b_key], learned=g.learned
            ).sum()
        if len(pairs):
            lq = messages[pairs[:, 0]] + messages[pairs[:, 1]]  # [E_int, k_max]
            lq = lq - logsumexp(lq, axis=-1, keepdims=True)
            q = jnp.exp(lq)
            entropy = -jnp.where(q > _TINY, q * lq, 0.0).sum()
            total = total + entropy
        return total

    return bfe
