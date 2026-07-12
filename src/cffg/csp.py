"""Hard-constraint (indicator) factor library and min-sum helpers (cff-e8zk).

Constraint factors are indicator functions over a runtime-chosen number of
discrete edges x0..x{n-1}: log-potential entries live in {0, NEG}. Their
messages are computed COMBINATORIALLY — the full potential tensor is never
materialized (K^n). Each type ships a message function and a node-energy
function, mirroring cffg.rules conventions:

  * message fn (inc [n, k_max] log-messages, cards, k_max, op) -> [n, k_max]
    normalized+padded outgoing log-messages. `op` is the marginalization
    reduction: logsumexp for sum-product / max for min-sum (max-product). The
    engine passes op = arena's semiring reduction, so one code path serves both
    the BFE ledger (sum) and MAP inference (max).

  * node-energy fn (inc [n, k_max], cards) -> scalar Bethe node term for the
    sum-product ledger. An indicator has log f = 0 on its support, so its node
    term is E_q[log q_a] on the support: F_a = E_q[weight] - log Z_a where
    weight(x) = Σ_j M[j][x_j] and Z_a is the constraint's (sum-semiring)
    partition function — both read off the SAME exact DP, no enumeration of the
    K^n joint (the -log Z_a on the support is exactly what makes the constrained
    BFE report -log Z of the constrained model, cff-e8zk spec note 3).

alldifferent uses an exact bitmask DP over used-value subsets (a submatrix
permanent in the sum semiring; the assignment problem in max) — O(2^K) in the
shared cardinality K, validated K, n <= 10 at compile. Larger n is the
permanent-approximation regime (Huang & Jebara, arXiv:0908.1769); the exact DP
here is the anchor those approximations are measured against. Max-product on
alldifferent is exact and convergent for max-weight bipartite matching iff the
MAP is unique (Bayati, Shah & Sharma, IEEE-IT 54(3) 2008).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from cffg.compiler import NEG

_TINY = 1e-12


def _norm_pad(x: jnp.ndarray, card: int, k_max: int) -> jnp.ndarray:
    """Normalize a [card] log-message and NEG-pad to k_max (constant shift; the
    normalizer is arbitrary for max-product, so logsumexp serves both)."""
    x = x - logsumexp(x)
    pad = jnp.full((k_max - card,), NEG, dtype=x.dtype)
    return jnp.concatenate([x, pad])


# --- alldifferent -----------------------------------------------------------


def _alldiff_raw(inc: jnp.ndarray, K: int, op) -> jnp.ndarray:
    """Raw (unnormalized) outgoing messages [n, K] for alldifferent over n edges
    of shared cardinality K. Message to edge i at value v = op over injective
    assignments of the OTHER n-1 edges to distinct values != v of the sum of
    their incoming messages. Bitmask DP over used-value subsets: for target i,
    dp[S] = op over injective assignments of edges != i using exactly value-set
    S; msg_to_i(v) = op over S not containing v of dp[S]."""
    n = inc.shape[0]
    full = 1 << K
    Sar = jnp.arange(full)
    bit_in = jnp.stack([((Sar >> v) & 1) == 1 for v in range(K)])  # [K, full]
    src = jnp.stack([Sar ^ (1 << v) for v in range(K)])  # [K, full]: S with v removed
    outs = []
    for i in range(n):
        dp = jnp.full((full,), NEG).at[0].set(0.0)
        for j in range(n):
            if j == i:
                continue
            mj = inc[j, :K]
            # newdp[S'] = op_v [ v in S' ] ( dp[S' \ v] + mj[v] )
            per_v = jnp.where(bit_in, dp[src] + mj[:, None], NEG)  # [K, full]
            dp = op(per_v, axis=0)
        msg_v = [op(jnp.where(bit_in[v], NEG, dp), axis=0) for v in range(K)]
        outs.append(jnp.stack(msg_v))
    return jnp.stack(outs)  # [n, K]


def alldifferent_messages(inc, cards, k_max, op):
    K = cards[0]  # shared cardinality (validated equal at compile)
    raw = _alldiff_raw(inc, K, op)
    return jnp.stack([_norm_pad(raw[i], K, k_max) for i in range(raw.shape[0])])


def alldifferent_energy(inc, cards):
    K = cards[0]
    raw = _alldiff_raw(inc, K, logsumexp)  # sum semiring for the ledger
    return _indicator_energy(inc, raw, cards)


# --- exactly_k / at_most_k (binary edges) -----------------------------------


def _count_raw(inc: jnp.ndarray, k: int, op, at_most: bool) -> jnp.ndarray:
    """Raw outgoing messages [n, 2] for a cardinality constraint on n binary
    edges: exactly_k enforces Σ x_j == k, at_most_k enforces Σ x_j <= k. Prefix-
    count DP: forward[i][c] = op over edges < i summing to c; backward[i][c] =
    op over edges > i summing to c; message to edge i at bit b combines the two
    halves subject to the total count constraint."""
    n = inc.shape[0]
    ncount = n + 1  # counts 0..n
    m0 = inc[:, 0]
    m1 = inc[:, 1]

    def roll_up(dp, mv0, mv1):  # add one edge: dp'[c] = op(dp[c]+m0, dp[c-1]+m1)
        stay = dp + mv0
        take = jnp.concatenate([jnp.full((1,), NEG), dp[:-1]]) + mv1
        return op(jnp.stack([stay, take]), axis=0)

    fwd = [jnp.full((ncount,), NEG).at[0].set(0.0)]  # fwd[i]: edges 0..i-1
    for i in range(n):
        fwd.append(roll_up(fwd[i], m0[i], m1[i]))
    bwd = [None] * (n + 1)
    bwd[n] = jnp.full((ncount,), NEG).at[0].set(0.0)  # bwd[i]: edges i..n-1
    for i in range(n - 1, -1, -1):
        bwd[i] = roll_up(bwd[i + 1], m0[i], m1[i])

    counts = jnp.arange(ncount)
    outs = []
    for i in range(n):
        left = fwd[i]  # over edges < i, count c_l
        right = bwd[i + 1]  # over edges > i, count c_r
        # combined[c] = op over c_l + c_r == c of left[c_l] + right[c_r]
        comb = left[:, None] + right[None, :]  # [ncount, ncount]
        tot = counts[:, None] + counts[None, :]  # c_l + c_r
        msg_b = []
        for b in range(2):
            if at_most:
                keep = tot <= (k - b)
            else:
                keep = tot == (k - b)
            msg_b.append(op(jnp.where(keep, comb, NEG).reshape(-1), axis=0))
        outs.append(jnp.stack(msg_b))  # [2]
    return jnp.stack(outs)  # [n, 2]


def exactly_k_messages(inc, cards, k_max, op, *, k):
    raw = _count_raw(inc, k, op, at_most=False)
    return jnp.stack([_norm_pad(raw[i], 2, k_max) for i in range(raw.shape[0])])


def at_most_k_messages(inc, cards, k_max, op, *, k):
    raw = _count_raw(inc, k, op, at_most=True)
    return jnp.stack([_norm_pad(raw[i], 2, k_max) for i in range(raw.shape[0])])


def exactly_k_energy(inc, cards, *, k):
    raw = _count_raw(inc, k, logsumexp, at_most=False)
    return _indicator_energy(inc, raw, cards)


def at_most_k_energy(inc, cards, *, k):
    raw = _count_raw(inc, k, logsumexp, at_most=True)
    return _indicator_energy(inc, raw, cards)


# --- table (explicit support enumeration) -----------------------------------


def _table_weights(inc: jnp.ndarray, table: jnp.ndarray) -> jnp.ndarray:
    """Per-tuple log-weight w_t = Σ_j M[j][table[t, j]] over the allowed tuples
    (table [T, n] of ints)."""
    n = table.shape[1]
    cols = [inc[j][table[:, j]] for j in range(n)]  # each [T]
    return jnp.stack(cols).sum(axis=0)  # [T]


def table_messages(inc, cards, k_max, op, *, table):
    n = table.shape[1]
    w = _table_weights(inc, table)  # [T]
    outs = []
    for j in range(n):
        Kj = cards[j]
        vals = table[:, j]  # [T]
        # msg_j(v) = op over tuples with table[:,j]==v of (w_t - M[j][v])
        contrib = w - inc[j][vals]
        msg = jnp.stack(
            [op(jnp.where(vals == v, contrib, NEG), axis=0) for v in range(Kj)]
        )
        outs.append(_norm_pad(msg, Kj, k_max))
    return jnp.stack(outs)


def table_energy(inc, cards, *, table):
    w = _table_weights(inc, table)  # [T]
    logZ = logsumexp(w)
    q = jnp.exp(w - logZ)
    # F_a = Σ_t q_t w_t - log Z_a  ( = Σ q_a log q_a on the support, log f = 0 )
    return (q * w).sum() - logZ


# --- shared indicator node energy -------------------------------------------


def _indicator_energy(inc: jnp.ndarray, raw: jnp.ndarray, cards) -> jnp.ndarray:
    """Bethe node term for an indicator factor from its exact sum-semiring
    messages `raw` [n, K]: F_a = E_q[weight] - log Z_a, with log f = 0 on the
    support. q_j(v) = softmax_v(M[j][v] + raw_j(v)) is the node-belief marginal
    on edge j; Z_a = Σ_v exp(M[0][v] + raw_0(v)) (consistent across edges)."""
    n = inc.shape[0]
    K0 = cards[0]
    belief0 = inc[0, :K0] + raw[0, :K0]
    logZ = logsumexp(belief0)
    e = jnp.asarray(0.0)
    for j in range(n):
        Kj = cards[j]
        b = inc[j, :Kj] + raw[j, :Kj]
        qj = jnp.exp(b - logsumexp(b))
        e = e + (qj * inc[j, :Kj]).sum()
    return e - logZ


# Dispatch tables (mirroring rules.BP_UPDATES / rules.NODE_ENERGIES). The k /
# table scalars are static compile-time detail, so they are bound per-factor by
# the engine rather than passed through here.
CONSTRAINT_MESSAGES = {
    "alldifferent": alldifferent_messages,
    "exactly_k": exactly_k_messages,
    "at_most_k": at_most_k_messages,
    "table": table_messages,
}
CONSTRAINT_ENERGIES = {
    "alldifferent": alldifferent_energy,
    "exactly_k": exactly_k_energy,
    "at_most_k": at_most_k_energy,
    "table": table_energy,
}


# --- decimation (outer loop; agent/runtime level, not engine machinery) ------


def clamp_edge(engine, state, var: str, value: int):
    """Clamp a discrete edge to a one-hot pointmass by writing BOTH message
    slots (mirroring the engine's delta-harden write). Returns a new State; the
    edge stays pinned across subsequent sweeps. Compile-time delta constraints
    are unavailable here (decimation chooses the edge at runtime), so this is
    the runtime analog — a documented state-level write."""
    import numpy as np

    e = engine.arena.edges[var]
    row = np.full(engine.arena.k_max, NEG)
    row[value] = 0.0
    row = jnp.asarray(row, dtype=state.messages.dtype)
    msgs = state.messages.at[e.slot0].set(row).at[e.slot1].set(row)
    from cffg.engine import _replace

    return _replace(state, messages=msgs)


def decimate(engine, state, *, max_rounds=None, n_sweeps=20, damp=0.0, tol=1e-9):
    """BP-guided decimation (Montanari, Ricci-Tersenghi & Semerjian,
    arXiv:0709.1667): run message passing to coherence, fix the lowest-entropy
    free edge to the argmax of its marginal, and repeat. Returns
    (state, decoded) where decoded maps every clamped variable to its value.

    Each round: run `n_sweeps` (soft), pick the unclamped edge whose marginal
    has the least Shannon entropy (the most-polarized decision), clamp it via a
    one-hot message write to both slots, and continue. Stops when all edges are
    clamped or `max_rounds` is reached. The clamp is the runtime cousin of the
    engine's compile-time delta machinery."""
    import numpy as np

    arena = engine.arena
    free = [
        e.name for e in arena.edges.values() if not e.dim and not e.ef
    ]
    if max_rounds is None:
        max_rounds = len(free)
    clamped: dict[str, int] = {}
    for _ in range(min(max_rounds, len(free))):
        state = engine.run(
            state, n_sweeps, tol=tol, damp=damp, harden_last=0
        ).require_converged()
        q = engine.marginals(state)
        best, best_ent, best_val = None, np.inf, None
        for name in free:
            if name in clamped:
                continue
            p = np.clip(q[name], 1e-12, 1.0)
            ent = float(-(p * np.log(p)).sum())
            if ent < best_ent:
                best, best_ent, best_val = name, ent, int(np.argmax(q[name]))
        if best is None:
            break
        clamped[best] = best_val
        state = clamp_edge(engine, state, best, best_val)
    state = engine.run(
        state, n_sweeps, tol=tol, damp=damp, harden_last=0
    ).require_converged()
    return state, clamped
