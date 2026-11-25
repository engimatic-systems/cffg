"""discrete readouts over explicit structure and current numerical inputs."""

from __future__ import annotations

import numpy as np

from cffg import ef
from cffg.compiled import NEG
from cffg.execution.transport import _disc_both, _disc_rows
from cffg.kernel_runtime import discrete_potential as _disc_potential
from cffg.kernel_runtime import parameters as _kernel_params
from cffg.readouts.selection import EdgeSelection, grouped_edges, selected_edges
from cffg.readouts.types import (
    Marginals,
)
from cffg.readouts.views import (
    DiscreteView,
    NodeView,
)
from cffg.runtime_types import State


def _card(edge):
    return 0 if edge.dim or edge.ef or edge.param_shape else edge.card


def marginals(view: DiscreteView, state: State, edges: EdgeSelection = None) -> dict[str, np.ndarray]:
    selected = selected_edges(view, edges, "categorical", _card)
    if not selected:
        return {}
    msgs = np.asarray(state.messages)
    out = {}
    for card, group in grouped_edges(selected, _card):
        slots = [(edge.slot0, edge.slot1) for edge in group]
        both = msgs[view.message_indices(slots, card)]
        # Restore literal log-zeros before multiplication can erase the sentinel.
        both = np.where(both == np.asarray(NEG, dtype=both.dtype), -np.inf, both)
        with np.errstate(invalid="ignore"):
            probabilities = ef.CATEGORICAL.mean(ef.CATEGORICAL.product(both[:, 0], both[:, 1]))
        out.update((edge.name, row) for edge, row in zip(group, probabilities, strict=True))
    return {edge.name: out[edge.name] for edge in selected}


def node_beliefs(view: NodeView, state: State) -> dict[str, np.ndarray]:
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
    # Keep the existing host materialization boundary.
    np.asarray(state.messages)

    def norm_exp(lb: np.ndarray) -> np.ndarray:
        m = lb.shape[0]
        flat = lb.reshape(m, -1)
        return ef.CATEGORICAL.mean(flat, axis=1).reshape(lb.shape)

    for g in view.groups:
        inc = np.asarray(_disc_rows(view, state.messages, g.gather, g.cards))
        if g.type == "equality":
            k = g.cards[0]
            q = norm_exp(ef.CATEGORICAL.product_axis(inc[:, :, :k], axis=1))
        elif g.mode == "vmp":
            both = np.asarray(_disc_both(view, state.messages, g.gather_both, g.cards))
            q = None
            for r, k in enumerate(g.cards):
                lq = ef.CATEGORICAL.product(both[:, r, 0, :k], both[:, r, 1, :k])
                edge_q = norm_exp(lq).reshape(len(g.members), *([1] * r), k)
                q = edge_q if q is None else q[..., None] * edge_q
        else:
            lb = np.asarray(_disc_potential(g, _kernel_params(g, state.params))).copy()
            for r, k in enumerate(g.cards):
                shape = [len(g.members)] + [1] * len(g.cards)
                shape[1 + r] = k
                lb = lb + inc[:, r, :k].reshape(shape)
            q = norm_exp(lb)
        for i, name in enumerate(g.members):
            out[name] = q[i]
    for g in view.dirichlet_groups:
        both = np.asarray(_disc_both(view, state.messages, g.gather_both, g.cards))
        q = None
        for r, k in enumerate(g.cards):
            lq = both[:, r, 0, :k] + both[:, r, 1, :k]
            edge_q = norm_exp(lq).reshape(len(g.members), *([1] * r), k)
            q = edge_q if q is None else q[..., None] * edge_q
        for i, name in enumerate(g.members):
            out[name] = q[i]
    return out


def decode(state: State, *, marginals: Marginals) -> dict[str, int]:
    """MAP decode x-hat: the per-edge argmax of the (max-)marginals. Defined
    in both semirings (sum-mode marginals decode too); in max mode at a tree
    fixed point this is the exact Viterbi assignment (cff-e8zk)."""
    return {name: int(np.argmax(q)) for name, q in marginals(state).items()}
