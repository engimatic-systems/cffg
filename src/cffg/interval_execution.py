"""Message transport and signed tree ledger for interval-only Forney forests."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg import intervals as iv


class IntervalState(NamedTuple):
    params: dict
    messages: tuple
    # Original CPU endpoints certify narrowing of future DATA bindings.
    # Transport takes only the device parameters and messages.
    sources: dict


def _source(node, state):
    low, high, h, count, tails = (state.params[k] for k in node.blobs)
    index = jnp.arange(low.shape[1])
    left = tails[:, 0, None] & (index == 0)
    right = tails[:, 1, None] & (index == count[:, None] - 1)
    return iv.IntervalMessage(
        jnp.where(left, -jnp.inf, low), jnp.where(right, jnp.inf, high), h, count
    )


def _send(node, leg, state, incoming):
    if node.kind == "interval_potential":
        return iv.IntervalResult(_source(node, state), jnp.int32(iv.IntervalStatus.OK))
    if node.kind == "interval_activation":
        op = iv.pullback if leg == 0 else iv.pushforward
        return op(incoming[1 - leg], *(state.params[k] for k in node.blobs))
    others = [m for j, m in enumerate(incoming) if j != leg]
    return iv.product(*others)


def propagate(plan, state, *, synchronous):
    """On failure keep every original message, including independent coordinates."""
    messages = list(state.messages)
    code, owner = jnp.int32(iv.IntervalStatus.OK), jnp.int32(-1)
    for i, leg in plan.order:
        node = plan.nodes[i]
        source = state.messages if synchronous else messages
        sent = _send(node, leg, state, [source[j] for j in node.incoming])
        bad = (code == iv.IntervalStatus.OK) & (sent.status != iv.IntervalStatus.OK)
        owner = jnp.where(bad, i, owner)
        code = jnp.where(bad, sent.status, code)
        messages[node.own[leg]] = sent.message
    messages = jax.tree.map(
        lambda new, old: jnp.where(code == iv.IntervalStatus.OK, new, old),
        tuple(messages),
        state.messages,
    )
    return state._replace(messages=messages), code, owner


def edge_belief(state, edge):
    result = iv.product(*(state.messages[j] for j in edge.slots))
    stats, normalized, mass = iv._measure(result.message)
    status = jnp.where(
        result.status != iv.IntervalStatus.OK, result.status, stats.status
    )
    return normalized, mass, stats, status


def ledger(plan, state):
    edges = [edge_belief(state, e) for e in plan.edges.values()]
    # All edges include their two endpoints. A boundary port supplies a
    # unary likelihood (the unit when missing), so also owns a node term.
    separators = jnp.stack([b[2].entropy for b in edges])
    code = jnp.stack([b[3] for b in edges])
    factor_terms = []
    for node in plan.nodes:
        edge = node.own[0] // 2
        belief, mass, _, _ = edges[edge]
        term = -separators[edge]
        if node.kind == "interval_potential":
            term -= iv._expected_height(belief, mass, _source(node, state))
        factor_terms.append(term)
    for name in plan.ports.values():
        i, j = plan.edges[name].slots
        belief, mass, _, _ = edges[i // 2]
        factor_terms.append(
            -separators[i // 2] - iv._expected_height(belief, mass, state.messages[j])
        )
    factor_terms = jnp.stack(factor_terms)
    value = jnp.sum(factor_terms) + jnp.sum(separators)
    # An undefined arithmetic or storage operation is not a support certificate.
    priority = (
        iv.IntervalStatus.INVALID,
        iv.IntervalStatus.NUMERICAL,
        iv.IntervalStatus.CAPACITY,
        iv.IntervalStatus.INFEASIBLE,
        iv.IntervalStatus.IMPROPER,
    )
    status = jnp.select(
        [jnp.any(code == s) for s in priority],
        priority,
        default=jnp.where(
            jnp.isfinite(value), iv.IntervalStatus.OK, iv.IntervalStatus.NUMERICAL
        ),
    )
    # Independent evidence identity after complete propagation. A substantial
    # discrepancy signals unresolved arithmetic, even if the ledger is finite.
    log_mass = jnp.stack([row[2].log_mass for row in edges])
    log_z = jnp.zeros((), value.dtype)
    cert = jnp.asarray(True)
    eps = jnp.finfo(value.dtype).eps
    budget = 128 * eps * (len(edges) + len(factor_terms))
    for component in plan.components:
        ref = component[0]
        idx = jnp.array(component)
        finite = code[ref] == iv.IntervalStatus.OK
        scale = 1 + jnp.abs(log_mass[ref])
        cert &= jnp.all(code[idx] == code[ref]) & (
            ~finite | jnp.all(jnp.abs(log_mass[idx] - log_mass[ref]) <= budget * scale)
        )
        log_z += log_mass[ref]
    cert &= (status != iv.IntervalStatus.OK) | (
        jnp.abs(value + log_z) <= budget * (1 + jnp.abs(log_z))
    )
    return factor_terms, separators, value, status, cert
