"""Compare staged iteration checks with enumeration and the host contracts."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.compiled import NEG
from cffg.compiler import compile_template
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.engine import Engine, RunStatus
from cffg.iteration_checks import make_iteration_support
from cffg.runtime_types import _replace


def _constraint_engine(kind, cards, payload, *, semiring="sum", harden=False):
    names = tuple(f"x{i}" for i in range(len(cards)))
    role = "table" if kind == "table" else "k"
    arrays = {} if payload is None else {"value": np.asarray(payload)}
    template = Template(
        "iteration-support",
        variables={name: VariableSpec(name, card) for name, card in zip(names, cards, strict=True)},
        factors={
            "constraint": FactorSpec(
                "constraint", kind, {name: name for name in names},
                param_blobs={} if payload is None else {role: "value"},
            )
        },
        ports={name: PortSpec(name, name, PortRole.DATA) for name in names},
        blobs={} if payload is None else {"value": BlobRef("value", "value.npy")},
        constraints=[ConstraintSpec(ConstraintKind.DELTA, "x0")] if harden else [],
    )
    return Engine(compile_template(template, arrays=arrays, semiring=semiring), schedule="flood")


def _constraint_state(engine, supports, payload, dtype):
    state = engine.init_state()
    plan, = engine.arena.constraint_groups
    messages = state.messages.astype(dtype)
    for slot, support in zip(plan.gather, supports, strict=True):
        indices = engine.arena.message_indices(slot, len(support))
        # Even an extreme finite value below NEG has exact support by contract.
        messages = messages.at[indices].set(jnp.where(support, -2e30, NEG).astype(dtype))
    params = dict(state.params)
    if plan.parameter_key is not None:
        params[plan.parameter_key] = jnp.asarray(payload, dtype=params[plan.parameter_key].dtype)
    return _replace(state, messages=messages, params=params)


def _enumerate_support(kind, supports, payload):
    for assignment in product(*(np.flatnonzero(row) for row in supports)):
        if kind == "exactly_k" and sum(assignment) == payload:
            return True
        if kind == "at_most_k" and sum(assignment) <= payload:
            return True
        if kind == "table" and tuple(assignment) in tuple(map(tuple, payload)):
            return True
        if kind == "alldifferent" and len(set(assignment)) == len(assignment):
            return True
    return False


@pytest.mark.parametrize("kind, cards, payloads", [
    ("exactly_k", (2, 2, 2), (0, 1, 3)),
    ("at_most_k", (2, 2, 2), (0, 1, 3)),
    ("table", (2, 3, 4), ([(1, 1, 3),(1, 2, 0)], [(0, 1, 3),(1, 2, 0)])),
    ("alldifferent", (3, 3, 3), (None,)),
])
def test_mapped_constraint_support_matches_live_payload_and_tuple_enumeration(kind, cards, payloads):
    engine = _constraint_engine(kind, cards, payloads[0])
    rng = np.random.default_rng(193)
    supports = [
        tuple(np.ones(card, dtype=bool) for card in cards),
        tuple(np.arange(card) == 0 for card in cards),
        tuple(np.arange(card) < 2 for card in cards),
        tuple(np.arange(card) > 0 for card in cards),
        tuple(np.zeros(card, dtype=bool) for card in cards),
        *(tuple(rng.random(card) < 0.55 for card in cards) for _ in range(12)),
    ]
    # For all-different, this Hall violation has no empty domain and no
    # pair forced to the same value. A greedy or pairwise test misses it.
    if kind == "alldifferent":
        assert all(sum(row) == 2 for row in supports[2])
        assert not _enumerate_support(kind, supports[2], None)

    check = jax.jit(jax.vmap(make_iteration_support(engine.arena)))
    for dtype in (jnp.float32, jnp.float64):
        cases = [(rows, payload) for rows in supports for payload in payloads]
        states = [_constraint_state(engine, rows, data, dtype) for rows, data in cases]
        expected = [not _enumerate_support(kind, rows, data) for rows, data in cases]
        np.testing.assert_array_equal(expected, [engine._has_zero_support(s) for s in states])
        batch = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)
        np.testing.assert_array_equal(check(batch), expected)


def test_all_different_at_the_compiler_capacity():
    engine = _constraint_engine("alldifferent", (10,) * 10, None)
    check = jax.jit(make_iteration_support(engine.arena))
    labels = np.arange(10)
    disjoint = tuple(labels == i for i in range(10))
    # Ten variables fit ten values, but this subset violates Hall's condition.
    hall = tuple(labels < 3 if i < 4 else labels == i for i in range(10))
    for rows, empty in ((disjoint, False), (hall, True)):
        state = _constraint_state(engine, rows, None, jnp.float32)
        assert bool(check(state)) == engine._has_zero_support(state) == empty


def test_edge_support_requires_two_facing_messages_and_checks_sentinel_exactly():
    engine = _constraint_engine("exactly_k", (2, 2), 0)
    state = engine.init_state()
    edge = engine.arena.edges["x0"]
    left = engine.arena.message_indices(edge.slot0, 2)
    right = engine.arena.message_indices(edge.slot1, 2)
    check = jax.jit(make_iteration_support(engine.arena))
    for supported in (False, True):
        # With [-2e30, NEG] on the left and [0, -1e30] on the right the
        # logits' sum is equal at the two values, but only label zero lives.
        facing = jnp.array([0, NEG]) if supported else jnp.array([NEG, 0])
        messages = state.messages.at[left].set(jnp.array([-2e30, NEG])).at[right].set(facing)
        marked = _replace(state, messages=messages)
        assert bool(check(marked)) == engine._has_zero_support(marked) == (not supported)


@pytest.mark.parametrize("harden", (False, True))
def test_max_stationarity_certifies_the_host_probability_tie(harden):
    table = [(0, 0),(1, 1)]
    engine = _constraint_engine("table", (3, 2), table, semiring="max", harden=harden)
    state = engine.init_state()
    edge = engine.arena.edges["x0"]
    left = engine.arena.message_indices(edge.slot0, 3)
    right = engine.arena.message_indices(edge.slot1, 3)
    # These distinct float32 logits can produce equal host probabilities.
    # Even with a stationary sweep, only one of the two assignments is legal.
    logits = np.array([-8.7424112e-08, 0, -1.4377002], dtype=np.float32)
    messages = state.messages.astype(jnp.float32).at[left].set(logits).at[right].set(0)
    state = _replace(state, messages=messages)
    assert np.argmax(logits) == 1
    host_decode = engine.decode(state)["x0"]
    assert np.isfinite(engine.objective(state)) == (host_decode == 0)
    plan, = engine.arena.constraint_groups
    for table in ([(0, 0),(1, 1)], [(0, 1),(1, 0)]):
        payload = jnp.asarray(table, dtype=state.params[plan.parameter_key].dtype)
        params = {**state.params, plan.parameter_key: payload}
        candidate = _replace(state, params=params)
        assert not engine._has_zero_support(candidate)
        result = engine.run(candidate, 2, damp=1., harden_last=int(harden), tol=1e-6)
        finite = np.isfinite(engine.objective(candidate))
        if finite and harden:
            assert result.status is RunStatus.BUDGET_EXHAUSTED
            assert result.iterations == 2
            # Hardening uses logit argmax, which selects one and changes the state.
            assert engine.decode(result.state)["x0"] == 1
        else:
            assert result.status is (RunStatus.CONVERGED if finite else RunStatus.DIVERGED)
            assert result.iterations == 1
            assert result.residual == (0.0 if finite else np.inf)
            np.testing.assert_array_equal(result.state.messages, candidate.messages)
