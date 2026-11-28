"""Indexed transition contractions and the public dense-replay boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.colony import Colony
from cffg.compiled import NEG
from cffg.compiled_iteration import CompiledIteration, controller_for
from cffg.execution.indexed_transition import _IndexedTransition
from cffg.models.rollout import build_rollout
from cffg.rules import bp_transition_mixture, energy_transition_mixture
from cffg.runtime_types import RunStatus, _replace


def _colony(density=1):
    card = 32
    eye = np.eye(card)
    transition = np.stack([
        sum(np.roll(eye, action + shift, axis=0) for shift in range(density)) / density
        for action in range(2)
    ])
    template, arrays = build_rollout(
        "indexed-transition", steps=2, past=1, d=np.r_[1., np.zeros(card - 1)],
        Bs=transition, e=np.array([.4, .6]), gfe=False,
        A=np.stack([np.linspace(.2, .8, card), np.linspace(.8, .2, card)]), c=np.array([.25, .75]),
    )
    colony = Colony()
    for lane in range(8):
        colony.spawn(template, arrays, observations={"act1": lane % 2, "obs1": lane // 2 % 2})
    engine, = colony.engines.values()
    return colony, engine


def _compare(reference, candidate, dtype):
    eps = np.finfo(dtype).eps
    for member, old in reference.items():
        new = candidate[member]
        assert new.status is old.status
        assert new.iterations == old.iterations and new.failure == old.failure
        a, b = np.asarray(old.state.messages), np.asarray(new.state.messages)
        assert a.dtype == b.dtype == dtype
        np.testing.assert_array_equal(a == dtype(NEG), b == dtype(NEG))
        np.testing.assert_allclose(a, b, rtol=64 * eps, atol=64 * eps)
        for attr in ("message_deltas", "free_energies", "residual"):
            np.testing.assert_allclose(getattr(old, attr), getattr(new, attr), rtol=256 * eps, atol=256 * eps)


def _run_from(colony, engine, states, *, dense):
    for row, state in zip(colony.instances.values(), states, strict=True):
        row.state = state
    controller = controller_for(engine)
    if dense:
        engine._compiled_iteration = CompiledIteration(engine, controller.hooks, allow_indexed=False)
    try:
        return colony.run(7, damp=.5, tol=None)
    finally:
        engine._compiled_iteration = controller


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_public_indexed_batch_and_live_parameter_replay(dtype, monkeypatch):
    with jax.enable_x64(dtype is np.float64):
        colony, engine = _colony()
        states = tuple(row.state for row in colony.instances.values())
        expected = _run_from(colony, engine, states, dense=True)
        assert all(run.status is RunStatus.BUDGET_EXHAUSTED for run in expected.values())
        controller = controller_for(engine)
        assert controller._indexed_batch(states) is not None
        indexed, branch = controller._indexed
        calls = []
        execute = branch._execute

        def record(*args, **kwargs):
            calls.append(True)
            return execute(*args, **kwargs)

        original = controller._execute
        dense_calls = []

        def record_dense(*args, **kwargs):
            dense_calls.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(branch, "_execute", record)
        monkeypatch.setattr(controller, "_execute", record_dense)
        actual = _run_from(colony, engine, states, dense=False)
        _compare(expected, actual, dtype)
        assert len(calls) == 1 and not dense_calls

        # A positive log cavity is legal for the dense rule but outside the
        # finite NEG-floor derivation of the indexed contraction.
        transition, = [g for g in engine.arena.groups if g.type == "transition_mixture"]
        row = engine.arena.message_indices(transition.gather[0, 0], transition.cards[0])
        hostile = states[:-1] + (_replace(states[-1], messages=states[-1].messages.at[row].set(1)),)
        expected = _run_from(colony, engine, hostile, dense=True)
        actual = _run_from(colony, engine, hostile, dense=False)
        _compare(expected, actual, dtype)
        assert len(calls) == 2 and len(dense_calls) == 1

        # A new array in one member must not inherit the old identity proof.
        path = dict(transition.parameter_paths)["logf"]
        params = {**states[-1].params, path: jnp.roll(states[-1].params[path], 1, axis=1)}
        changed = states[:-1] + (_replace(states[-1], params=params),)
        assert not indexed.matches(changed)
        expected = _run_from(colony, engine, changed, dense=True)
        actual = _run_from(colony, engine, changed, dense=False)
        _compare(expected, actual, dtype)
        assert len(calls) == 2 and len(dense_calls) == 2

        edge = engine.arena.edges["z0"]
        first = engine.arena.message_indices(edge.slot0, edge.card)
        second = engine.arena.message_indices(edge.slot1, edge.card)
        zero = jnp.full(edge.card, NEG, dtype=dtype)
        messages = states[-1].messages.at[first].set(zero.at[0].set(0)).at[second].set(zero.at[1].set(0))
        inconsistent = states[:-1] + (_replace(states[-1], messages=messages),)
        expected = _run_from(colony, engine, inconsistent, dense=True)
        actual = _run_from(colony, engine, inconsistent, dense=False)
        assert expected[max(expected)].status is RunStatus.INFEASIBLE
        _compare(expected, actual, dtype)
        assert len(calls) == 3 and len(dense_calls) == 3


@pytest.mark.parametrize("density", [8, 32])
def test_dense_or_low_sparsity_stays_in_the_stock_controller(density):
    colony, engine = _colony(density=density)
    states = tuple(row.state for row in colony.instances.values())
    controller = controller_for(engine)
    assert controller._indexed_batch(states) is None
    assert controller._indexed is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_indexed_floor_and_cutoff_boundary(dtype):
    with jax.enable_x64(dtype is np.float64):
        cards = (8, 8, 3)
        dense = np.full((1, *cards), NEG, dtype=dtype)
        dense[0, 0, 0, 0] = 0
        dense[0, 1, 1, 1] = -2
        plan = _IndexedTransition.pack(dense)
        assert plan is not None
        rows = jnp.zeros((1, 3, 8), dtype=dtype)
        rows = rows.at[0, 1, 1].set(dtype(NEG) / 2)
        actual = jax.jit(plan.messages)(rows)
        expected = jax.jit(lambda x: bp_transition_mixture(x, dense, cards, 8))(rows)
        np.testing.assert_array_equal(actual, expected)
        actual = jax.jit(plan.energy)(rows)
        expected = jax.jit(lambda x: energy_transition_mixture(x, dense, cards))(rows)
        np.testing.assert_array_equal(actual, expected)
        # A rounded probability at the existing atom cutoff must be replayed
        # before a regrouped normalization could change cutoff membership.
        probability = 1e-12
        rows = rows.at[0, 1, 1].set(np.log(probability) - np.log1p(-probability) + 2)
        assert np.isnan(plan.energy(rows)).all()
