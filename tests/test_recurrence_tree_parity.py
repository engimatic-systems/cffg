"""A declared decision list preserves the existing recurrence execution."""

from copy import deepcopy
from dataclasses import replace

import jax
import numpy as np

from cffg.colony import Colony
from cffg.compiler import NEG, compile_template
from cffg.core.recurrence_tree import RecurrenceTree
from cffg.core.serialize import from_dict, from_json, to_dict, to_json
from cffg.engine import Engine, RunStatus
from cffg.experiments.recurrence_smoothing import (
    _case,
    forward_backward,
    probability_tolerance,
    recurrence_chain_template,
    stick_transition,
)
from test_engine_contracts import record_compiled_batches
from test_recurrence_link import _template


def _tree_variants(template, card):
    explicit = deepcopy(template)
    tree = RecurrenceTree.decision_list(card)
    for name, factor in explicit.factors.items():
        if factor.type == "recurrence_link":
            explicit.factors[name] = replace(factor, recurrence_tree=tree)
    round_trip = from_json(to_json(explicit))

    old_document = to_dict(explicit)
    old_document["schema_version"] = "1"
    for factor in old_document["factors"].values():
        factor.pop("recurrence_tree", None)
    return round_trip, from_dict(old_document)


def _assert_stacks(expected, actual):
    for a, b in zip(expected.recurrence_groups, actual.recurrence_groups, strict=True):
        assert (a.name, a.members, a.cards, a.dim) == (b.name, b.members, b.cards, b.dim)
        for field in ("d_gather_both", "d_scatter", "drive_gather_both", "drive_scatter", "component_ids"):
            np.testing.assert_array_equal(getattr(b, field), getattr(a, field))
        for suffix in ("weights", "offsets"):
            key = f"{a.name}:{suffix}"
            np.testing.assert_array_equal(actual.log_params[key], expected.log_params[key])


def _assert_beliefs(expected_engine, expected, actual_engine, actual, *, atol):
    assert actual.status is expected.status is RunStatus.CONVERGED
    assert actual.inner_status is expected.inner_status is RunStatus.CONVERGED
    assert actual.failure is expected.failure is None
    messages = np.asarray(actual.state.messages)
    baseline = np.asarray(expected.state.messages)
    assert messages.dtype == baseline.dtype
    zero = messages.dtype.type(NEG)
    np.testing.assert_array_equal(messages == zero, baseline == zero)
    np.testing.assert_allclose(messages, baseline, atol=atol, rtol=atol)
    for field in ("g_eta", "g_lam"):
        reference = getattr(expected.state, field)
        values = getattr(actual.state, field)
        assert values.dtype == reference.dtype
        np.testing.assert_allclose(values, reference, atol=atol, rtol=atol)

    actual_marginals = actual_engine.marginals(actual.state)
    expected_marginals = expected_engine.marginals(expected.state)
    assert actual_marginals.keys() == expected_marginals.keys()
    for edge, reference in expected_marginals.items():
        np.testing.assert_allclose(actual_marginals[edge], reference, atol=atol)

    expected_pairs = expected_engine.recurrence_pair_beliefs(expected.state)
    actual_pairs = actual_engine.recurrence_pair_beliefs(actual.state)
    assert expected_pairs.keys() == actual_pairs.keys()
    for name, row in expected_pairs.items():
        assert row["proper"] and actual_pairs[name]["proper"]
        for key, reference in row.items():
            values = actual_pairs[name][key]
            if reference.dtype == bool:
                np.testing.assert_array_equal(values, reference)
            else:
                assert np.isfinite(values).all()
                np.testing.assert_allclose(values, reference, atol=atol, rtol=atol)

    np.testing.assert_allclose(actual.free_energies[-1], expected.free_energies[-1], rtol=atol, atol=atol)
    np.testing.assert_allclose(actual_engine.bfe(actual.state), expected_engine.bfe(expected.state), rtol=atol, atol=atol)
    return actual_pairs


def test_decision_list_reconstruction_matches_uncertain_drive():
    _, _, _, weights, offsets = _case(3)
    template, arrays = _template(
        weights=weights,
        offsets=offsets,
        p_in=np.array([0.5, 0.3, 0.2]),
        p_out=np.array([0.12, 0.68, 0.20]),
        mean=np.array([0.3]),
        covariance=np.array([[0.7]]),
    )
    explicit, migrated = _tree_variants(template, 3)
    old_arena = compile_template(template, arrays=arrays)
    explicit_arena = compile_template(explicit, arrays=arrays)
    migrated_arena = compile_template(migrated, arrays=arrays)
    _assert_stacks(old_arena, explicit_arena)
    _assert_stacks(old_arena, migrated_arena)

    # Compare both reconstructed sources using the same public mixed schedule.
    reference, engine = Engine(migrated_arena), Engine(explicit_arena)
    expected = reference.run(reference.init_state(), 80, tol=1e-10)
    actual = engine.run(engine.init_state(), 80, tol=1e-10)
    atol = 256 * np.finfo(np.float64).eps
    pairs = _assert_beliefs(reference, expected, engine, actual, atol=atol)
    assert pairs["recurrence"]["drive_covariance"][0, 0] > 0.1
    assert actual.iterations == expected.iterations
    np.testing.assert_allclose(actual.message_deltas, expected.message_deltas, atol=atol, rtol=atol)
    np.testing.assert_allclose(actual.free_energies, expected.free_energies, atol=atol, rtol=atol)


def test_decision_list_float32_chain_preserves_support_in_flood_and_colony(monkeypatch):
    dtype = np.float32
    initial, evidence, drives, weights, offsets = _case(3, dtype=dtype)
    initial[:] = [1.0, 0.0, 0.0]
    evidence[-1] = [0.21, 0.0, 0.79]
    template, arrays, observations = recurrence_chain_template(
        initial, evidence, drives, weights, offsets, dtype=dtype,
    )
    explicit, migrated = _tree_variants(template, 3)
    alternate = {**observations, "drive1": observations["drive1"] - dtype(0.8)}
    atol = probability_tolerance(dtype, 3)
    with jax.enable_x64(False):
        reference = Engine(compile_template(migrated, arrays=arrays))
        flood = Engine(compile_template(explicit, arrays=arrays), schedule="flood")
        _assert_stacks(reference.arena, flood.arena)
        assert reference.schedule.kind == "mixed"
        assert flood.schedule.kind == "flood"
        expected = [reference.run(reference.bind_data(reference.init_state(), obs), 80, tol=atol / 8)
                    for obs in (observations, alternate)]
        flooded = flood.run(flood.bind_data(flood.init_state(), observations), 80, tol=atol / 8)
        _assert_beliefs(reference, expected[0], flood, flooded, atol=atol)

        colony = Colony()
        ids = [
            colony.spawn(template, arrays, observations=observations),
            colony.spawn(explicit, arrays, observations=alternate),
        ]
        colony_engine, = colony.engines.values()
        batches = record_compiled_batches(monkeypatch)
        outcomes = colony.run(80, tol=atol / 8)
        assert [(len(states), budget) for states, budget in batches] == [(2, 80)]
        for iid, baseline, obs in zip(ids, expected, (observations, alternate), strict=True):
            result = outcomes[iid]
            assert result.state.messages.dtype == dtype
            pairs = _assert_beliefs(reference, baseline, colony_engine, result, atol=atol)
            np.testing.assert_array_equal(pairs["switch0"]["pair_support"][1:], False)
            np.testing.assert_array_equal(pairs["switch0"]["pair_probability"][1:], 0.0)
            np.testing.assert_array_equal(pairs["switch2"]["pair_support"][:, 1], False)
            transitions = np.stack([stick_transition(obs[f"drive{i}"], weights, offsets) for i in range(3)])
            oracle, _ = forward_backward(initial, evidence, transitions)
            np.testing.assert_allclose(
                np.stack([pairs[f"switch{i}"]["pair_probability"] for i in range(3)]),
                oracle.consecutive, atol=atol,
            )
