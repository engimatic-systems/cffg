"""One cached structure must not substitute another graph's constraint data."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_engine_contracts import assert_result_equal, assert_state_equal

from cffg.colony import Colony
from cffg.compiler import CompileError, compile_template
from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.serialize import structure_hash
from cffg.engine import Engine, RunStatus, _replace
from cffg.resolution import resolve as resolution


def constraint_model(kind, value, cardinality=2):
    role = "table" if kind == "table" else "k"
    template = Template(
        "constraint-data",
        variables={name: VariableSpec(name, cardinality) for name in ("a", "b")},
        factors={
            "limit": FactorSpec(
                "limit", kind, {"x0": "a", "x1": "b"}, param_blobs={role: "payload"}
            )
        },
        ports={name: PortSpec(name, name, PortRole.DATA) for name in ("a", "b")},
        blobs={"payload": BlobRef("payload", "payload.npy")},
    )
    return template, {"payload": np.asarray(value)}


def examples(kind):
    return ([[0, 0], [1, 1]], [[0, 1], [1, 0]]) if kind == "table" else (0, 1)


def cast_state(state, dtype):
    return jax.tree_util.tree_map(
        lambda value: (
            value.astype(dtype) if jnp.issubdtype(value.dtype, jnp.floating) else value
        ),
        state,
    )


def observe_batches(monkeypatch):
    batches = []
    original = eqx.filter_vmap

    def observe(function, *args, **kwargs):
        mapped = original(function, *args, **kwargs)
        if function.__name__ != "one":
            return mapped

        def execute(state):
            batches.append(state)
            return mapped(state)

        return execute

    monkeypatch.setattr(eqx, "filter_vmap", observe)
    return batches


@pytest.mark.parametrize("kind", ("exactly_k", "at_most_k", "table"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("observed", (False, True))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("schedule", ("auto", "flood"))
def test_distinct_constraint_instances_match_solo_in_both_orders(
    kind, reverse, observed, dtype, schedule, monkeypatch
):
    colony = Colony()
    ids, references = [], []
    values = examples(kind)[:: -1 if reverse else 1]
    observations = {"a": 0, "b": 1} if observed else {}
    for value in values:
        template, arrays = constraint_model(kind, value)
        engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
        state = cast_state(engine.bind_data(engine.init_state(), observations), dtype)
        references.append(engine.run(state, 8, tol=1e-7, harden_last=8))
        iid = colony.spawn(template, arrays, observations=observations)
        colony.instances[iid].state = cast_state(colony.instances[iid].state, dtype)
        ids.append(iid)
    assert len(colony.engines) == 1
    key = colony.instances[ids[0]].key
    assert key == structure_hash(template) == colony.instances[ids[1]].key
    colony.engines[key] = Engine(colony.engines[key].arena, schedule=schedule)
    batches = observe_batches(monkeypatch)
    actual = colony.run(8, tol=1e-7)
    for iid, expected in zip(ids, references, strict=True):
        assert_result_equal(actual[iid], expected)
        assert colony.instances[iid].state is actual[iid].state
    assert batches and {state.messages.shape[0] for state in batches} == {2}
    statuses = [actual[iid].status for iid in ids]
    if observed:
        assert set(statuses) == {RunStatus.INFEASIBLE, RunStatus.CONVERGED}
    else:
        assert statuses == [RunStatus.CONVERGED, RunStatus.CONVERGED]


def test_table_lengths_and_dtypes_partition_batches_without_new_engines(monkeypatch):
    small = np.array([[0, 0], [1, 1]])
    diagonal = np.column_stack((np.arange(10), np.arange(10)))
    large = np.concatenate((diagonal, (diagonal + [0, 1]) % 10))
    colony = Colony()
    references = {}
    for dtype in (np.float32, np.float64):
        for table in (small, large, small[::-1, ::-1], large[::-1, ::-1]):
            template, arrays = constraint_model("table", table, cardinality=10)
            engine = Engine(compile_template(template, arrays=arrays))
            state = cast_state(engine.init_state(), dtype)
            iid = colony.spawn(template, arrays)
            colony.instances[iid].state = cast_state(colony.instances[iid].state, dtype)
            references[iid] = engine.run(state, 8, tol=1e-7, harden_last=8)
    assert len(colony.engines) == 1
    batches = observe_batches(monkeypatch)
    actual = colony.run(8, tol=1e-7)
    for iid, expected in references.items():
        assert_result_equal(actual[iid], expected)
    assert batches and {state.messages.shape[0] for state in batches} == {2}
    key = next(iter(colony.engines.values())).arena.constraint_groups[0].parameter_key
    assert {(state.messages.dtype, state.params[key].shape) for state in batches} == {
        (np.dtype(dtype), (2, rows, 2))
        for dtype in (np.float32, np.float64)
        for rows in (2, 20)
    }


@pytest.mark.parametrize("kind", ("exactly_k", "at_most_k", "table"))
@pytest.mark.parametrize("semiring", ("sum", "max"))
def test_update_feasibility_and_objective_read_the_current_state_data(kind, semiring):
    engines, states = [], []
    for value in examples(kind):
        template, arrays = constraint_model(kind, value)
        engine = Engine(compile_template(template, arrays=arrays, semiring=semiring))
        state = engine.init_state()
        messages = state.messages
        for name, probabilities in (("a", [0.8, 0.2]), ("b", [0.3, 0.7])):
            indices = engine.arena.message_indices(engine.arena.data_slots[name], 2)
            messages = messages.at[indices].set(jnp.log(jnp.asarray(probabilities)))
        states.append(_replace(state, messages=messages))
        engines.append(engine)
    original, current = states
    # A shared program must accept another instance's same-shaped data directly.
    reused = _replace(original, params=current.params)
    assert_state_equal(engines[0].step(reused), engines[1].step(current))
    assert not np.array_equal(
        engines[0].step(original).messages, engines[0].step(reused).messages
    )
    if semiring == "sum":
        assert engines[0].bfe(reused) == engines[1].bfe(current)
        assert not np.isclose(engines[0].bfe(original), engines[0].bfe(reused))
    original = engines[0].bind_data(original, {"a": 0, "b": 1})
    reused = engines[0].bind_data(reused, {"a": 0, "b": 1})
    assert engines[0]._has_zero_support(original)
    assert not engines[0]._has_zero_support(reused)
    assert np.isinf(engines[0].objective(original))
    assert engines[0].objective(reused) == 0.0


def test_constraint_parameter_address_cannot_overwrite_another_family(monkeypatch):
    from test_compiler_contracts import model

    template, arrays = model("constraint")
    pack = resolution.pack_discrete

    def conflicting_pack(*args):
        return {**pack(*args), "constraint:c:k": np.asarray(99)}

    monkeypatch.setattr(resolution, "pack_discrete", conflicting_pack)
    with pytest.raises(CompileError, match="constraint parameter address collision"):
        compile_template(template, arrays=arrays)
