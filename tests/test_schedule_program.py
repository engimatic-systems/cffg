"""Schedule composition boundaries; existing schedule tests own numerical oracles."""

import ast
import inspect
from dataclasses import FrozenInstanceError, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp
from test_ef_engine import gamma_poisson
from test_schedule import A, D, unit_categorical, vmp_unit_template

from cffg import schedule_program
from cffg.compiler import compile_template
from cffg.core import BlobRef, FactorSpec, Template, VariableSpec
from cffg.engine import Engine, _make_bfe, _replace
from cffg.execution.owners import SCHEDULED_EXECUTORS
from cffg.execution.synchronous import make_sweep
from cffg.schedule import ScanStage
from cffg.schedule_program import _batch_stage_fn, _flat_scan_fn, build_schedule_program


def program_arena(kind, semiring="sum"):
    if kind == "mixed":
        template, arrays = vmp_unit_template(), {"A": A, "d": D}
    elif kind == "colored":
        names = ("a", "b", "c")
        template = Template(
            "program-ring",
            variables={name: VariableSpec(name, 2) for name in names},
            factors={
                str(index): FactorSpec(
                    str(index),
                    "matrix_cat",
                    {"in": name, "out": names[(index + 1) % 3]},
                    param_blobs={"matrix": "B"},
                )
                for index, name in enumerate(names)
            },
            blobs={"B": BlobRef("B", "B")},
        )
        arrays = {"B": np.asarray([[0.8, 0.3], [0.2, 0.7]])}
    else:
        template, arrays = unit_categorical(), {"p": np.asarray([0.2, 0.3, 0.5])}
    return compile_template(template, arrays=arrays, semiring=semiring)


def state_args(state):
    return (
        state.messages,
        state.g_eta,
        state.g_lam,
        state.g_obs,
        state.g_mask,
        state.ef_eta,
        state.ef_obs,
        state.ef_mask,
        state.params,
    )


def state_in_dtype(engine, dtype):
    return jax.tree_util.tree_map(
        lambda value: value.astype(dtype) if value.dtype.kind == "f" else value,
        engine.init_state(),
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("kind", ("flood", "two_pass", "colored", "mixed"))
def test_program_composes_only_required_stages_and_preserves_facade(kind, dtype):
    arena = program_arena(kind)
    selection = "flood" if kind == "flood" else "auto"
    engine = Engine(arena, schedule=selection)
    assert engine._schedule_program.step is engine._step
    assert engine._schedule_program.iter_step is engine._iter_step
    requests = []

    def counted_sweep(received, skip_scheduled):
        assert received is arena
        requests.append(skip_scheduled)
        return make_sweep(received, skip_scheduled)

    program = build_schedule_program(
        arena,
        selection,
        SCHEDULED_EXECUTORS,
        make_sweep=counted_sweep,
        free_energy=_make_bfe(arena),
    )
    assert requests == ([False, True] if kind == "mixed" else [False])
    assert program.schedule.kind == kind
    assert program.schedule.describe() == engine.schedule.describe()
    assert program.schedule.fingerprint == engine.schedule.fingerprint
    assert program.scan_registerized == engine.scan_registerized
    assert (program.exact_unit is not None) == (kind == "two_pass")
    assert (program.propagate is not None) == (kind == "two_pass")
    assert (program.prime is not None) == bool(program.schedule.prime_stages)
    assert (program.reconstruct is not None) == (
        kind == "mixed" and bool(program.schedule.forest_stages)
    )
    if kind in ("two_pass", "flood"):
        assert program.iter_step is program.step
    with pytest.raises(FrozenInstanceError):
        program.step = None

    state = state_in_dtype(engine, dtype)
    args = state_args(state)
    actual = program.iter_step(*args, 0.25, False)
    reference = engine._iter_step(*args, 0.25, False)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True):
        assert left.shape == right.shape and left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("semiring", ("sum", "max"))
def test_exact_unit_uses_injected_ledger_only_when_defined(dtype, semiring):
    arena = program_arena("two_pass", semiring)
    engine = Engine(arena)
    calls = []

    def ledger(messages, *args):
        calls.append(None)
        return jnp.asarray(37.0, dtype=messages.dtype)

    program = build_schedule_program(
        arena,
        "auto",
        SCHEDULED_EXECUTORS,
        make_sweep=make_sweep,
        free_energy=ledger,
    )
    result = program.exact_unit(*state_args(state_in_dtype(engine, dtype)))
    assert len(result) == 10
    assert int(result[9].reason) == 0
    assert result[6].shape == () and result[6].dtype == np.dtype(dtype)
    assert result[7].dtype == result[8].dtype == np.dtype(bool)
    if semiring == "sum":
        assert len(calls) == 1 and float(result[6]) == 37.0
    else:
        assert not calls and np.isnan(result[6])


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_mixed_schedule_transports_inner_failure_facts_without_classifying(dtype):
    arena = program_arena("mixed")
    engine = Engine(arena)

    def marked_sweep(received, skip_scheduled):
        raw = make_sweep(received, skip_scheduled)

        def sweep(messages, *args):
            result = raw(messages, *args)
            return (
                *result[:4],
                jnp.asarray(0.125, dtype=messages.dtype),
                jnp.asarray(2),
                result[6],
            )

        return sweep

    program = build_schedule_program(
        arena,
        "auto",
        SCHEDULED_EXECUTORS,
        make_sweep=marked_sweep,
        free_energy=_make_bfe(arena),
    )
    result = program.iter_step(*state_args(state_in_dtype(engine, dtype)), 0.0, False)
    assert result[4].dtype == np.dtype(dtype)
    assert float(result[4]) == 0.125
    assert result[5].shape == () and int(result[5]) == 2


def test_schedule_source_does_not_own_family_selection_or_public_status():
    tree = ast.parse(inspect.getsource(schedule_program))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {
        "cffg.engine",
        "cffg.kernels",
        "cffg.kernel_runtime",
        "cffg.rules",
        "cffg.csp",
        "cffg.gauss",
        "cffg.efrules",
    }
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not attributes & {"type", "binding", "param_keys", "parameter_paths"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "RunStatus" not in names


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_ef_registerized_prefetch_uses_owner_and_reuses_dynamic_observations(dtype):
    template, arrays = gamma_poisson(32)
    arena = compile_template(template, arrays=arrays)
    engine = Engine(arena)
    selections = []

    def select(plan, params, rows):
        selections.append(plan.type)
        return SCHEDULED_EXECUTORS.ef_parameters(plan, params, rows)

    owners = replace(SCHEDULED_EXECUTORS, ef_parameters=select)
    program = build_schedule_program(
        arena, "auto", owners, make_sweep=make_sweep, free_energy=_make_bfe(arena)
    )
    assert program.scan_registerized and all(program.scan_registerized)
    initial = state_in_dtype(engine, dtype)
    trace_count = None
    previous = None
    for increment in (0, 1):
        counts = {
            f"count{i}": None if i % 7 == 0 else i % 5 + increment for i in range(32)
        }
        state = engine.bind_data(initial, counts)
        args = (
            state.messages,
            state.g_eta,
            state.g_lam,
            state.ef_eta,
            state.g_obs,
            state.g_mask,
            state.ef_obs,
            state.ef_mask,
            state.params,
        )
        actual = program.propagate(*args)
        if trace_count is None:
            trace_count = len(selections)
            assert {"gamma", "equality", "poisson_obs"} <= set(selections)
        else:
            assert len(selections) == trace_count
            assert not np.array_equal(actual[3], previous)
        previous = actual[3]

        flat = args[:4]
        for stage in program.schedule.stages:
            build = _flat_scan_fn if isinstance(stage, ScanStage) else _batch_stage_fn
            flat = eqx.filter_jit(build(arena, stage, logsumexp))(*flat[:4], *args[4:])
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(flat), strict=True):
            np.testing.assert_allclose(left, right, atol=64 * np.finfo(dtype).eps)

        result = engine.ef_marginals(_replace(state, ef_eta=actual[3]))
        observed = [value for value in counts.values() if value is not None]
        expected = np.asarray([2.0 + sum(observed), 1.5 + len(observed)], dtype=dtype)
        np.testing.assert_allclose(
            result["lam0"].params, expected, rtol=64 * np.finfo(dtype).eps
        )
