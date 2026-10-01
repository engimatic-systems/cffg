"""Constraint data survives safe model and agent-state reconstruction."""

import json
from functools import partial

import jax
import numpy as np
import pytest
from test_agent_mu3 import ALPHA0, assert_result_exact, binary_builder
from test_agent_spec import _two_block_builder
from test_constraint_data import constraint_model, examples

from cffg._agent_checkpoint import _planning_inputs
from cffg.agent import RollingAgent
from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core.serialize import structure_hash
from cffg.engine import Engine


def constrained_builder(*, kind, value, builder=_two_block_builder, **kwargs):
    template, arrays = builder(**kwargs)
    constraint, data = constraint_model(kind, value)
    template.variables.update(constraint.variables)
    template.factors.update(constraint.factors)
    template.ports.update(constraint.ports)
    template.blobs.update(constraint.blobs)
    return template, {**arrays, **data}


def agent_for(kind, value):
    return RollingAgent(
        partial(constrained_builder, kind=kind, value=value),
        horizon=1,
        n_sweeps=8,
        select="map",
    )


@pytest.mark.parametrize("kind", ("exactly_k", "at_most_k", "table"))
def test_bundle_rebuilds_structural_plan_and_numeric_constraint_data(kind, tmp_path):
    value = examples(kind)[1]
    template, arrays = constraint_model(kind, value)
    path = tmp_path / "constraint.cffgz"
    save_bundle(path, template, arrays)
    restored, data, _manifest = load_bundle(path)
    assert structure_hash(restored) == structure_hash(template)
    engine = Engine(compile_template(restored, arrays=data))
    plan = engine.arena.constraint_groups[0]
    assert not hasattr(plan, "k") and not hasattr(plan, "table")
    np.testing.assert_array_equal(engine.init_state().params[plan.parameter_key], value)
    state = engine.bind_data(engine.init_state(), {"a": 0, "b": 1})
    assert not engine._has_zero_support(state)


@pytest.mark.parametrize("kind", ("exactly_k", "at_most_k", "table"))
@pytest.mark.parametrize("version", ("current", "agent_v1", "legacy"))
def test_analytic_checkpoints_reconstruct_constraint_data(kind, version, tmp_path):
    value = examples(kind)[1]
    original = agent_for(kind, value)
    original.beliefs = {
        name: np.full_like(belief, 1 / belief.size)
        for name, belief in original.beliefs.items()
    }
    current = tmp_path / "current.npz"
    original.checkpoint(current)
    path = current
    if version != "current":
        with np.load(current, allow_pickle=False) as archive:
            fields = {
                name: archive[name].copy()
                for name in archive.files
                if name in {"tick", "episode_tick", "rng_state"}
                or name.startswith("belief:")
            }
        if version == "agent_v1":
            fields.update(
                {"agent:version": np.asarray(1), "agent:mode": np.asarray("analytic")}
            )
        path = tmp_path / f"{version}.npz"
        np.savez(path, **fields)
    resumed = agent_for(kind, value)
    resumed.restore(path)
    for name, expected in original.beliefs.items():
        np.testing.assert_array_equal(resumed.beliefs[name], expected)
    state = _planning_inputs(
        resumed, resumed.learnables, resumed.beliefs, resumed.goal_priors
    )
    plan = resumed.plan.arena.constraint_groups[0]
    np.testing.assert_array_equal(state.params[plan.parameter_key], value)
    state = resumed.plan.bind_data(state, {"a": 0, "b": 1})
    assert not resumed.plan._has_zero_support(state)
    filter_plan = resumed.filter.arena.constraint_groups[0]
    np.testing.assert_array_equal(
        resumed.filter.init_state().params[filter_plan.parameter_key], value
    )


@pytest.mark.parametrize("kind", ("exactly_k", "at_most_k", "table"))
def test_strict_checkpoint_still_rejects_different_constraint_data(kind, tmp_path):
    first, second = examples(kind)
    original = agent_for(kind, first)
    path = tmp_path / "strict.npz"
    original.checkpoint(path)
    other = agent_for(kind, second)
    assert original._plan_structure_hash == other._plan_structure_hash
    before = {name: belief.copy() for name, belief in other.beliefs.items()}
    with pytest.raises(ValueError, match="parameter mismatch"):
        other.restore(path)
    for name, expected in before.items():
        np.testing.assert_array_equal(other.beliefs[name], expected)


def functional_agent(kind):
    return RollingAgent(
        partial(
            constrained_builder,
            kind=kind,
            value=examples(kind)[1],
            builder=binary_builder,
        ),
        horizon=1,
        learn_a=True,
        initial_counts=ALPHA0,
        mu3=True,
        mu3_key=jax.random.PRNGKey(31),
        mu3_samples=32,
        mu3_min_ess_fraction=0.01,
        n_sweeps=1,
        select="map",
    )


def prepared_checkpoint(kind, tmp_path):
    source = functional_agent(kind)
    initial = _planning_inputs(
        source, source.learnables, source.beliefs, source.goal_priors
    )
    source._planning.run(source.plan, initial, 1, tol=1e-10, run_options={})
    path = tmp_path / "prepared.npz"
    source.checkpoint(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return source, arrays, path


def omit_old_constraint_fields(source, arrays):
    keys = {plan.parameter_key for plan in source.plan.arena.constraint_groups}
    for key in keys:
        del arrays[f"mu3:state:param:{key}"]
    metadata = json.loads(str(arrays["mu3:state:metadata"]))
    metadata["params"] = [key for key in metadata["params"] if key not in keys]
    arrays["mu3:state:metadata"] = np.asarray(json.dumps(metadata))


def write_current(path, arrays):
    arrays["checkpoint_fields"] = np.asarray(json.dumps(sorted(arrays)))
    np.savez(path, **arrays)


@pytest.mark.parametrize("kind", (
    "exactly_k",
    pytest.param("at_most_k", marks=pytest.mark.slow),
    pytest.param("table", marks=pytest.mark.slow),
))
@pytest.mark.parametrize("envelope", ("current", "agent_v1"))
def test_prepared_mu3_cache_rebuilds_pre_constraint_parameter_layout(
    kind, envelope, tmp_path
):
    source, arrays, path = prepared_checkpoint(kind, tmp_path)
    restored = functional_agent(kind)
    restored.restore(path)
    assert_result_exact(restored.last_plan_result, source.last_plan_result)
    omit_old_constraint_fields(source, arrays)
    if envelope == "agent_v1":
        arrays = {
            key: value
            for key, value in arrays.items()
            if key in {"agent:mode", "tick", "episode_tick", "rng_state", "belief"}
            or key.startswith(("belief:", "learn:", "goal:", "mu3:"))
        }
        arrays["agent:version"] = np.asarray(1)
        np.savez(path, **arrays)
    else:
        write_current(path, arrays)
    restored.restore(path)
    assert_result_exact(restored.last_plan_result, source.last_plan_result)
    np.testing.assert_array_equal(
        restored.planning_key_cursor, source.planning_key_cursor
    )


@pytest.mark.parametrize(
    "corruption", ("declared_missing", "missing_message", "extra_array")
)
def test_constraint_cache_migration_does_not_relax_other_field_checks(
    corruption, tmp_path
):
    source, arrays, path = prepared_checkpoint("exactly_k", tmp_path)
    key = source.plan.arena.constraint_groups[0].parameter_key
    if corruption == "declared_missing":
        del arrays[f"mu3:state:param:{key}"]
    else:
        omit_old_constraint_fields(source, arrays)
        if corruption == "missing_message":
            del arrays["mu3:state:messages"]
        else:
            arrays["mu3:state:param:unknown"] = np.asarray(0)
    write_current(path, arrays)
    restored = functional_agent("exactly_k")
    with pytest.raises(KeyError, match="missing state|field inventory"):
        restored.restore(path)
    assert restored.last_plan_result is None
