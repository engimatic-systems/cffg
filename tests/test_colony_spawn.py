"""Parameter-only spawning shares a compiler plan and keeps fresh validation."""

from copy import deepcopy
from dataclasses import replace

import jax
import numpy as np
import pytest

from cffg import compiler, inits, kernels
from cffg.colony import Colony
from cffg.core import BlobRef, Expose, PlateSpec, Template
from cffg.core.serialize import structure_hash
from cffg.engine import Engine, RunStatus
from cffg.kernel_runtime import validate_gaussian_shapes

from test_compiler_contracts import model
from test_engine_contracts import assert_result_equal
from test_exact_colony import gaussian_goal_chain
from test_schedule import unit_categorical


def params_equal(state, reference):
    assert set(state.params) == set(reference.params)
    for name, value in state.params.items():
        np.testing.assert_array_equal(value, reference.params[name], err_msg=name)


def test_reuses_gaussian_plan_and_abstract_checks_without_skipping_numeric_validation(monkeypatch):
    with jax.enable_x64(False):
        template, first, observations = gaussian_goal_chain()
        second = {**first, "mean": np.array([0.6]), "process_cov": np.array([[0.18]])}
        expected = [
            _independent_run(template, arrays, {"g1": y})
            for arrays, y in ((first, [0.5]), (second, [1.4]))
        ]
        invalid = {**second, "process_cov": np.zeros((1, 1), dtype=np.float32)}
        with pytest.raises(compiler.CompileError) as fresh_error:
            compiler.compile_template(template, arrays=invalid)

        colony = Colony()
        first_id = colony.spawn(template, first, observations={"g1": [0.5]})

        def forbidden(*args, **kwargs):
            pytest.fail("a cached spawn reran a structural compiler stage")

        for name in ("expand_plates", "build_topology", "assemble_arena", "snapshot_registry"):
            monkeypatch.setattr(compiler, name, forbidden)
        monkeypatch.setattr("cffg.kernel_runtime.validate_gaussian_shapes", forbidden)
        with pytest.raises(compiler.CompileError) as reused_error:
            colony.spawn(template, invalid, observations=observations)
        assert str(reused_error.value) == str(fresh_error.value)

        second_id = colony.spawn(template, second)
        assert (first_id, second_id) == (0, 1)
        assert len(colony.engines) == 1
        # Array overrides may still belong to the caller after a spawn.
        second["process_cov"][0, 0] = -99
        colony.bind(second_id, {"g1": [1.4]})
        for result, fresh in zip(colony.run(5, tol=1e-6).values(), expected, strict=True):
            assert_result_equal(result, fresh)


def _independent_run(template, arrays, observations):
    engine = Engine(compiler.compile_template(template, arrays=arrays))
    return engine.run(engine.bind_data(engine.init_state(), observations), 5, tol=1e-6)


@pytest.mark.parametrize("name", [
    "hmm", "ef", "gfe", "mixture", "recurrence", "learning",
    "parameter_edge", "hybrid_gfe", "constraint",
])
def test_cached_sources_reuse_all_family_packers(name):
    template, arrays = model(name)
    fresh = Engine(compiler.compile_template(template, arrays=arrays)).init_state()
    colony = Colony()
    colony.spawn(template, arrays)
    other = colony.spawn(deepcopy(template), arrays)
    assert len(colony.engines) == 1
    params_equal(colony.instances[other].state, fresh)


def test_plated_initializers_are_read_again_without_rebuilding_the_graph(monkeypatch):
    fragment = unit_categorical()
    fragment.blobs["p"] = BlobRef("p", "local-{index}.npy", inits.spec("one_hot", index=0))
    template = Template("scoped", plates={"p": PlateSpec(
        fragment, (3,), (Expose(),), per_copy_blobs=("p",),
    )})
    changed = deepcopy(template)
    changed.plates["p"].fragment.blobs["p"] = BlobRef(
        "p", "relocated-{index}.npy", inits.spec("one_hot", index=1)
    )
    assert structure_hash(template) == structure_hash(changed)
    override = {"p/1/p": np.array([0., 0., 1.])}
    original = Engine(compiler.compile_template(template)).init_state()
    fresh = Engine(compiler.compile_template(changed, arrays=override)).init_state()
    colony = Colony()
    first = colony.spawn(template, {})

    def forbidden(*args, **kwargs):
        pytest.fail("same-structure plate expanded twice")

    monkeypatch.setattr(compiler, "lower_plates", forbidden)
    second = colony.spawn(changed, override)
    params_equal(colony.instances[first].state, original)
    params_equal(colony.instances[second].state, fresh)
    assert len(colony.engines) == 1


def test_same_version_callback_replacement_gets_a_new_executable(monkeypatch):
    template = unit_categorical()
    arrays = {"p": np.array([0.1, 0.2, 0.7])}
    colony = Colony()
    first = colony.spawn(template, arrays)
    original = kernels.KERNELS["categorical"]
    pack = original.pack_params

    def reversed_weights(*args):
        values = pack(*args)
        return {"logf": values["logf"][:, ::-1].copy()}

    alternate = replace(original, binding=replace(original.binding, pack_params=reversed_weights))
    monkeypatch.setitem(kernels.KERNELS, "categorical", alternate)
    fresh = Engine(compiler.compile_template(template, arrays=arrays)).init_state()
    second = colony.spawn(template, arrays)
    assert len(colony.engines) == 2
    assert colony.instances[first].key != colony.instances[second].key
    params_equal(colony.instances[second].state, fresh)
    result = colony.run(2, tol=1e-6)
    assert all(row.status is RunStatus.CONVERGED for row in result.values())
    assert np.argmax(colony.marginals(first)["x"]) == 2
    assert np.argmax(colony.marginals(second)["x"]) == 0


def test_shape_validation_is_repeated_in_a_different_precision_context(monkeypatch):
    template, arrays, _ = gaussian_goal_chain()
    colony = Colony()
    seen = []

    def capture(plan, packed):
        seen.append(jax.config.x64_enabled)
        return validate_gaussian_shapes(plan, packed)

    monkeypatch.setattr("cffg.kernel_runtime.validate_gaussian_shapes", capture)
    with jax.enable_x64(False):
        first = colony.spawn(template, arrays)
    with jax.enable_x64(True):
        second = colony.spawn(template, arrays)
    assert False in seen and True in seen
    assert colony.instances[first].state.g_eta.dtype == np.dtype("float32")
    assert colony.instances[second].state.g_eta.dtype == np.dtype("float64")
    assert len(colony.engines) == 2


def test_new_packed_layout_is_checked_even_under_the_same_template(monkeypatch):
    template = unit_categorical()
    original = kernels.KERNELS["categorical"]
    pack = original.pack_params

    def variable_layout(template, members, resolved):
        result = pack(template, members, resolved)
        if resolved["p"][0] > .5:
            return {"logf": result["logf"][:, :2]}
        return result

    monkeypatch.setitem(
        kernels.KERNELS, "categorical",
        replace(original, binding=replace(original.binding, pack_params=variable_layout)),
    )
    colony = Colony()
    valid = {"p": np.array([.2, .3, .5])}
    invalid = {"p": np.array([.7, .2, .1])}
    assert colony.spawn(template, valid) == 0
    with pytest.raises(compiler.CompileError, match="log-potential shape") as fresh:
        compiler.compile_template(template, arrays=invalid)
    with pytest.raises(compiler.CompileError) as cached:
        colony.spawn(template, invalid)
    assert str(cached.value) == str(fresh.value)
    assert colony.spawn(template, valid) == 1
    assert len(colony.engines) == 1


def test_canonical_hash_does_not_bypass_invalid_declarations():
    template = unit_categorical()
    arrays = {"p": np.array([.2, .3, .5])}
    malformed = deepcopy(template)
    factor = next(iter(malformed.factors.values()))
    malformed.factors["duplicate"] = factor
    assert structure_hash(malformed) == structure_hash(template)
    colony = Colony()
    colony.spawn(template, arrays)
    with pytest.raises(ValueError, match="fully connected"):
        colony.spawn(malformed, arrays)
    assert len(colony) == 1 and len(colony.engines) == 1


def test_distinguishes_nonlinear_approximations_and_reuses_current_theta():
    from nonlinear_fixture import adaptive_pair

    linear, arrays = adaptive_pair(method="linearization")
    sigma, other = adaptive_pair(method="sigma_point")
    colony = Colony()
    a = colony.spawn(linear, arrays)
    b = colony.spawn(sigma, other)
    changed = {**arrays, "theta": np.array([.45])}
    c = colony.spawn(linear, changed)
    assert len(colony.engines) == 2
    assert colony.instances[a].key == colony.instances[c].key != colony.instances[b].key
    engine = Engine(compiler.compile_template(linear, arrays=changed))
    params_equal(colony.instances[c].state, engine.init_state())


def test_shared_plate_scope_rebinds_current_initializers_without_relowering(monkeypatch):
    from cffg.core import LearnableSpec
    from cffg.core.compose import ComposeError, scoped
    from cffg.core.plates import expand_plates
    from cffg.core.serialize import from_json, to_json
    from test_plates import pixel_fragment

    def model(seed):
        fragment = pixel_fragment(seed)
        fragment.learnables = {
            "shared": LearnableSpec("shared", "dirichlet", ("A",)),
            "private": LearnableSpec("private", "dirichlet", ("d",), ("prior",)),
        }
        spec = PlateSpec(fragment, (2,), per_copy_blobs=("d",))
        return scoped(Template("retina", plates={"left": spec, "right": spec}), "view/eye")

    colony = Colony()
    first = colony.spawn(model(11), {})
    first_params = {key: np.asarray(value).copy()
                    for key, value in colony.instances[first].state.params.items()}

    def unexpected_lowering(*args, **kwargs):
        pytest.fail("cached plate spawn repeated structural lowering")

    monkeypatch.setattr("cffg.colony._prepare_compilation", unexpected_lowering)
    incoming = from_json(to_json(model(23)))
    arrays = {"view/eye/right/1/d": np.array([0.0, 0.75, 0.25])}
    obs = {"view/eye/left/0/x": 1}
    iid = colony.spawn(incoming, arrays, observations=obs)
    assert len(colony.engines) == 1
    # A tied path disagreement must fail even on a hit. Source data is
    # excluded from the structure hash and inspected on each spawn.
    conflict = model(23)
    fragment = conflict.plates["view/eye/right"].fragment
    fragment.blobs["A"] = replace(fragment.blobs["A"], path="other.npy")
    with pytest.raises(ComposeError, match="conflicting declarations for tied blob"):
        colony.spawn(conflict, {})
    assert len(colony) == 2
    assert len(colony.engines) == 1

    fresh = Engine(compiler.compile_template(expand_plates(incoming), arrays=arrays))
    reference = fresh.bind_data(fresh.init_state(), obs)
    state = colony.instances[iid].state
    params_equal(state, reference)
    for key, original in first_params.items():
        np.testing.assert_array_equal(colony.instances[first].state.params[key], original)
    np.testing.assert_array_equal(state.messages, reference.messages)
    assert any(not np.array_equal(value, state.params[key]) for key, value in first_params.items())
    assert next(iter(colony.engines.values())).arena.learn_plans == fresh.arena.learn_plans
