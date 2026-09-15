"""Portable function identity, configured grouping, and source reconstruction."""

from dataclasses import replace

import jax
import numpy as np
import pytest
from nonlinear_fixture import model

from cffg.bundle import BundleError, load_bundle, save_bundle
from cffg.compiler import CompileError, compile_template
from cffg.core import BlobRef, FunctionSpec, validate
from cffg.core.compose import merge, scoped
from cffg.core.edits import AddFactor, action_from_dict, action_to_dict
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.engine import Engine
from cffg.estimate import estimate
from cffg.mean_functions import (
    MEAN_IMPLEMENTATIONS, MeanVersionError, MissingMeanImplementationError,
)


def test_json_bundle_and_edit_identity(tmp_path):
    template, arrays = model("sigma_point", observed=True)
    restored = from_json(to_json(template))
    assert structure_hash(restored) == structure_hash(template)
    action = AddFactor(template.factors["sensor"])
    assert action_from_dict(action_to_dict(action)) == action
    path = tmp_path / "sensor.cffgz"
    save_bundle(path, template, arrays)
    loaded = load_bundle(path)
    assert loaded.template.factors["sensor"].function == template.factors["sensor"].function
    for name, value in arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], value)
    arenas = [compile_template(t, arrays=a) for t, a in ((template, arrays), (loaded.template, loaded.arrays))]
    engines = [Engine(arena) for arena in arenas]
    states = [engine.step(engine.init_state()) for engine in engines]
    for a, b in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(engines[0].bfe(states[0]), engines[1].bfe(states[1]), atol=1e-12)


def test_missing_and_incompatible_implementations_fail_on_load(tmp_path, monkeypatch):
    template, arrays = model()
    text = to_json(template)
    path = tmp_path / "sensor.cffgz"
    save_bundle(path, template, arrays)
    implementation = MEAN_IMPLEMENTATIONS["cffg.mean.cubic"]
    monkeypatch.setitem(MEAN_IMPLEMENTATIONS, implementation.impl_id, replace(implementation, version="2.0.0"))
    with pytest.raises(MeanVersionError, match="incompatible"):
        from_json(text)
    monkeypatch.delitem(MEAN_IMPLEMENTATIONS, implementation.impl_id)
    with pytest.raises(MissingMeanImplementationError, match="missing"):
        from_json(text)
    with pytest.raises(BundleError) as error:
        load_bundle(path)
    assert isinstance(error.value.__cause__, MissingMeanImplementationError)


@pytest.mark.parametrize("change, message", [
    ("noise", "positive"), ("shape", "shape"), ("configuration", "configuration"),
])
def test_invalid_numerical_sources_or_configuration_fail_compile(change, message):
    template, arrays = model()
    if change == "noise":
        arrays["V"][:] = 0
    elif change == "shape":
        arrays["theta"] = np.ones(3)
    else:
        template.factors["sensor"] = replace(template.factors["sensor"], function=FunctionSpec(
            "cffg.mean.cubic", "1.0.0", configuration={"unexpected": True}))
    with pytest.raises(CompileError, match=message):
        compile_template(template, arrays=arrays)


def test_configuration_groups_and_member_values_are_separate():
    template, arrays = model()
    second = scoped(template, "b")
    combined = merge(scoped(template, "a"), second, "pair")
    values = {f"{prefix}/{name}": value for prefix in ("a", "b") for name, value in arrays.items()}
    values["b/theta"] = np.array([.6])
    validate(combined)
    arena = compile_template(combined, arrays=values)
    groups = [g for g in arena.gauss_groups if g.kernel.binding.iterative]
    assert len(groups) == 1 and len(groups[0].members) == 2
    state = Engine(arena).init_state()
    np.testing.assert_array_equal(state.params[groups[0].name + ":theta"], [[.3], [.6]])
    factor = combined.factors["b/sensor"]
    combined.factors["b/sensor"] = replace(factor, function=FunctionSpec("cffg.mean.cubic", "1.0.0", method="sigma_point"))
    regrouped = compile_template(combined, arrays=values)
    assert len([g for g in regrouped.gauss_groups if g.kernel.binding.iterative]) == 2


def test_persistent_storage_accounts_for_every_array():
    sizes = []
    for dim in (1, 32):
        template, arrays = model(dim=dim)
        engine = Engine(compile_template(template, arrays=arrays))
        state = engine.init_state()
        report = estimate(engine, calibrate=False)
        actual = sum(leaf.nbytes for leaf in jax.tree.leaves(state))
        assert report.state_bytes == actual
        assert actual == (6 * dim + 7 * dim * dim + 1) * state.g_eta.dtype.itemsize
        sizes.append(actual)
    assert sizes[1] <= 32 ** 2 * sizes[0]


def test_configured_shape_file_initializer_and_named_writes(tmp_path):
    from cffg import inits, set_params
    from cffg.compiler import _blob_shape

    template, arrays = model(affine=True)
    assert _blob_shape(template, "theta") == (2,)
    template.blobs["theta"] = BlobRef("theta", "theta.npy", inits.spec("zeros"))
    other_arrays = {k: v for k, v in arrays.items() if k != "theta"}
    initialized = compile_template(template, arrays=other_arrays)
    group = next(g for g in initialized.gauss_groups if g.kernel.binding.iterative)
    path = dict(group.parameter_paths)["theta"]
    np.testing.assert_array_equal(initialized.log_params[path], [[0., 0.]])
    np.save(tmp_path / "theta.npy", arrays["theta"])
    filed = compile_template(template, arrays=other_arrays, root=tmp_path)
    np.testing.assert_array_equal(filed.log_params[path], arrays["theta"][None])
    overridden = compile_template(template, arrays={**arrays, "theta": np.array([2., 1.])}, root=tmp_path)
    np.testing.assert_array_equal(overridden.log_params[path], [[2., 1.]])
    engine = Engine(overridden)
    params = engine.init_state().params
    updated = set_params(engine, params, {"sensor": {"parameters": np.array([3., 2.]), "cov": np.array([[.5]])}})
    np.testing.assert_array_equal(updated[path], [[3., 2.]])
    np.testing.assert_array_equal(updated[dict(group.parameter_paths)["W"]], [[[2.]]])
    np.testing.assert_array_equal(params[path], [[2., 1.]])
    with pytest.raises(ValueError, match="finite"):
        set_params(engine, params, {"sensor": {"parameters": np.array([np.nan, 0.])}})


def test_function_configuration_is_immutable_and_hash_relevant():
    original = {"architecture": [2, 3], "activation": "tanh"}
    spec = FunctionSpec("example.mean.network", "1.0.0", configuration=original)
    same = FunctionSpec("example.mean.network", "1.0.0", configuration=dict(reversed(list(original.items()))))
    assert spec == same
    original["architecture"].append(4)
    spec.configuration["architecture"].append(5)
    assert spec == same
    assert spec != FunctionSpec(spec.impl_id, "1.0.1", configuration=same.configuration)
    assert spec != FunctionSpec(spec.impl_id, spec.version, method="sigma_point", configuration=same.configuration)
