"""Consumer-level next-trial and one-sequence resume proofs for every family."""

import io
import json

import equinox as eqx
import jax
import numpy as np
import pytest

from cffg import conj, load_bundle, save_bundle
from cffg.bundle import BundleError
from cffg.compiler import compile_template
from cffg.core import PlateSpec, PortRole, PortSpec, Template
from cffg.core.compose import scoped
from cffg.core.plates import expand_plates
from cffg.engine import Engine
from cffg.learn import FAMILIES, Learnables
from cffg.models.rslds_fit import vague_rslds_fit_priors

from test_bundle import _write_zip, _zip_contents
from test_gaussian_learnables import _learning_template
from test_hybrid_learnables import _mixture_template
from test_learning_schema import _categorical
from test_recurrence_checkpoint import _model


def _same_arrays(actual, expected):
    assert set(actual) == set(expected)
    for name, value in actual.items():
        assert value.dtype == expected[name].dtype, name
        assert value.shape == expected[name].shape, name
        np.testing.assert_array_equal(value, expected[name], err_msg=name)


def _same_learning(actual, expected):
    assert set(actual.plans) == set(expected.plans)
    for name, plan in actual.plans.items():
        _same_arrays(actual.initial_params[name], expected.initial_params[name])
        _same_arrays(actual.params[name], expected.params[name])
        family = FAMILIES[plan.family]
        for method in ("mean", "rollover"):
            _same_arrays(getattr(family, method)(actual.params[name], plan.ftype),
                         getattr(family, method)(expected.params[name], plan.ftype))
        for role in family.mean(actual.params[name], plan.ftype):
            _same_arrays({role: actual.expected(name, role)}, {role: expected.expected(name, role)})
            _same_arrays({role: actual.rollover(name, role)}, {role: expected.rollover(name, role)})
    assert actual.persistent_descriptors() == expected.persistent_descriptors()


def _fixture(kind, dtype):
    if kind == "rslds":
        template, arrays = _model(dtype)
        priors = vague_rslds_fit_priors(2, 1, 1).learning_priors()
        first = {f"y{step}": np.array([value], dtype=dtype) for step, value in enumerate((0.3, -0.25, 0.6))}
        later = {f"y{step}": np.array([value], dtype=dtype) for step, value in enumerate((-0.1, 0.45, 0.2))}
    else:
        template, arrays = _learning_template(2)
        priors = {"a_state": conj.vague_niw(2), "z_dynamics": conj.vague_mniw(2, 2)}
        first = {"observation": np.array([0.2, -0.3], dtype=dtype)}
        later = {"observation": np.array([-0.5, 0.7], dtype=dtype)}
    namespace = "consumer/subject%1"
    template = scoped(template, namespace)
    arrays = {f"{namespace}/{key}": value.astype(dtype) for key, value in arrays.items()}
    priors = {f"{namespace}/{key}": value for key, value in priors.items()}
    first = {f"{namespace}/{key}": value for key, value in first.items()}
    later = {f"{namespace}/{key}": value for key, value in later.items()}
    engine = Engine(compile_template(template, arrays=arrays))
    initial = Learnables(engine, priors=priors)
    values = {name: {key: value.astype(dtype) for key, value in params.items()}
              for name, params in initial.initial_params.items()}
    learning = Learnables.from_persistent_state(engine, values, values)
    return template, arrays, engine, learning, first, later


def _run(engine, data, learning=None):
    state = engine.init_state()
    if learning is not None:
        params = learning.write(engine, state.params)
        state = eqx.tree_at(lambda state: state.params, state, params)
    state = engine.bind_data(state, data)
    result = engine.run(state, 300, tol=1e-5 if state.g_eta.dtype == np.float32 else 1e-9)
    result.require_converged()
    return result


@pytest.mark.parametrize("cadence,dtype,kind", [
    pytest.param("accumulate", np.float32, "rslds", marks=pytest.mark.slow),
    ("accumulate", np.float32, "gaussian"),
    pytest.param("accumulate", np.float64, "rslds", marks=pytest.mark.slow),
    ("accumulate", np.float64, "gaussian"),
    ("replace", np.float32, "rslds"),
    ("replace", np.float32, "gaussian"),
    ("replace", np.float64, "rslds"),
    ("replace", np.float64, "gaussian"),
])
def test_all_families_resume_identical_next_update(tmp_path, kind, dtype, cadence):
    with jax.enable_x64(dtype == np.float64):
        template, arrays, engine, learning, first, later = _fixture(kind, dtype)
        path = tmp_path / "initial.cffgz"
        save_bundle(path, template, arrays, learnables=learning)
        initial = load_bundle(path)
        _same_learning(initial.learnables, learning)
        _same_arrays(initial.arrays, arrays)
        first_result = _run(engine, first)
        learning.accumulate(engine, first_result)
        initial.learnables.accumulate(engine, first_result)
        _same_learning(initial.learnables, learning)

        save_bundle(path, template, arrays, learnables=learning)
        resumed = load_bundle(path)
        restored_engine = Engine(compile_template(resumed.template, arrays=resumed.arrays))
        _same_learning(resumed.learnables, learning)
        _same_arrays(resumed.learnables.write(restored_engine, restored_engine.init_state().params),
                     learning.write(engine, engine.init_state().params))
        later_result = _run(restored_engine, later, resumed.learnables)
        getattr(learning, cadence)(engine, later_result)
        getattr(resumed.learnables, cadence)(restored_engine, later_result)
        _same_learning(resumed.learnables, learning)
        installed = resumed.learnables.write(restored_engine, restored_engine.init_state().params)
        _same_arrays(installed, learning.write(engine, engine.init_state().params))
        assert all(value.dtype == dtype for value in installed.values())
        assert all(np.all(np.isfinite(value)) for value in installed.values())
        if cadence == "replace":
            fresh = Learnables.from_persistent_state(engine, learning.initial_params, learning.initial_params)
            fresh.replace(engine, later_result)
            _same_learning(resumed.learnables, fresh)
        resaved = tmp_path / "later.cffgz"
        save_bundle(resaved, resumed.template, resumed.arrays, learnables=resumed.learnables)
        _same_learning(load_bundle(resaved).learnables, learning)


def test_workflow_fixtures_cover_the_runtime_family_registry():
    covered = set()
    for kind in ("rslds", "gaussian"):
        _, _, _, learning, _, _ = _fixture(kind, np.float64)
        covered.update(plan.family for plan in learning.plans.values())
    assert covered == set(FAMILIES)


@pytest.mark.parametrize("cadence", ("accumulate", "replace"))
def test_sparse_component_statistics_remain_exact_after_resume(tmp_path, cadence):
    template, arrays, sources = _mixture_template()
    engine = Engine(compile_template(template, arrays=arrays))
    learning = Learnables(engine, priors={"neutral_dynamics": tuple(conj.vague_mniw(2, 2) for _ in range(3))})
    first = engine.run(engine.init_state(), 100, tol=1e-9)
    first.require_converged()
    learning.accumulate(engine, first)
    path = tmp_path / "sparse.cffgz"
    save_bundle(path, template, arrays, learnables=learning)
    resumed = load_bundle(path).learnables
    stats = engine.gaussian_node_stats(first.state)
    family = FAMILIES["component_mniw"]
    for source in sources:
        for value in family.stats("gaussian_mixture_linear", stats[source]).values():
            np.testing.assert_array_equal(value[-1], np.zeros_like(value[-1]))
    getattr(learning, cadence)(engine, first)
    getattr(resumed, cadence)(engine, first)
    _same_learning(resumed, learning)
    for key, value in resumed.params["neutral_dynamics"].items():
        np.testing.assert_array_equal(value[-1], resumed.initial_params["neutral_dynamics"][key][-1])


def test_64_plate_owners_resume_pooling_and_zero_contributions(tmp_path):
    fragment, _ = _categorical()
    fragment.ports = {"observation": PortSpec("observation", "x", PortRole.DATA)}
    fragment = scoped(fragment, "scope/subject")
    template = Template(name="population", plates={
        "patients": PlateSpec(fragment, dims=(64,), per_copy_blobs=("scope/subject/p",)),
    })
    expanded = expand_plates(template)
    arrays = {name: np.array([0.25, 0.75], dtype=np.float32) for name in expanded.blobs}
    with jax.enable_x64(False):
        engine = Engine(compile_template(template, arrays=arrays))
        priors = {name: {"alpha": np.ones(2, dtype=np.float32)} for name in expanded.learnables}
        learning = Learnables.from_persistent_state(engine, priors, priors)
        path = tmp_path / "population.cffgz"
        save_bundle(path, template, arrays, learnables=learning)
        loaded = load_bundle(path)
        result = _run(engine, {name: 0 for name in expanded.ports})
        learning.accumulate(engine, result)
        loaded.learnables.accumulate(engine, result)
        _same_learning(loaded.learnables, learning)
        assert len(loaded.learnables.plans) == 64
        for name, values in loaded.learnables.params.items():
            np.testing.assert_array_equal(values["alpha"], [2.0, 1.0])
            assert values["alpha"][1] == loaded.learnables.initial_params[name]["alpha"][1]
        _same_arrays(loaded.learnables.write(engine, engine.init_state().params),
                     learning.write(engine, engine.init_state().params))


@pytest.mark.parametrize("corruption", ("d8", "unidentified", "incomplete", "tampered", "shape", "dtype", "improper_prior"))
def test_recurrence_bundle_restore_requires_current_complete_state(tmp_path, corruption):
    template, arrays, _, learning, _, _ = _fixture("rslds", np.float64)
    path = tmp_path / "recurrence.cffgz"
    save_bundle(path, template, arrays, learnables=learning)
    contents = _zip_contents(path)
    manifest = json.loads(contents["manifest.json"])
    record = next(r for r in manifest["learning"]["learnables"] if r["family"] == "pg_weights")
    if corruption in ("d8", "unidentified"):
        if corruption == "d8":
            record["semantic_identity"] = "cffg.recurrence.d8-singleton-product/1.0.0"
        else:
            del record["semantic_identity"]
    else:
        role = "prior" if corruption == "improper_prior" else "posterior"
        row = next(row for row in record["arrays"] if row["role"] == role and row["key"] == "precision")
        if corruption == "incomplete":
            del contents[row["name"]]
        else:
            value = np.load(io.BytesIO(contents[row["name"]]))
            if corruption == "shape":
                value = value.reshape(-1)
            elif corruption == "dtype":
                value = value.astype(np.float32)
            elif corruption == "improper_prior":
                value = -value
            else:
                value.flat[0] += 1
            buffer = io.BytesIO()
            np.save(buffer, value)
            contents[row["name"]] = buffer.getvalue()
    contents["manifest.json"] = json.dumps(manifest).encode()
    _write_zip(path, contents)
    before = learning.state_dict()
    with pytest.raises(BundleError):
        load_bundle(path)
    _same_arrays(learning.state_dict(), before)


@pytest.mark.slow
def test_fitted_rslds_exports_complete_state_and_resumes(tmp_path):
    from test_rslds_fit import _fit, _observations

    with jax.enable_x64(False):
        observations = _observations(8)
        fit = _fit(observations, outer_iterations=1, inner_iterations=200)
        fit.last_inference.require_converged()
        path = tmp_path / "fit.cffgz"
        fit.save_bundle(path)
        loaded = load_bundle(path)
        _same_learning(loaded.learnables, fit.learnables)
        _same_arrays(loaded.arrays, {**fit.initialization.arrays(), **fit.parameters})
        engine = Engine(compile_template(loaded.template, arrays=loaded.arrays))
        _same_arrays(loaded.learnables.write(engine, engine.init_state().params),
                     fit.learnables.write(fit.engine, fit.engine.init_state().params))
        data = {f"y{step}": value for step, value in enumerate(observations + np.float32(0.1))}
        later = _run(engine, data, loaded.learnables)
        fit.learnables.accumulate(fit.engine, later)
        loaded.learnables.accumulate(engine, later)
        _same_learning(loaded.learnables, fit.learnables)
        save_bundle(path, loaded.template, loaded.arrays, learnables=loaded.learnables)
        _same_learning(load_bundle(path).learnables, loaded.learnables)


def test_documented_save_load_infer_update_and_resave(tmp_path, monkeypatch):
    from pathlib import Path

    readme = Path(__file__).parents[1] / "README.md"
    section = readme.read_text().split("<!-- bundle-resume:start -->")[1].split("<!-- bundle-resume:end -->")[0]
    source = section.split("```python\n")[1].split("```", 1)[0]
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(source, str(readme), "exec"), namespace)
    loaded = load_bundle(tmp_path / "sensor.cffgz")
    np.testing.assert_array_equal(loaded.learnables.initial_params["probabilities"]["alpha"], [1, 1])
    np.testing.assert_array_equal(loaded.learnables.params["probabilities"]["alpha"], [2, 2])
    _same_learning(loaded.learnables, namespace["learning"])


@pytest.mark.parametrize("schedule", ("auto", "flood"))
def test_gaussian_resume_accepts_each_inference_execution_surface(tmp_path, schedule):
    template, arrays, _, learning, first, later = _fixture("gaussian", np.float64)
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    learning.accumulate(engine, _run(engine, first))
    path = tmp_path / "gaussian.cffgz"
    save_bundle(path, template, arrays, learnables=learning)
    loaded = load_bundle(path)
    result = _run(engine, later, loaded.learnables)
    learning.replace(engine, result)
    loaded.learnables.replace(engine, result)
    _same_learning(loaded.learnables, learning)
