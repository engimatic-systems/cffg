"""Complete bundle inventories, atomic publication, and hostile archive inputs."""

import copy
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
import zipfile

import numpy as np
import pytest

from cffg import LoadedBundle, conj
from cffg.bundle import BUNDLE_SCHEMA_VERSION, BundleError, load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core import PlateSpec, Template
from cffg.core.compose import scoped
from cffg.core.plates import expand_plates
from cffg.engine import Engine
from cffg.learn import FAMILIES, Learnables

from test_bundle import _write_zip, _zip_contents
from test_gaussian_learnables import _learning_template
from test_learning_schema import _categorical


def _learner(dtype=np.float64):
    template, arrays = _learning_template(2)
    arrays = {key: value.astype(dtype) for key, value in arrays.items()}
    engine = Engine(compile_template(template, arrays=arrays))
    original = Learnables(engine, priors={
        "a_state": conj.vague_niw(2), "z_dynamics": conj.vague_mniw(2, 2),
    })
    priors = {name: {key: value.astype(dtype) for key, value in values.items()}
              for name, values in original.initial_params.items()}
    posteriors = copy.deepcopy(priors)
    for values in posteriors.values():
        values["nu"] += 2
    return template, arrays, Learnables.from_persistent_state(engine, priors, posteriors)


def _assert_state(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        assert set(actual[name]) == set(expected[name])
        for key, value in actual[name].items():
            other = expected[name][key]
            assert value.dtype == other.dtype
            assert value.shape == other.shape
            assert value.tobytes() == other.tobytes()


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_complete_state_restores_without_external_priors(tmp_path, dtype):
    template, arrays, learning = _learner(dtype)
    path = tmp_path / "model.cffgz"
    save_bundle(path, template, arrays, learnables=learning)
    loaded = load_bundle(path)
    assert isinstance(loaded, LoadedBundle)
    assert loaded.resumable
    assert loaded.manifest["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert tuple(loaded) == (loaded.template, loaded.arrays, loaded.manifest)
    _assert_state(loaded.learnables.params, learning.params)
    _assert_state(loaded.learnables.initial_params, learning.initial_params)
    for values in loaded.learnables.initial_params.values():
        assert all(not value.flags.writeable for value in values.values())
    for name, value in arrays.items():
        assert loaded.arrays[name].dtype == value.dtype
        assert loaded.arrays[name].tobytes() == value.tobytes()
    assert any(np.any(value == 0) for values in learning.params.values() for value in values.values())


@pytest.mark.parametrize("corruption", ("absent", "missing_prior", "missing_posterior", "extra_key", "extra_owner", "identity", "shape", "dtype", "nan", "improper", "point_missing", "point_extra", "point_shape", "point_dtype", "point_nan", "model"))
def test_invalid_save_preserves_destination(tmp_path, corruption):
    template, arrays, learning = _learner()
    if corruption == "absent":
        learning = None
    elif corruption == "missing_prior":
        learning.initial_params = {}
    elif corruption == "missing_posterior":
        del learning.params["a_state"]["mu"]
    elif corruption == "extra_key":
        learning.params["a_state"]["extra"] = np.ones(2)
    elif corruption == "extra_owner":
        learning.params["other"] = {"alpha": np.ones(2)}
    elif corruption == "identity":
        plan = learning.plans["a_state"]
        learning.plans["a_state"] = replace(plan, semantic_identity="unknown/1")
    elif corruption in ("shape", "dtype", "nan", "improper"):
        values = learning.params["a_state"]
        key, value = {
            "shape": ("mu", np.zeros(3)), "dtype": ("mu", np.zeros(2, dtype=int)),
            "nan": ("mu", np.full(2, np.nan)), "improper": ("psi", -np.eye(2)),
        }[corruption]
        values[key] = value
    elif corruption == "model":
        spec = template.learnables["a_state"]
        template.learnables["a_state"] = replace(spec, sources=())
    else:
        name = next(iter(arrays))
        if corruption == "point_missing":
            del arrays[name]
        elif corruption == "point_extra":
            arrays["extra"] = np.zeros(1)
        elif corruption == "point_shape":
            arrays[name] = np.zeros((99,))
        elif corruption == "point_dtype":
            arrays[name] = arrays[name].astype(complex)
        else:
            arrays[name] = np.full_like(arrays[name], np.nan)
    path = tmp_path / "model.cffgz"
    for existing in (False, True):
        if existing:
            path.write_bytes(b"keep existing destination")
        with pytest.raises((ValueError, KeyError)):
            save_bundle(path, template, arrays, learnables=learning)
        if existing:
            assert path.read_bytes() == b"keep existing destination"
        else:
            assert not path.exists()
        assert set(tmp_path.iterdir()) == ({path} if existing else set())


@pytest.mark.parametrize("corruption", ("version", "learning_version", "missing_inventory", "empty_inventory", "identity", "missing_array", "extra_array", "duplicate", "unsafe", "symlink", "digest", "prior_nan", "prior_improper", "posterior_nan", "shape", "dtype", "point_dtype", "ownership", "member_name"))
def test_tampered_archive_is_refused(tmp_path, corruption):
    template, arrays, learning = _learner()
    path = tmp_path / "model.cffgz"
    save_bundle(path, template, arrays, learnables=learning)
    contents = _zip_contents(path)
    manifest = json.loads(contents["manifest.json"])
    record = manifest["learning"]["learnables"][0]
    row = next(row for row in record["arrays"] if row["role"] == "prior" and row["key"] == "psi")
    if corruption == "version":
        manifest["schema_version"] = "999"
    elif corruption == "learning_version":
        manifest["learning"]["schema_version"] = "999"
    elif corruption == "missing_inventory":
        del manifest["learning"]
    elif corruption == "empty_inventory":
        manifest["learning"]["learnables"] = []
        contents = {name: value for name, value in contents.items() if not name.startswith("learning/")}
    elif corruption == "identity":
        record["semantic_identity"] = "unknown/1"
    elif corruption == "missing_array":
        del contents[row["name"]]
    elif corruption == "extra_array":
        contents["learning/extra.npy"] = contents[row["name"]]
    elif corruption == "unsafe":
        contents["../outside.npy"] = b"unsafe"
    elif corruption == "digest":
        row["sha256"] = "0" * 64
    elif corruption in ("prior_nan", "prior_improper", "posterior_nan", "shape", "dtype"):
        if corruption == "posterior_nan":
            row = next(row for row in record["arrays"] if row["role"] == "posterior" and row["key"] == "psi")
        value = np.load(io.BytesIO(contents[row["name"]]))
        if corruption.endswith("nan"):
            value.flat[0] = np.nan
        elif corruption == "prior_improper":
            value = -value
        elif corruption == "shape":
            value = value.reshape(-1)
        else:
            value = value.astype(np.float32)
        buffer = io.BytesIO()
        np.save(buffer, value)
        contents[row["name"]] = buffer.getvalue()
        # Invalid priors still fail with an honest digest; hashes alone are insufficient.
        if corruption.endswith("nan") or corruption == "prior_improper":
            import hashlib
            row["sha256"] = hashlib.sha256(value.tobytes()).hexdigest()
    elif corruption == "point_dtype":
        name = next(iter(arrays))
        manifest["blob_arrays"][name]["dtype"] = "<f4"
    elif corruption == "ownership":
        record["sources"] = []
    elif corruption == "member_name":
        member = row["name"]
        row["name"] = "learning/renamed.npy"
        contents[row["name"]] = contents.pop(member)
    contents["manifest.json"] = json.dumps(manifest).encode()
    _write_zip(path, contents)
    if corruption == "duplicate":
        with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning):
            archive.writestr("model.json", contents["model.json"])
    if corruption == "symlink":
        with zipfile.ZipFile(path, "a") as archive:
            info = zipfile.ZipInfo("link.npy")
            info.external_attr = 0o120777 << 16
            archive.writestr(info, "outside")
    before = learning.state_dict()
    with pytest.raises(BundleError):
        load_bundle(path)
    for key, value in before.items():
        np.testing.assert_array_equal(learning.state_dict()[key], value)


def test_64_scoped_plate_owners_and_literal_sparse_points(tmp_path):
    template, _ = _categorical()
    fragment = scoped(scoped(template, "outer%scope"), "inner:scope")
    parent = Template(name="population", plates={
        "agents": PlateSpec(fragment, dims=(64,), per_copy_blobs=("inner:scope/outer%scope/p",)),
    })
    expanded = expand_plates(parent)
    arrays = {name: np.array([0.0, 1.0], dtype=np.float32) for name in expanded.blobs}
    engine = Engine(compile_template(parent, arrays=arrays))
    priors = {name: {"alpha": np.ones(2, dtype=np.float32)} for name in expanded.learnables}
    learning = Learnables.from_persistent_state(engine, priors, priors)
    path = tmp_path / "population.cffgz"
    save_bundle(path, parent, arrays, learnables=learning)
    loaded = load_bundle(path)
    assert len(loaded.learnables.plans) == 64
    assert len(_zip_contents(path)) == 2 + 64 + 128
    _assert_state(loaded.learnables.params, learning.params)
    for name, value in arrays.items():
        assert loaded.arrays[name].dtype == np.float32
        assert loaded.arrays[name].tobytes() == value.tobytes()


def test_public_update_and_save_observe_one_committed_boundary(tmp_path, monkeypatch):
    template, arrays = _categorical()
    engine = Engine(compile_template(template, arrays=arrays))
    learning = Learnables(engine, priors={"q": np.ones(2)})
    result = engine.run(engine.init_state(), 10)
    result.require_converged()
    entered, release, saving, saved = Event(), Event(), Event(), Event()
    family = FAMILIES["dirichlet"]
    fold = family.fold

    def paused_fold(self, *args):
        entered.set()
        assert release.wait(5)
        return fold(*args)

    def save():
        saving.set()
        save_bundle(tmp_path / "state.cffgz", template, arrays, learnables=learning)
        saved.set()

    monkeypatch.setattr(type(family), "fold", paused_fold)
    with ThreadPoolExecutor(max_workers=2) as pool:
        updating = pool.submit(learning.accumulate, engine, result)
        assert entered.wait(5)
        writing = pool.submit(save)
        try:
            assert saving.wait(5)
            assert not saved.wait(0.05)
        finally:
            release.set()
        updating.result()
        writing.result()
    _assert_state(load_bundle(tmp_path / "state.cffgz").learnables.params, learning.params)


def test_replace_failure_preserves_destination_and_removes_sibling(tmp_path, monkeypatch):
    import cffg.bundle as bundle

    template, arrays, learning = _learner()
    path = tmp_path / "state.cffgz"
    path.write_bytes(b"existing")

    def fail(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(bundle.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        save_bundle(path, template, arrays, learnables=learning)
    assert path.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [path]


def test_fixed_model_has_explicit_empty_learning_inventory(tmp_path):
    template, arrays = _categorical()
    template.learnables.clear()
    path = tmp_path / "fixed.cffgz"
    save_bundle(path, template, arrays)
    loaded = load_bundle(path)
    assert loaded.resumable
    assert loaded.manifest["learning"]["learnables"] == []
    assert loaded.learnables.params == {}
