"""`.cffgz` bundles: full-restore round-trips and loud integrity failures."""

import io
import json
import zipfile

import numpy as np
import pytest

from cffg.bundle import BundleError, load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core import (
    BlobRef,
    Expose,
    FactorSpec,
    ModelError,
    PlateSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.serialize import SCHEMA_VERSION, structure_hash, to_dict, to_json
from cffg.engine import Engine
from cffg.models.gridworld import build_gridworld
from cffg.models.tmaze import build_tmaze


def _plate_model() -> tuple[Template, dict[str, np.ndarray]]:
    """Three emissions with tied A and a distinct latent prior per copy."""
    fragment = Template(
        name="leaf",
        variables={"z": VariableSpec("z", 3), "x": VariableSpec("x", 2)},
        factors={
            "prior": FactorSpec(
                "prior", "categorical", edges={"out": "z"},
                param_blobs={"statistic": "d"},
            ),
            "emit": FactorSpec(
                "emit", "matrix_cat", edges={"out": "x", "in": "z"},
                param_blobs={"matrix": "A"},
            ),
        },
        ports={"x": PortSpec("x", "x", PortRole.DATA)},
        blobs={"d": BlobRef("d", "source/d.npy"), "A": BlobRef("A", "source/A.npy")},
    )
    template = Template(
        name="plate-host",
        plates={
            "leaf": PlateSpec(
                fragment, (3,), (Expose(),), per_copy_blobs=("d",)
            )
        },
    )
    arrays = {
        "A": np.array([[0.85, 0.25, 0.4], [0.15, 0.75, 0.6]]),
        "leaf/0/d": np.array([0.7, 0.2, 0.1]),
        "leaf/1/d": np.array([0.1, 0.7, 0.2]),
        "leaf/2/d": np.array([0.2, 0.2, 0.6]),
    }
    return template, arrays


def _nested_plate_model() -> Template:
    middle, _ = _plate_model()
    return Template(
        name="nested-host",
        plates={"outer": PlateSpec(middle, (2,), (Expose(),))},
    )


def _zip_contents(path):
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _write_zip(path, contents):
    with zipfile.ZipFile(path, "w") as zf:
        for name, raw in contents.items():
            zf.writestr(name, raw)


def test_round_trip_reproduces_inference(tmp_path):
    template, arrays = build_tmaze()
    path = tmp_path / "tmaze.cffgz"
    save_bundle(path, template, arrays)

    loaded_t, loaded_arrays, manifest = load_bundle(path)
    assert structure_hash(loaded_t) == structure_hash(template)
    assert manifest["structure_hash"] == structure_hash(template)

    a = Engine(compile_template(template, arrays=arrays))
    b = Engine(compile_template(loaded_t, arrays=loaded_arrays))
    sa = a.run(a.init_state(), 400, tol=1e-13).require_converged()
    sb = b.run(b.init_state(), 400, tol=1e-13).require_converged()
    np.testing.assert_allclose(a.marginals(sa)["u1"], b.marginals(sb)["u1"], atol=1e-12)


def test_bundle_is_self_contained(tmp_path):
    template, arrays = build_gridworld(k=5)
    path = tmp_path / "gw.cffgz"
    save_bundle(path, template, arrays)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    assert "model.json" in names and "manifest.json" in names
    assert {f"blobs/{b}.npy" for b in template.blobs} <= names


def test_plate_round_trip_preserves_spec_and_compiled_inference(tmp_path):
    template, arrays = _plate_model()
    path = tmp_path / "plate.cffgz"
    save_bundle(path, template, arrays)

    loaded, loaded_arrays, manifest = load_bundle(path)
    assert structure_hash(loaded) == structure_hash(template) == manifest["structure_hash"]
    assert loaded.plates["leaf"].dims == (3,)
    assert loaded.plates["leaf"].bonds == (Expose(),)
    assert loaded.plates["leaf"].per_copy_blobs == ("d",)
    # Only deployment paths change; the complete nested structure survives.
    original = to_dict(template)
    restored = to_dict(loaded)
    original["plates"]["leaf"]["fragment"]["blobs"] = restored["plates"]["leaf"]["fragment"]["blobs"]
    assert restored == original
    assert loaded.plates["leaf"].fragment.blobs["d"].path == "blobs/leaf/{index}/d.npy"
    assert set(loaded_arrays) == set(arrays)
    for name in arrays:
        np.testing.assert_array_equal(loaded_arrays[name], arrays[name])

    before = compile_template(template, arrays=arrays)
    after = compile_template(loaded, arrays=loaded_arrays)
    assert before.edges == after.edges
    assert before.data_slots == after.data_slots
    assert before.member_of == after.member_of
    assert [(g.name, g.members) for g in before.groups] == [
        (g.name, g.members) for g in after.groups
    ]
    for key in before.log_params:
        np.testing.assert_array_equal(before.log_params[key], after.log_params[key])

    observations = {"leaf/0/x": 0, "leaf/1/x": 1, "leaf/2/x": 0}
    engine_a, engine_b = Engine(before), Engine(after)
    state_a = engine_a.bind_data(engine_a.init_state(), observations)
    state_b = engine_b.bind_data(engine_b.init_state(), observations)
    np.testing.assert_array_equal(state_a.messages, state_b.messages)
    state_a = engine_a.run(state_a, 20, tol=1e-12).require_converged()
    state_b = engine_b.run(state_b, 20, tol=1e-12).require_converged()
    for name, marginal in engine_a.marginals(state_a).items():
        np.testing.assert_allclose(marginal, engine_b.marginals(state_b)[name], atol=1e-12)
    assert engine_a.bfe(state_a) == pytest.approx(engine_b.bfe(state_b), abs=1e-12)


def test_extracted_plate_bundle_compiles_from_rewritten_paths(tmp_path):
    template, arrays = _plate_model()
    path = tmp_path / "plate.cffgz"
    save_bundle(path, template, arrays)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(path) as zf:
        zf.extractall(extracted)

    loaded, _, _ = load_bundle(path)
    from_files = compile_template(loaded, root=extracted)
    explicit = compile_template(template, arrays=arrays)
    for key in explicit.log_params:
        np.testing.assert_array_equal(from_files.log_params[key], explicit.log_params[key])


def test_tampered_blob_rejected(tmp_path):
    template, arrays = build_gridworld(k=5)
    path = tmp_path / "gw.cffgz"
    save_bundle(path, template, arrays)

    # rewrite one blob with different bytes, keep everything else
    with zipfile.ZipFile(path) as zf:
        contents = {n: zf.read(n) for n in zf.namelist()}
    buf = io.BytesIO()
    np.save(buf, np.zeros((5,)))
    contents["blobs/d.npy"] = buf.getvalue()
    with zipfile.ZipFile(path, "w") as zf:
        for n, raw in contents.items():
            zf.writestr(n, raw)

    with pytest.raises(BundleError, match="sha256 mismatch"):
        load_bundle(path)


def test_manifest_structure_hash_checked(tmp_path):
    template, arrays = build_gridworld(k=5)
    path = tmp_path / "gw.cffgz"
    save_bundle(path, template, arrays)
    with zipfile.ZipFile(path) as zf:
        contents = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(contents["manifest.json"])
    manifest["structure_hash"] = "0" * 64
    contents["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as zf:
        for n, raw in contents.items():
            zf.writestr(n, raw)
    with pytest.raises(BundleError, match="structure hash mismatch"):
        load_bundle(path)


def test_missing_arrays_refused(tmp_path):
    template, arrays = build_gridworld(k=5)
    del arrays["Bs"]
    with pytest.raises(BundleError, match="missing for blobs"):
        save_bundle(tmp_path / "x.cffgz", template, arrays)


@pytest.mark.parametrize("missing", ["A", "leaf/1/d"])
def test_missing_nested_or_per_copy_array_refused_before_write(tmp_path, missing):
    template, arrays = _plate_model()
    del arrays[missing]
    path = tmp_path / "missing.cffgz"
    with pytest.raises(BundleError, match="arrays missing for blobs"):
        save_bundle(path, template, arrays)
    assert not path.exists()


@pytest.mark.parametrize("logical_name", ["A", "leaf/1/d"])
def test_tampered_nested_or_per_copy_blob_rejected(tmp_path, logical_name):
    template, arrays = _plate_model()
    path = tmp_path / "plate.cffgz"
    save_bundle(path, template, arrays)
    contents = _zip_contents(path)
    manifest = json.loads(contents["manifest.json"])
    member = manifest["blob_paths"][logical_name]
    contents[member] = b"not a numpy array"
    _write_zip(path, contents)
    with pytest.raises(BundleError, match=rf"blob {logical_name!r} sha256 mismatch"):
        load_bundle(path)


@pytest.mark.parametrize("logical_name", ["A", "leaf/1/d"])
def test_missing_nested_or_per_copy_blob_rejected(tmp_path, logical_name):
    template, arrays = _plate_model()
    path = tmp_path / "plate.cffgz"
    save_bundle(path, template, arrays)
    contents = _zip_contents(path)
    manifest = json.loads(contents["manifest.json"])
    del contents[manifest["blob_paths"][logical_name]]
    _write_zip(path, contents)
    with pytest.raises(BundleError, match="missing blob file"):
        load_bundle(path)


def test_original_flat_schema_one_bundle_remains_loadable(tmp_path):
    template, arrays = build_gridworld(k=5)
    path = tmp_path / "legacy-flat.cffgz"
    save_bundle(path, template, arrays)
    contents = _zip_contents(path)
    manifest = json.loads(contents["manifest.json"])
    del manifest["blob_paths"]  # exact inventory shape emitted before plate support
    contents["manifest.json"] = json.dumps(manifest).encode()
    _write_zip(path, contents)

    loaded, loaded_arrays, _ = load_bundle(path)
    assert structure_hash(loaded) == structure_hash(template)
    assert set(loaded_arrays) == set(arrays)


def test_legacy_flat_manifest_rejects_plate_instead_of_truncating(tmp_path):
    template, arrays = _plate_model()
    path = tmp_path / "legacy-plate.cffgz"
    save_bundle(path, template, arrays)
    contents = _zip_contents(path)
    manifest = json.loads(contents["manifest.json"])
    del manifest["blob_paths"]
    contents["manifest.json"] = json.dumps(manifest).encode()
    _write_zip(path, contents)

    with pytest.raises(BundleError, match="legacy flat blob manifest"):
        load_bundle(path)


def test_save_rejects_nested_plate_before_creating_destination(tmp_path):
    path = tmp_path / "nested.cffgz"
    with pytest.raises(
        ModelError,
        match=r"plate 'outer': fragment declares nested plates \['leaf'\]",
    ):
        save_bundle(path, _nested_plate_model(), {})
    assert not path.exists()


def test_load_rejects_hand_built_nested_plate_bundle(tmp_path):
    template = _nested_plate_model()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "structure_hash": structure_hash(template),
        "blob_sha256": {},
        "blob_paths": {},
    }
    path = tmp_path / "hand-built-nested.cffgz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("model.json", to_json(template))
        zf.writestr("manifest.json", json.dumps(manifest))

    with pytest.raises(
        BundleError,
        match=(
            r"invalid bundled plate structure:[\s\S]*plate 'outer': fragment "
            r"declares nested plates \['leaf'\]"
        ),
    ):
        load_bundle(path)
