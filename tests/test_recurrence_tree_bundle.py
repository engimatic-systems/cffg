"""Stored recurrence declarations, executable admission, and legacy learner migration."""

import json
from dataclasses import replace

import numpy as np
import pytest

from cffg.bundle import BundleError, load_bundle, save_bundle
from cffg.colony import Colony
from cffg.compiler import CompileError, compile_template
from cffg.core.recurrence_tree import RecurrenceTree
from cffg.core.serialize import document_structure_hash, structure_hash
from cffg.engine import Engine
from cffg.learn import Learnables
from cffg.models.rslds_fit import vague_rslds_fit_priors
from test_bundle import _representative_rslds, _write_zip, _zip_contents
from test_recurrence_link import _template


def _model(tree):
    template, arrays = _template(
        weights=np.arange(24, dtype=np.float32).reshape(4, 3, 2),
        offsets=np.arange(12, dtype=np.float32).reshape(4, 3),
    )
    template.factors["recurrence"] = replace(template.factors["recurrence"], recurrence_tree=tree)
    return template, arrays


def test_balanced_source_roundtrip_and_executable_admission(tmp_path, monkeypatch):
    path = tmp_path / "recurrence.cffgz"
    colony = Colony()
    tree = RecurrenceTree.decision_list(4)
    template, arrays = _model(tree)
    save_bundle(path, template, arrays)
    canonical = load_bundle(path)
    assert canonical.resumable and not canonical.source_only
    assert canonical.template.factors["recurrence"].recurrence_tree == tree
    colony.spawn(canonical.template, canonical.arrays)

    def unexpected_arena(*args, **kwargs):
        pytest.fail("a source-only model must never assemble an executable arena")

    monkeypatch.setattr("cffg.compiler.assemble_arena", unexpected_arena)
    tree = RecurrenceTree.balanced(4)
    template, arrays = _model(tree)
    save_bundle(path, template, arrays)
    loaded = load_bundle(path)
    assert loaded.source_only and not loaded.resumable and loaded.learnables is None
    assert loaded.template.factors["recurrence"].recurrence_tree == tree
    assert loaded.manifest["structure_hash"] != canonical.manifest["structure_hash"]
    assert set(loaded.arrays) == set(arrays)
    for name, value in arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], value, strict=True)
    for semiring in ("sum", "max"):
        with pytest.raises(CompileError, match="cff-lv65"):
            compile_template(loaded.template, arrays=loaded.arrays, semiring=semiring)
    with pytest.raises(CompileError, match="cff-lv65"):
        colony.spawn(loaded.template, loaded.arrays)
    assert len(colony.instances) == 1


def test_source_only_checks_parameters_before_replacing_archive(tmp_path):
    path = tmp_path / "fixed.cffgz"
    template, arrays = _model(RecurrenceTree.balanced(4))
    save_bundle(path, template, arrays)
    original = path.read_bytes()
    for name, bad_value, message in (
        ("weights", arrays["weights"].transpose(1, 0, 2), "parameter shape"),
        ("covariance", np.diag([-1., 1.]), "positive definite"),
        ("offsets", np.full((4, 3), np.nan), "finite"),
    ):
        with pytest.raises(BundleError, match=message):
            save_bundle(path, template, {**arrays, name: bad_value})
        assert path.read_bytes() == original


def test_source_only_checks_archive_members_and_learning_inventory(tmp_path):
    path = tmp_path / "fixed.cffgz"
    template, arrays = _model(RecurrenceTree.balanced(4))
    save_bundle(path, template, arrays)
    contents = _zip_contents(path)
    for name, value, message in (
        ("blobs/weights.npy", b"damaged", "sha256 mismatch"),
        ("unexpected.npy", b"extra", "unexpected ZIP members"),
    ):
        _write_zip(path, {**contents, name: value})
        with pytest.raises(BundleError, match=message):
            load_bundle(path)
    for field, value, message in (
        ("schema_version", "unrecognized", "unknown learning manifest version"),
        ("learnables", [{"name": "ghost", "arrays": []}], "cannot restore learning state"),
    ):
        manifest = json.loads(contents["manifest.json"])
        manifest["learning"][field] = value
        _write_zip(path, {**contents, "manifest.json": json.dumps(manifest)})
        with pytest.raises(BundleError, match=message):
            load_bundle(path)


def test_legacy_model_hash_checked_before_migrating_learning_state(tmp_path):
    path = tmp_path / "learned.cffgz"
    template, arrays = _representative_rslds()
    priors = vague_rslds_fit_priors(3, 2, 3).learning_priors()
    learning = Learnables(Engine(compile_template(template, arrays=arrays)), priors=priors)
    # A non-prior posterior catches an accidental reset during migration.
    learning.params["initial_regime"]["alpha"] += np.array([4., 1., 0.])
    save_bundle(path, template, arrays, learnables=learning)
    contents = _zip_contents(path)
    document = json.loads(contents["model.json"])
    expected_hash = document_structure_hash(document)
    assert document["schema_version"] == "1.2"
    document["schema_version"] = "1"
    for factor in document["factors"].values():
        factor.pop("recurrence_tree", None)
    contents["model.json"] = json.dumps(document)
    _write_zip(path, contents)
    with pytest.raises(BundleError, match="structure hash mismatch"):
        load_bundle(path)

    legacy_hash = document_structure_hash(document)
    assert expected_hash != legacy_hash
    manifest = json.loads(contents["manifest.json"])
    manifest["structure_hash"] = legacy_hash
    contents["manifest.json"] = json.dumps(manifest)
    _write_zip(path, contents)
    migrated = load_bundle(path)
    assert migrated.manifest["structure_hash"] == legacy_hash
    assert structure_hash(migrated.template) == expected_hash
    assert migrated.resumable and not migrated.source_only
    assert all(f.recurrence_tree.is_decision_list() for f in migrated.template.factors.values()
               if f.type == "recurrence_link")
    assert migrated.learnables.persistent_descriptors() == learning.persistent_descriptors()
    save_bundle(path, migrated.template, migrated.arrays, learnables=migrated.learnables)
    assert json.loads(_zip_contents(path)["model.json"])["schema_version"] == "1.2"
    assert load_bundle(path).manifest["structure_hash"] == expected_hash

    saved = _zip_contents(path)
    alternate = json.loads(saved["model.json"])
    alternate["factors"]["switch1"]["recurrence_tree"] = RecurrenceTree.balanced(3).to_dict()
    changed = json.loads(saved["manifest.json"])
    changed["structure_hash"] = document_structure_hash(alternate)
    _write_zip(path, {**saved, "manifest.json": json.dumps(changed), "model.json": json.dumps(alternate)})
    with pytest.raises(BundleError, match="cannot restore learning state"):
        load_bundle(path)

    manifest["learning"]["learnables"][0]["semantic_identity"] = "unrecognized"
    contents["manifest.json"] = json.dumps(manifest)
    _write_zip(path, contents)
    with pytest.raises(BundleError, match="identity"):
        load_bundle(path)
