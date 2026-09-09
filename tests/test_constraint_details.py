"""P-substitution declarations cannot silently replace conflicting meaning."""

import json
from dataclasses import replace

import pytest
from test_bundle import _write_zip, _zip_contents
from test_mu3 import _arrays, _model

from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core import ConstraintKind, ConstraintSpec, ModelError, validate
from cffg.core.serialize import from_json, structure_hash, to_dict, to_json


def declaration(target="obs", **detail):
    return ConstraintSpec(ConstraintKind.P_SUBSTITUTION, target, tuple(detail.items()))


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_declarations_fail_before_compiler_collapse(reverse):
    template = _model()
    original = next(
        c for c in template.constraints if c.kind is ConstraintKind.P_SUBSTITUTION
    )
    conflict = replace(
        original,
        detail=tuple({**original.detail_dict(), "substituted_role": "in"}.items()),
    )
    template.constraints = [
        c for c in template.constraints if c.kind is not ConstraintKind.P_SUBSTITUTION
    ]
    pair = [original, conflict]
    template.constraints.extend(reversed(pair) if reverse else pair)
    with pytest.raises(ModelError, match="conflicting.*substituted_role"):
        compile_template(template, arrays=_arrays())


def test_raw_detail_key_conflict_is_rejected_before_serialization(hmm_slice):
    hmm_slice.constraints.append(
        ConstraintSpec(
            ConstraintKind.P_SUBSTITUTION,
            "obs",
            (("substituted_role", "in"), ("substituted_role", "out")),
        )
    )
    with pytest.raises(ModelError, match="conflicting.*substituted_role"):
        validate(hmm_slice)
    with pytest.raises(ValueError, match="conflicting.*substituted_role"):
        to_json(hmm_slice)


def test_identical_repeats_are_preserved_and_detail_order_does_not_matter(hmm_slice):
    first = declaration(substituted_role="out", extra={"b": [1, 2], "a": "ok"})
    second = declaration(extra={"a": "ok", "b": [1, 2]}, substituted_role="out")
    hmm_slice.constraints.extend((first, second))
    validate(hmm_slice)
    payload = to_json(hmm_slice)
    restored = from_json(payload)
    assert len(restored.constraints) == 2
    assert to_json(restored) == payload
    assert to_dict(hmm_slice)["constraints"][0]["detail"] == first.detail_dict()


@pytest.mark.parametrize(
    "first,second,key",
    [
        (
            {"parameter_message": "mu3_is_v1"},
            {"parameter_message": "unknown"},
            "parameter_message",
        ),
        ({"extra": {"a": [1, 2]}}, {"extra": {"a": [2, 1]}}, "extra"),
        ({}, {"extra": None}, "extra"),
        ({"extra": 1}, {"extra": 1.0}, "extra"),
        ({"extra": 0.0}, {"extra": -0.0}, "extra"),
        ({"extra": True}, {"extra": 1}, "extra"),
    ],
)
def test_complete_metadata_conflicts_are_order_independent(
    hmm_slice, first, second, key
):
    pair = [
        declaration(substituted_role="out", **first),
        declaration(substituted_role="out", **second),
    ]
    hmm_slice.constraints.extend(pair)
    with pytest.raises(ModelError) as forward:
        validate(hmm_slice)
    hmm_slice.constraints.reverse()
    with pytest.raises(ModelError) as backward:
        validate(hmm_slice)
    assert forward.value.problems == backward.value.problems
    assert len(forward.value.problems) == 1
    assert f"'{key}':" in forward.value.problems[0]
    with pytest.raises(ValueError, match="conflicting"):
        to_json(hmm_slice)


def test_identical_raw_keys_keep_existing_serialized_identity(hmm_slice):
    original = declaration(substituted_role="out", extra={"x": [1, 2]})
    hmm_slice.constraints.append(original)
    expected = to_json(hmm_slice)
    expected_hash = structure_hash(hmm_slice)
    hmm_slice.constraints[0] = replace(
        original, detail=original.detail + tuple(reversed(original.detail))
    )
    validate(hmm_slice)
    assert to_json(hmm_slice) == expected
    assert structure_hash(hmm_slice) == expected_hash


def test_single_declaration_diagnostics_and_serializer_scope_are_unchanged(hmm_slice):
    hmm_slice.constraints.extend(
        [
            declaration(target="unknown", substituted_role="out"),
            declaration(substituted_role="sideways"),
        ]
    )
    with pytest.raises(ModelError) as original:
        validate(hmm_slice)
    assert original.value.problems == [
        "p_substitution constraint: unknown factor 'unknown'",
        "p_substitution on 'obs': role 'sideways' is not one of its edges",
    ]
    assert len(from_json(to_json(hmm_slice)).constraints) == 2
    hmm_slice.constraints.append(declaration(substituted_role="out"))
    with pytest.raises(ModelError) as combined:
        validate(hmm_slice)
    assert combined.value.problems[:-1] == original.value.problems
    assert "conflicting" in combined.value.problems[-1]


def test_json_input_preserves_conflicts_until_validation(hmm_slice):
    hmm_slice.constraints.append(declaration(substituted_role="out"))
    payload = to_dict(hmm_slice)
    payload["constraints"].append(
        {
            "kind": "p_substitution",
            "target": "obs",
            "detail": {"substituted_role": "in"},
        }
    )
    restored = from_json(json.dumps(payload))
    assert len(restored.constraints) == 2
    with pytest.raises(ModelError, match="conflicting"):
        validate(restored)
    with pytest.raises(ModelError, match="conflicting"):
        compile_template(restored, arrays={})


def test_conflict_reporting_is_sorted_and_complete_for_32_factors():
    template = _model(slices=32)
    original = list(template.constraints)
    template.constraints.extend(
        replace(c, detail=tuple({**c.detail_dict(), "substituted_role": "in"}.items()))
        for c in original
        if c.kind is ConstraintKind.P_SUBSTITUTION
    )
    with pytest.raises(ModelError) as forward:
        validate(template)
    template.constraints.reverse()
    with pytest.raises(ModelError) as backward:
        validate(template)
    assert forward.value.problems == backward.value.problems
    assert len(forward.value.problems) == 32
    for target, message in zip(
        sorted(f"obs{i}" for i in range(32)), forward.value.problems
    ):
        assert f"on {target!r}: conflicting" in message


@pytest.mark.parametrize("legacy", [False, True])
def test_repeated_declarations_reconstruct_from_current_and_legacy_bundles(
    tmp_path, legacy
):
    template = _model()
    original = next(
        c for c in template.constraints if c.kind is ConstraintKind.P_SUBSTITUTION
    )
    template.constraints.append(
        replace(original, detail=tuple(reversed(original.detail)))
    )
    path = tmp_path / "repeated.cffgz"
    save_bundle(path, template, _arrays())
    if legacy:
        contents = _zip_contents(path)
        manifest = json.loads(contents["manifest.json"])
        del manifest["blob_paths"]
        contents["manifest.json"] = json.dumps(manifest).encode()
        _write_zip(path, contents)
    restored, arrays, _ = load_bundle(path)
    assert len(restored.constraints) == len(template.constraints)
    assert structure_hash(restored) == structure_hash(template)
    assert len(compile_template(restored, arrays=arrays).gfe_groups) == 1


def test_bundle_save_rejects_conflicts_before_creating_destination(tmp_path):
    template = _model()
    template.constraints.append(declaration(target="obs0", substituted_role="in"))
    path = tmp_path / "conflicting.cffgz"
    with pytest.raises(ValueError, match="conflicting"):
        save_bundle(path, template, _arrays())
    assert not path.exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_loaded_bundle_rejects_conflicts_during_structure_hash_check(tmp_path, legacy):
    path = tmp_path / "conflicting-input.cffgz"
    save_bundle(path, _model(), _arrays())
    contents = _zip_contents(path)
    model = json.loads(contents["model.json"])
    model["constraints"].append(
        {
            "kind": "p_substitution",
            "target": "obs0",
            "detail": {"substituted_role": "in"},
        }
    )
    contents["model.json"] = json.dumps(model).encode()
    if legacy:
        manifest = json.loads(contents["manifest.json"])
        del manifest["blob_paths"]
        contents["manifest.json"] = json.dumps(manifest).encode()
    _write_zip(path, contents)
    with pytest.raises(ValueError, match="conflicting.*substituted_role"):
        load_bundle(path)
