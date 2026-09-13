"""Versioned learning meaning, complete array ownership, and safe migration."""

import copy
from dataclasses import replace
import json
import zipfile

import numpy as np
import pytest

from cffg import conj
from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core.compose import merge, scoped
from cffg.core.edits import RemoveFactor, apply_edits
from cffg.core.ir import BlobRef, FactorSpec, LearnableSpec, PlateSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.learning_identity import LEARNING_IDENTITIES
from cffg.core.plates import expand_plates
from cffg.core.serialize import (
    from_dict, from_json, legacy_learning_structure_hash, structure_hash, to_dict, to_json,
)
from cffg.engine import Engine
from cffg.learn import FAMILIES, Learnables
from cffg.learning_schema import (
    PERSISTENT_MANIFEST_KEY, PersistentParameter, encode_persistent_manifest,
    persistent_schema, resolve_persistent_state, validate_persistent_manifest,
)

from test_agent import fresh
from test_gaussian_learnables import _learning_template
from test_hybrid_learnables import _mixture_template, _recurrence_template


def _categorical():
    template = Template(
        name="counts",
        variables={"x": VariableSpec("x", cardinality=2)},
        ports={"output": PortSpec("output", "x", PortRole.GLUE)},
        blobs={"p": BlobRef("p", "p.npy")},
        factors={"f": FactorSpec("f", "categorical", edges={"out": "x"}, param_blobs={"statistic": "p"})},
        learnables={"q": LearnableSpec("q", "dirichlet", ("p",), ("f",))},
    )
    return template, {"p": np.asarray([0.25, 0.75])}


@pytest.fixture(scope="module")
def families():
    fixtures = []
    t, arrays = _categorical()
    fixtures.append((t, arrays, {"q": np.ones(2)}))
    t, arrays = _learning_template(2)
    fixtures.append((t, arrays, {"a_state": conj.vague_niw(2), "z_dynamics": conj.vague_mniw(2, 2)}))
    t, arrays, _ = _mixture_template()
    fixtures.append((t, arrays, {"neutral_dynamics": tuple(conj.vague_mniw(2, 2) for _ in range(3))}))
    t, arrays, _ = _recurrence_template()
    fixtures.append((t, arrays, {"neutral_recurrence": conj.vague_pg_weights(3, 2)}))
    result = {}
    for template, arrays, priors in fixtures:
        engine = Engine(compile_template(template, arrays=arrays))
        learning = Learnables(engine, priors=priors)
        for plan in learning.plans.values():
            result[plan.family] = (template, arrays, engine, plan, learning.initial_params[plan.name])
    return result


@pytest.mark.parametrize("family_name", tuple(LEARNING_IDENTITIES))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_every_family_describes_and_restores_exact_arrays(families, family_name, dtype):
    template, arrays, engine, plan, prior = families[family_name]
    state = {key: np.asarray(value, dtype=dtype) for key, value in prior.items()}
    descriptor = resolve_persistent_state(plan, FAMILIES[family_name], state, state)
    assert descriptor.semantic_identity == LEARNING_IDENTITIES[family_name]
    assert {a.role for a in descriptor.arrays} == {"prior", "posterior"}
    assert len(descriptor.arrays) == 2 * len(prior)
    assert {a.dtype for a in descriptor.arrays} == {np.dtype(dtype).str}
    assert len({a.name for a in descriptor.arrays}) == len(descriptor.arrays)
    assert descriptor.sources == plan.sources
    assert {t.blob for t in descriptor.targets} == set(plan.shapes)
    # Use a different destination prior dtype to exercise persistence without casting.
    destination = Learnables(engine, priors={plan.name: dict(prior)}, only=(plan.name,))
    checkpoint = {f"v2:{plan.name}:{key}": value for key, value in state.items()}
    checkpoint[PERSISTENT_MANIFEST_KEY] = encode_persistent_manifest((descriptor,))
    destination.load_state_dict(checkpoint, exact=True)
    for key, value in state.items():
        assert destination.params[plan.name][key].dtype == value.dtype
        np.testing.assert_array_equal(destination.params[plan.name][key], value)
    for key, value in prior.items():
        np.testing.assert_array_equal(destination.initial_params[plan.name][key], value)
    assert to_dict(from_json(to_json(template))) == to_dict(template)


@pytest.mark.parametrize("family_name", tuple(LEARNING_IDENTITIES))
def test_family_schema_rejects_incomplete_shapes_roles_and_dtypes(families, family_name):
    _t, _a, _e, plan, prior = families[family_name]
    family = FAMILIES[family_name]
    key = next(iter(prior))
    for invalid in (
        {k: v for k, v in prior.items() if k != key},
        {**prior, key: np.zeros((99,))},
        {**prior, key: np.asarray(prior[key], dtype=int)},
        {**prior, "extra": np.zeros(1)},
    ):
        with pytest.raises(ValueError, match=plan.name):
            resolve_persistent_state(plan, family, prior, invalid)
    with pytest.raises(ValueError, match=plan.name):
        persistent_schema(replace(plan, targets=(replace(plan.targets[0], role="wrong"),)), family)
    with pytest.raises(ValueError, match=plan.name):
        persistent_schema(replace(plan, targets=(replace(plan.targets[0], form="wrong"),)), family)


def test_duplicate_family_keys_and_missing_identity_are_rejected(families):
    _t, _a, _e, plan, _p = families["dirichlet"]

    class DuplicateKeys:
        semantic_identity = plan.semantic_identity
        parameter_forms = ("expected",)
        check = FAMILIES["dirichlet"].check

        def persistent_parameters(self, role_shapes):
            return (PersistentParameter("alpha", (2,)),) * 2

    with pytest.raises(ValueError, match="q.*duplicate persistent keys"):
        persistent_schema(plan, DuplicateKeys())
    with pytest.raises(ValueError, match="q.*missing.*identity"):
        persistent_schema(replace(plan, semantic_identity=""), FAMILIES["dirichlet"])


def test_invalid_public_descriptors_cannot_be_encoded(families):
    _t, _a, _e, plan, prior = families["dirichlet"]
    descriptor = resolve_persistent_state(plan, FAMILIES["dirichlet"], prior, prior)
    for invalid in (
        replace(descriptor, semantic_identity=""),
        replace(descriptor, arrays=descriptor.arrays[:-1]),
        replace(descriptor, arrays=descriptor.arrays + descriptor.arrays[:1]),
        replace(descriptor, arrays=(replace(descriptor.arrays[0], dtype="float64"), descriptor.arrays[1])),
    ):
        with pytest.raises(ValueError, match="q"):
            encode_persistent_manifest((invalid,))


def test_identity_changes_hash_but_numeric_values_do_not():
    template, arrays = _categorical()
    original = structure_hash(template)
    template.blobs["p"] = BlobRef("p", "another.npy", (("kind", "uniform"),))
    arrays["p"] = np.ones(2)
    assert structure_hash(template) == original
    template.learnables["q"] = replace(template.learnables["q"], semantic_identity="cffg.learning.dirichlet/2.0.0")
    assert structure_hash(template) != original
    with pytest.raises(ValueError, match="q.*incompatible semantic identity"):
        compile_template(template, arrays=arrays)


@pytest.mark.parametrize("family_name", tuple(LEARNING_IDENTITIES))
def test_pre_ticket_declarations_migrate_only_known_unchanged_semantics(families, family_name):
    template, _arrays, _engine, plan, _prior = families[family_name]
    document = to_dict(template)
    document.pop("learning_schema_version")
    for spec in document["learnables"].values():
        spec.pop("semantic_identity")
    if family_name == "pg_weights":
        with pytest.raises(ValueError, match=f"{plan.name}.*D8.*D9"):
            from_dict(document)
        assert legacy_learning_structure_hash(template) is None
    else:
        migrated = from_dict(document)
        assert to_dict(migrated) == to_dict(template)
        assert structure_hash(migrated) == structure_hash(template)


@pytest.mark.parametrize("identity", (None, "", "cffg.learning.dirichlet/2.0.0", "cffg.recurrence.d8-singleton-product/1.0.0"))
def test_current_json_never_silently_fills_missing_or_unknown_identity(identity):
    template, _ = _categorical()
    document = to_dict(template)
    if identity is None:
        document["learnables"]["q"].pop("semantic_identity")
    else:
        document["learnables"]["q"]["semantic_identity"] = identity
    with pytest.raises(ValueError, match="q.*identity"):
        from_dict(document)


def test_unknown_declaration_schema_version():
    t, _ = _categorical()
    with pytest.raises(ValueError, match="unknown learning schema version"):
        from_dict({**to_dict(t), "learning_schema_version": "2"})


@pytest.mark.parametrize("mutation", (
    lambda d: d.update(semantic_identity="cffg.learning.dirichlet/99.0.0"),
    lambda d: d.update(sources=[]),
    lambda d: d["targets"][0].update(role="matrix"),
    lambda d: d["targets"][0].update(form="counts"),
    lambda d: d["arrays"].append(copy.deepcopy(d["arrays"][0])),
    lambda d: d["arrays"].pop(),
    lambda d: d["arrays"][0].update(dtype="float64"),
    lambda d: d["arrays"][0].update(shape=[99]),
    lambda d: d["arrays"][0].update(name="../q.npy"),
))
def test_tampered_descriptor_is_rejected_before_posterior_or_points_change(families, mutation):
    _t, _a, engine, plan, prior = families["dirichlet"]
    learning = Learnables(engine, priors={plan.name: dict(prior)})
    checkpoint = learning.state_dict()
    document = json.loads(checkpoint[PERSISTENT_MANIFEST_KEY].item())
    mutation(document["learnables"][0])
    checkpoint[PERSISTENT_MANIFEST_KEY] = np.asarray(json.dumps(document))
    before = learning.hyperparameter_snapshot()
    point = {k: np.asarray(v).copy() for k, v in engine.arena.log_params.items()}
    with pytest.raises(ValueError, match="q"):
        learning.load_state_dict(checkpoint)
    for key, value in before.items():
        np.testing.assert_array_equal(learning.hyperparameter_snapshot()[key], value)
    for key, value in point.items():
        np.testing.assert_array_equal(engine.arena.log_params[key], value)


def test_manifest_version_payload_and_completeness_fail(families):
    _t, _a, engine, plan, prior = families["niw"]
    learning = Learnables(engine, priors={plan.name: dict(prior)}, only=(plan.name,))
    checkpoint = learning.state_dict()
    document = json.loads(checkpoint[PERSISTENT_MANIFEST_KEY].item())
    document["schema_version"] = "2.0.0"
    with pytest.raises(ValueError, match="unknown learning manifest version"):
        validate_persistent_manifest(np.asarray(json.dumps(document)), learning.persistent_descriptors())
    missing = dict(checkpoint)
    missing.pop(f"v2:{plan.name}:nu")
    with pytest.raises(KeyError, match="missing learnable"):
        learning.load_state_dict(missing)
    changed = dict(checkpoint)
    changed[f"v2:{plan.name}:nu"] = np.asarray(99.0)
    with pytest.raises(ValueError, match="digest mismatch"):
        learning.load_state_dict(changed)


def test_nested_scopes_plates_copy_edits_and_64_learnables():
    template, _ = _categorical()
    fragment = scoped(scoped(template, "outer"), "inner")
    parent = Template(name="population", plates={
        "agents": PlateSpec(fragment, dims=(64,), per_copy_blobs=("inner/outer/p",)),
    })
    for operation in (
        lambda: scoped(parent, "scope"),
        lambda: merge(parent, Template(name="empty"), "population"),
    ):
        with pytest.raises(ValueError, match="learnables.*inner/outer/q"):
            operation()
    parent = apply_edits(copy.deepcopy(from_json(to_json(parent))), [])
    expanded = merge(expand_plates(parent), Template(name="empty"), "population")
    arrays = {name: np.asarray([0.4, 0.6]) for name in expanded.blobs}
    engine = Engine(compile_template(parent, arrays=arrays))
    learning = Learnables(engine, priors={name: np.ones(2) for name in expanded.learnables})
    descriptors = learning.persistent_descriptors()
    assert len(descriptors) == 64
    assert len({array.name for d in descriptors for array in d.arrays}) == 128
    assert all(d.semantic_identity == LEARNING_IDENTITIES["dirichlet"] for d in descriptors)
    learning.load_state_dict(learning.state_dict(), exact=True)
    with pytest.raises(ValueError, match="learnable 'q'"):
        apply_edits(template, [RemoveFactor("f")])
    assert "f" in template.factors
    assert from_json(to_json(parent)) == parent


@pytest.mark.parametrize("name", ("../q", "/q", "q//a", "q/.", "q\\a", "q\x00"))
def test_unsafe_scoped_names_fail_before_archive_writing(tmp_path, name):
    template, arrays = _categorical()
    template.learnables = {name: replace(template.learnables["q"], name=name)}
    path = tmp_path / "model.cffgz"
    path.write_bytes(b"keep destination")
    with pytest.raises(ValueError, match="learnable"):
        save_bundle(path, template, arrays)
    assert path.read_bytes() == b"keep destination"


def test_bundle_migrates_old_model_hash_and_rejects_invalid_new_declarations(tmp_path):
    template, arrays = _categorical()
    path = tmp_path / "model.cffgz"
    save_bundle(path, template, arrays)
    with zipfile.ZipFile(path) as archive:
        records = {name: archive.read(name) for name in archive.namelist()}
    document = json.loads(records["model.json"])
    old_hash = legacy_learning_structure_hash(from_dict(document))
    document.pop("learning_schema_version")
    document["learnables"]["q"].pop("semantic_identity")
    records["model.json"] = json.dumps(document).encode()
    manifest = json.loads(records["manifest.json"])
    manifest["structure_hash"] = old_hash
    records["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in records.items():
            archive.writestr(name, value)
    restored, _arrays, _manifest = load_bundle(path)
    assert restored.learnables["q"].semantic_identity == LEARNING_IDENTITIES["dirichlet"]
    template.learnables["q"] = replace(template.learnables["q"], semantic_identity="")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="q.*identity"):
        save_bundle(path, template, arrays)
    assert path.read_bytes() == before


def test_pre_ticket_agent_envelope_restores_without_new_metadata(tmp_path):
    _env, source = fresh(learn=("B",), priors={"B": np.ones((3, 6, 6))})
    path = tmp_path / "agent.npz"
    source.checkpoint(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    data.pop(f"learn:{PERSISTENT_MANIFEST_KEY}")
    data["plan_structure_hash"] = np.asarray(source._legacy_plan_structure_hash)
    data["filter_structure_hash"] = np.asarray(source._legacy_filter_structure_hash)
    data["checkpoint_fields"] = np.asarray(json.dumps(sorted(data)))
    np.savez(path, **data)
    _env, restored = fresh(learn=("B",), priors={"B": np.ones((3, 6, 6))})
    restored.restore(path)
    for name, state in source.learnables.params.items():
        np.testing.assert_equal(restored.learnables.params[name], state)
