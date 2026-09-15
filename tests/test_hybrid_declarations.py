"""Serialized D12 ownership is explicit and survives every supported copy path."""

import copy
import io
import json
import zipfile
from dataclasses import FrozenInstanceError, replace

import pytest

from cffg.core import (
    HYBRID_APPROXIMATION,
    BlobRef,
    Broadcast,
    Chain,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    HybridAdapterSpec,
    HybridComponentSpec,
    HybridRegionSpec,
    HybridSeparatorSpec,
    ModelError,
    PlateSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
    expand_plates,
    validate,
)
from cffg.core.compose import ComposeError, bond, compose, merge, scoped, unroll
from cffg.core.edits import AddConstraint, AddVariable, RebindBlobPath, RemoveFactor, RemoveVariable, ReplaceFactor, apply_edits, check_edits
from cffg.core.serialize import (
    document_structure_hash, from_dict, from_json, legacy_learning_structure_hash, load, save, structure_hash, to_dict, to_json,
)


def declared():
    """One alternating R/H/D region with a live ordinary Gaussian neighbor."""
    t = Template("declared")
    t.variables = {
        **{n: VariableSpec(n, cardinality=3) for n in ("z0", "z1")},
        **{n: VariableSpec(n, dim=2) for n in ("xr", "xd", "xp", "y", "observation")},
    }
    t.factors = {
        "r": FactorSpec("r", "recurrence_link", {"in": "z0", "out": "z1", "drive": "xr"}, param_blobs={"weights": "w", "offsets": "c"}),
        "d": FactorSpec("d", "gaussian_mixture_linear", {"in": "xd", "out": "y", "selector": "z1"}, param_blobs={"matrix": "A", "offset": "b", "cov": "Q"}),
        "eq": FactorSpec("eq", "equality", {"a": "xr", "b": "xd", "c": "xp"}),
        "pz": FactorSpec("pz", "categorical", {"out": "z0"}, param_blobs={"statistic": "pi"}),
        "px": FactorSpec("px", "gaussian", {"out": "xp"}, param_blobs={"mean": "m", "cov": "V"}),
        "observe": FactorSpec("observe", "gaussian_linear", {"in": "y", "out": "observation"}, param_blobs={"matrix": "C", "offset": "e", "cov": "W"}),
    }
    t.blobs = {name: BlobRef(name, f"data/{name}.npy") for f in t.factors.values() for name in f.param_blobs.values()}
    t.ports = {"sensor": PortSpec("sensor", "observation", PortRole.DATA)}
    t.hybrid_components = {
        "association": HybridComponentSpec(
            "association",
            separators=(
                HybridSeparatorSpec("left", "z0", "xr", (0, 2)),
                HybridSeparatorSpec("middle", "z1", "xd", (0, 2)),
                HybridSeparatorSpec("right", "z1", "y", (0, 2)),
            ),
            regions=(
                HybridRegionSpec("switch", "recurrence", "r", ("left", "middle")),
                HybridRegionSpec("channel", "dynamics", "d", ("middle", "right")),
                HybridRegionSpec("label_prior", "unary", "pz", ("left",)),
                HybridRegionSpec("state_prior", "unary", "px", ("left",)),
            ),
            adapters=(HybridAdapterSpec("readout", "right", "y"),),
            equalities=("eq",),
        )
    }
    validate(t)
    return t


def update(t, **changes):
    t.hybrid_components["association"] = replace(t.hybrid_components["association"], **changes)
    return t


def exposed():
    t = declared()
    spec = t.hybrid_components["association"]
    for name in ("pz", "px"):
        del t.factors[name]
    t.ports.update({"label": PortSpec("label", "z0", PortRole.GLUE), "state": PortSpec("state", "xp", PortRole.GLUE)})
    update(t, regions=spec.regions[:2], adapters=(*spec.adapters, HybridAdapterSpec("label_input", "left", "z0"), HybridAdapterSpec("state_input", "left", "xp")))
    validate(t)
    return t


def test_json_file_hash_and_legacy_identity(tmp_path):
    t = declared()
    document = to_dict(t)
    assert document["hybrid_components"]["association"]["approximation"] == HYBRID_APPROXIMATION
    assert to_json(from_json(to_json(t))) == to_json(t)
    save(t, tmp_path / "model.json")
    assert to_json(load(tmp_path / "model.json")) == to_json(t)
    spec = t.hybrid_components["association"]
    reordered = copy.deepcopy(t)
    reordered.factors = dict(reversed(list(t.factors.items())))
    update(reordered, separators=tuple(replace(s, active_support=tuple(reversed(s.active_support))) for s in reversed(spec.separators)), regions=tuple(reversed(spec.regions)), adapters=tuple(reversed(spec.adapters)))
    assert structure_hash(t) == structure_hash(reordered)
    raw = copy.deepcopy(document)
    for s in raw["hybrid_components"]["association"]["separators"].values():
        s["active_support"].reverse()
    assert document_structure_hash(raw) == structure_hash(t)
    assert structure_hash(t) != structure_hash(scoped(t, "copy"))
    relocated = apply_edits(t, [RebindBlobPath("A", "new/A.npy")])
    assert structure_hash(t) == structure_hash(relocated)
    assert legacy_learning_structure_hash(t) is None
    t.hybrid_components.clear()
    assert "hybrid_components" not in to_dict(t)
    assert to_dict(from_dict(to_dict(t))) == to_dict(t)
    assert legacy_learning_structure_hash(t) is not None


@pytest.mark.parametrize("change", [
    {"approximation": "hybrid-regime-ec-v2"},
    {"support_semantics": "adaptive"},
    {"separators": ()},
    {"regions": ()},
])
def test_named_invalid_component(change):
    t = update(declared(), **change)
    with pytest.raises(ModelError, match="association"):
        validate(t)
    with pytest.raises(ValueError, match="association"):
        to_json(t)


@pytest.mark.parametrize("support", [(), (0, 0), (-1,), (3,), (True,), (0.5,), ((0, 2),)])
def test_invalid_or_empty_support_is_named(support):
    t = declared()
    spec = t.hybrid_components["association"]
    update(t, separators=(replace(spec.separators[0], active_support=support), *spec.separators[1:]))
    with pytest.raises(ModelError, match="association.*left"):
        validate(t)


@pytest.mark.parametrize("field,value", [("approximation", "unknown"), ("approximation", None), ("support_semantics", "adaptive"), ("separators", {}), ("equalities", None)])
def test_decode_rejects_unknown_or_malformed_identity(field, value):
    doc = to_dict(declared())
    doc["hybrid_components"]["association"][field] = value
    with pytest.raises(ValueError, match="association"):
        from_dict(doc)


def test_decode_does_not_default_missing_or_drop_extra_semantics():
    for edit in (lambda s: s.pop("approximation"), lambda s: s.update(extra="meaning")):
        doc = to_dict(declared())
        edit(doc["hybrid_components"]["association"])
        with pytest.raises(ValueError, match="association"):
            from_dict(doc)


def test_support_consistency_domains_and_duplicate_ownership():
    t = declared()
    spec = t.hybrid_components["association"]
    update(t, separators=(*spec.separators[:2], replace(spec.separators[2], active_support=(0,))))
    with pytest.raises(ModelError, match="inconsistent fixed active support"):
        validate(t)
    t = declared()
    t.variables["y"] = VariableSpec("y", dim=3)
    with pytest.raises(ModelError, match="association.*dimensions"):
        validate(t)
    t = declared()
    update(t, regions=(*spec.regions, replace(spec.regions[0], name="duplicate")))
    with pytest.raises(ModelError, match="association.*overlaps objective owner"):
        validate(t)
    t = declared()
    t.hybrid_components["second"] = replace(spec, name="second")
    with pytest.raises(ModelError, match="second.*overlaps"):
        validate(t)


def test_source_selected_channel_nonlinear_scope_and_constraints_reject_by_name():
    t = declared()
    spec = t.hybrid_components["association"]
    changed = replace(t.factors["d"], edges={**t.factors["d"].edges, "selector": "z0"})
    with pytest.raises(ModelError, match="association.*channel.*source/pair-selected"):
        apply_edits(t, [ReplaceFactor(changed)])
    update(t, regions=(spec.regions[0], replace(spec.regions[1], kind="nonlinear"), *spec.regions[2:]))
    with pytest.raises(ModelError, match="association.*channel.*unsupported"):
        validate(t)
    for kind in (ConstraintKind.P_SUBSTITUTION, ConstraintKind.MEAN_FIELD):
        t = declared()
        assert any("association" in problem for problem in check_edits(t, [AddConstraint(ConstraintSpec(kind, "r"))]))


def test_transactional_edits_preserve_or_reject_references():
    t = declared()
    original = to_json(t)
    changed = apply_edits(t, [RebindBlobPath("w", "changed.npy")])
    assert changed.hybrid_components == t.hybrid_components
    assert any("association" in problem for problem in check_edits(t, [RemoveFactor("r")]))
    with pytest.raises(ModelError, match="association"):
        apply_edits(t, [RemoveFactor("r")])
    assert to_json(t) == original
    with pytest.raises(ModelError, match="association.*active support"):
        apply_edits(t, [RemoveVariable("z0"), AddVariable(VariableSpec("z0", cardinality=2))])
    assert to_json(t) == original


@pytest.mark.parametrize("per_copy", [False, True])
def test_compact_expansion_and_blob_rewriting_preserve_references(per_copy):
    from cffg.bundle import _rewrite_blob_paths

    frag = declared()
    private = ("A",) if per_copy else ()
    frag.blobs["A"] = BlobRef("A", "matrix/{index}.npy")
    compact = Template("copies", plates={"batch": PlateSpec(frag, (2,), per_copy_blobs=private)})
    assert to_json(from_json(to_json(compact))) == to_json(compact)
    rewritten = _rewrite_blob_paths(compact)
    assert rewritten.plates["batch"].fragment.hybrid_components == frag.hybrid_components
    assert structure_hash(rewritten) == structure_hash(compact)
    expanded = expand_plates(rewritten)
    assert set(expanded.hybrid_components) == {"batch/0/association", "batch/1/association"}
    for i in range(2):
        c = expanded.hybrid_components[f"batch/{i}/association"]
        assert all(r.factor.startswith(f"batch/{i}/") for r in c.regions)
        assert all(s.regime in expanded.variables and s.state in expanded.variables for s in c.separators)
        assert all(a.variable in expanded.variables for a in c.adapters)
        assert all(e in expanded.factors for e in c.equalities)
        assert expanded.factors[f"batch/{i}/d"].param_blobs["matrix"] == (f"batch/{i}/A" if per_copy else "A")
    # Recursive blob transport works even though nested plate grounding is deferred.
    nested = Template("nested", plates={"outer": PlateSpec(compact, (2,))})
    assert structure_hash(_rewrite_blob_paths(nested)) == structure_hash(nested)
    with pytest.raises(ComposeError, match="compact plates"):
        scoped(compact, "scope")


def test_single_copy_broadcast_renames_separator_and_adapter_references():
    frag = exposed()
    compact = Template("single", plates={"p": PlateSpec(frag, (1,), bonds=(Broadcast("label", "hub"),))})
    expanded = expand_plates(compact)
    c = expanded.hybrid_components["p/0/association"]
    assert next(s for s in c.separators if s.name == "p/0/left").regime == "p/hub"
    assert next(a for a in c.adapters if a.name == "p/0/label_input").variable == "p/hub"
    validate(expanded)


def test_flat_composition_and_direct_bond_preserve_all_fields():
    t = exposed()
    prior = Template("prior", variables={"label": VariableSpec("label", cardinality=3)}, factors={"prior": FactorSpec("prior", "categorical", {"out": "label"}, param_blobs={"statistic": "prob"})}, blobs={"prob": BlobRef("prob", "p.npy")}, ports={"attach": PortSpec("attach", "label", PortRole.GLUE)})
    result = compose(prior, t, {"attach": "label"}, name="composed")
    c = result.hybrid_components["b/association"]
    assert next(s for s in c.separators if s.name == "b/left").regime == "a/label"
    assert c.approximation == HYBRID_APPROXIMATION
    joined = merge(scoped(prior, "a"), scoped(t, "b"), "direct")
    assert bond(joined, "a/attach", "b/label") is joined
    assert joined.hybrid_components == result.hybrid_components
    # Empty chain wiring is a supported disjoint expansion of declarations.
    assert len(unroll(t, 2, {}).hybrid_components) == 2


def test_chain_wiring_and_post_broadcast_validation():
    t = exposed()
    t.ports["sensor"] = replace(t.ports["sensor"], role=PortRole.GLUE)
    compact = Template("chain", plates={"p": PlateSpec(t, (2,), bonds=(Chain("sensor", "state"),))})
    expanded = expand_plates(compact)
    validate(expanded)
    second = expanded.hybrid_components["p/1/association"]
    assert next(a for a in second.adapters if a.name == "p/1/state_input").variable == "p/0/observation"
    validate(unroll(t, 2, {"sensor": "state"}))
    # Broadcasting raw scalar regime copies into another hybrid recurrence
    # cannot bypass the named ordinary-boundary admission check.
    bad = Template("broadcast", plates={"p": PlateSpec(t, (2,), bonds=(Broadcast("label", "hub"),))})
    with pytest.raises(ModelError, match="association.*association cut"):
        expand_plates(bad)


def test_categorical_matrix_attachment_uses_an_ordinary_adapter():
    t = exposed()
    t.variables["prior_z"] = VariableSpec("prior_z", cardinality=3)
    t.factors["ordinary"] = FactorSpec("ordinary", "matrix_cat", {"in": "prior_z", "out": "z0"}, param_blobs={"matrix": "probabilities"})
    t.blobs["probabilities"] = BlobRef("probabilities", "matrix.npy")
    t.ports["label"] = replace(t.ports["label"], variable="prior_z")
    validate(t)
    t.constraints.append(ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "ordinary", (("substituted_role", "out"),)))
    with pytest.raises(ModelError, match="association.*ordinary.*unsupported"):
        validate(t)


def test_association_cut_and_failed_direct_bond_are_named_and_atomic():
    t = exposed()
    outside = Template("outside", variables={"z": VariableSpec("z", cardinality=3), "x": VariableSpec("x", dim=2)}, factors={"mix": FactorSpec("mix", "gaussian_mixture", {"selector": "z", "out": "x"}, param_blobs={"mean": "mean", "cov": "cov"})}, blobs={name: BlobRef(name, name) for name in ("mean", "cov")}, ports={"z": PortSpec("z", "z", PortRole.GLUE), "x": PortSpec("x", "x", PortRole.GLUE)})
    joined = merge(scoped(t, "h"), scoped(outside, "o"), "joined")
    original = to_json(joined)
    with pytest.raises(ComposeError, match="h/association.*association cut"):
        bond(joined, "h/label", "o/z")
    assert to_json(joined) == original


def test_direct_scoping_and_merge_do_not_bypass_validation():
    invalid = update(declared(), approximation="unknown")
    with pytest.raises(ComposeError, match="association"):
        scoped(invalid, "scope")
    with pytest.raises(ComposeError, match="association"):
        merge(invalid, Template("empty"), "merged")


def test_topology_snapshot_owns_nested_declarations():
    from cffg.topology import build_topology

    t = declared()
    original = t.hybrid_components["association"]
    layout = build_topology(t)
    t.hybrid_components.clear()
    assert layout.hybrid_components["association"] == original
    assert layout.hybrid_components["association"] is not original
    with pytest.raises(TypeError):
        layout.hybrid_components["other"] = original
    with pytest.raises(FrozenInstanceError):
        layout.hybrid_components["association"].approximation = "changed"
    with pytest.raises(TypeError):
        layout.hybrid_components["association"].separators[0].active_support[0] = 1
    support = [0, 2]
    separator = HybridSeparatorSpec("copy", "z", "x", support)
    support.clear()
    assert separator.active_support == (0, 2)


def test_compile_and_bundle_guards_precede_lowering_and_destination_changes(tmp_path, monkeypatch):
    from cffg import compiler
    from cffg.bundle import BundleError, save_bundle

    t = declared()
    monkeypatch.setattr(compiler, "build_topology", lambda _: pytest.fail("premature lowering"))
    with pytest.raises(compiler.CompileError, match="association.*production compilation"):
        compiler.compile_template(t)
    path = tmp_path / "keep.cffgz"
    path.write_bytes(b"existing destination")
    with pytest.raises(BundleError, match="association.*publication"):
        save_bundle(path, t, {})
    assert path.read_bytes() == b"existing destination"
    assert list(tmp_path.iterdir()) == [path]


def test_bundle_load_rejects_a_declared_plan_before_array_resolution():
    from cffg.bundle import BUNDLE_SCHEMA_VERSION, BundleError, load_bundle

    document = to_dict(declared())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("model.json", json.dumps(document))
        archive.writestr("manifest.json", json.dumps({"schema_version": BUNDLE_SCHEMA_VERSION, "structure_hash": document_structure_hash(document)}))
    buffer.seek(0)
    with pytest.raises(BundleError, match="association.*production compilation"):
        load_bundle(buffer)
