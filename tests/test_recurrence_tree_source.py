"""Portable recurrence trees, branch orientation, and the legacy source migration."""

from copy import deepcopy
from dataclasses import replace

import pytest

from cffg.core import (
    BlobRef,
    FactorSpec,
    ModelError,
    PlateSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
    expand_plates,
    validate,
)
from cffg.core.compose import scoped
from cffg.core.edits import ReplaceFactor, apply_edits, bundle_from_json, bundle_to_json
from cffg.core.recurrence_tree import Gate, RecurrenceTree
from cffg.core.serialize import (
    document_structure_hash,
    from_dict,
    from_json,
    structure_hash,
    to_dict,
    to_json,
)


def graph(k=4, tree=None):
    variables = {name: VariableSpec(name, cardinality=k) for name in ("source", "dest")}
    variables["drive"] = VariableSpec("drive", dim=2)
    roles = {"in": "source", "out": "dest", "drive": "drive"}
    params = {"weights": "W", "offsets": "b"}
    return Template(
        name="tree",
        variables=variables,
        ports={name: PortSpec(name, name, PortRole.GLUE) for name in variables},
        factors={
            "rec": FactorSpec(
                "rec",
                "recurrence_link",
                roles,
                param_blobs=params,
                recurrence_tree=tree,
            )
        },
        blobs={name: BlobRef(name, f"{name}.npy") for name in params.values()},
    )


@pytest.mark.parametrize("k", [2, 3, 4, 13, 1300])
def test_constructors_cover_every_destination_and_canonicalize_order(k):
    for tree in (RecurrenceTree.decision_list(k), RecurrenceTree.balanced(k)):
        assert tree.cardinality == k
        restored = RecurrenceTree(tuple(reversed(tree.gates)))
        assert tree.gates == restored.gates
        document = to_json(graph(k, tree))
        assert from_json(document).factors["rec"].recurrence_tree == tree
        assert to_json(from_json(document)) == document
    assert RecurrenceTree.decision_list(k).is_decision_list()
    assert RecurrenceTree.balanced(k).is_decision_list() == (k == 2)


@pytest.mark.parametrize(
    "gates",
    [
        (),
        (Gate("g", 0, "g"),),
        (Gate("r", 0, 1), Gate("s", "s", 2)),
        (Gate("r", 0, 1), Gate("s", 2, 3)),
        (Gate("r", 0, "bad"),),
        (Gate("r", 0, "s"), Gate("s", 0, 2)),
        (Gate("r", 0, "s"), Gate("s", 2, 3)),
        (Gate("r", "s", "s"), Gate("s", 0, 1)),
        (Gate("r", 0, 1), Gate("r", 1, 2)),
        (Gate("r", True, 0),),
    ],
)
def test_invalid_graphs_rejected(gates):
    with pytest.raises(ValueError):
        RecurrenceTree(gates)


def test_source_hash_and_branch_orientation():
    base = graph(3, RecurrenceTree((Gate("root", 0, "child"), Gate("child", 1, 2))))
    document = to_dict(base)
    for index, change in [
        (0, {"accept": "child", "reject": 0}),
        (1, {"accept": 2, "reject": 1}),
    ]:
        changed = deepcopy(document)
        changed["factors"]["rec"]["recurrence_tree"]["gates"][index].update(change)
        assert structure_hash(from_dict(changed)) != structure_hash(base)
    renamed = graph(3, RecurrenceTree.decision_list(3))
    assert structure_hash(renamed) != structure_hash(base)
    assert base.factors["rec"].recurrence_tree.is_decision_list()
    bad = deepcopy(document)
    bad["factors"]["rec"]["recurrence_tree"]["gates"][0]["left"] = 0
    with pytest.raises(ValueError, match="accept"):
        from_dict(bad)


def test_legacy_source_migrates_only_from_a_topology_free_document():
    implicit = graph(4)
    document = to_dict(implicit)
    assert document["schema_version"] == "1.2"
    legacy = deepcopy(document)
    legacy["schema_version"] = "1"
    with pytest.raises(ValueError, match="1.2"):
        from_dict(legacy)
    del legacy["factors"]["rec"]["recurrence_tree"]
    restored = from_dict(legacy)
    assert restored.factors["rec"].recurrence_tree == RecurrenceTree.decision_list(4)
    assert to_dict(restored) == document
    assert document_structure_hash(legacy) != structure_hash(restored)
    legacy["schema_version"] = "1.2"
    with pytest.raises(ValueError, match="1.2"):
        from_dict(legacy)


def test_wrong_size_or_factor_rejected_and_edit_is_atomic():
    source = graph()
    factor = source.factors["rec"]
    bad = replace(factor, recurrence_tree=RecurrenceTree.balanced(5))
    with pytest.raises(ModelError, match="cardinality"):
        apply_edits(source, [ReplaceFactor(bad)])
    assert source.factors["rec"] == factor
    wrong = graph(tree=RecurrenceTree.balanced(4))
    wrong.factors["rec"] = replace(wrong.factors["rec"], type="equality")
    with pytest.raises(ModelError, match="recurrence tree"):
        validate(wrong)
    with pytest.raises(ValueError, match="recurrence tree"):
        to_dict(wrong)


def test_edit_scope_and_compact_plate_preserve_gate_identity():
    source = graph()
    tree = RecurrenceTree.balanced(4)
    edit = ReplaceFactor(replace(source.factors["rec"], recurrence_tree=tree))
    payload = bundle_to_json([edit])
    assert "replace_recurrence_factor" in payload
    with pytest.raises(ValueError, match="topology-aware"):
        bundle_from_json(payload.replace("replace_recurrence_factor", "replace_factor"))
    result = apply_edits(source, bundle_from_json(payload))
    assert scoped(result, "p").factors["p/rec"].recurrence_tree == tree
    host = Template(name="host", plates={"grid": PlateSpec(result, (3,))})
    wire = to_dict(host)
    assert wire["schema_version"] == wire["plates"]["grid"]["fragment"]["schema_version"] == "1.2"
    with pytest.raises(ValueError, match="plate"):
        from_dict({**wire, "schema_version": "1"})
    grounded = expand_plates(from_dict(wire))
    assert [f.recurrence_tree for f in grounded.factors.values()] == [tree] * 3
