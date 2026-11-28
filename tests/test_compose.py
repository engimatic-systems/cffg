"""Composition and unrolling: gluing is port bonding, nothing more."""

from dataclasses import replace

import pytest

from cffg.core import (
    BlobRef,
    Broadcast,
    Chain,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    Expose,
    LearnableSpec,
    PlateSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
    validate,
)
from cffg.core.compose import ComposeError, compose, merge, scoped, unroll
from cffg.core.plates import expand_plates
from cffg.core.serialize import from_dict, from_json, structure_hash, to_dict, to_json
from conftest import K, make_hmm_slice


def make_state_prior(cardinality: int = K) -> Template:
    """p(z0) = Cat(z0|d): terminates a chain's head."""
    return Template(
        name="state_prior",
        variables={"z0": VariableSpec("z0", cardinality)},
        factors={
            "init": FactorSpec(
                "init", "categorical", edges={"out": "z0"}, param_blobs={"statistic": "d"}
            )
        },
        ports={"attach": PortSpec("attach", "z0", PortRole.GLUE)},
        blobs={"d": BlobRef("d", "blobs/d.safetensors")},
    )


def test_unroll_three_steps(hmm_slice):
    rolled = unroll(hmm_slice, 3, {"tail": "head"})
    # Each of the 2 bonds unifies two variables into one and consumes 2 ports.
    assert len(rolled.variables) == 3 * 5 - 2
    assert len(rolled.factors) == 3 * 3
    assert set(rolled.ports) == {"t0/head", "t2/tail", "t0/sense", "t1/sense", "t2/sense"}
    validate(rolled)


def test_unroll_is_deterministic(hmm_slice):
    assert structure_hash(unroll(hmm_slice, 3, {"tail": "head"})) == structure_hash(
        unroll(make_hmm_slice(), 3, {"tail": "head"})
    )
    assert structure_hash(unroll(hmm_slice, 2, {"tail": "head"})) != structure_hash(
        unroll(hmm_slice, 3, {"tail": "head"})
    )


def test_unroll_carries_constraints_per_copy(hmm_slice):
    hmm_slice.constraints.append(
        ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "obs", (("substituted_role", "out"),))
    )
    rolled = unroll(hmm_slice, 2, {"tail": "head"})
    targets = {c.target for c in rolled.constraints}
    assert targets == {"t0/obs", "t1/obs"}


def test_compose_prior_onto_chain(hmm_slice):
    rolled = unroll(hmm_slice, 2, {"tail": "head"})
    model = compose(
        make_state_prior(),
        rolled,
        {"attach": "t0/head"},
        name="hmm2_with_prior",
        scope_a="prior",
        scope_b="chain",
    )
    # Only the far tail and the two data ports remain open.
    assert set(model.ports) == {"chain/t2/tail".replace("t2", "t1"), "chain/t0/sense", "chain/t1/sense"}
    validate(model)


def test_bond_cardinality_mismatch_rejected(hmm_slice):
    with pytest.raises(ComposeError, match="domain"):
        compose(make_state_prior(cardinality=K + 1), hmm_slice, {"attach": "head"}, name="bad")


def test_merge_requires_scoping(hmm_slice):
    with pytest.raises(ComposeError, match="name clash"):
        merge(hmm_slice, make_hmm_slice(), name="clash")


def test_chain_ports_must_be_glue(hmm_slice):
    with pytest.raises(ComposeError, match="role glue"):
        unroll(hmm_slice, 2, {"tail": "sense"})


def test_unrolled_chain_is_a_valid_operand_again(hmm_slice):
    """Composition is closed: an unrolled chain composes with another chain."""
    a = unroll(hmm_slice, 2, {"tail": "head"})
    b = unroll(hmm_slice, 2, {"tail": "head"})
    week = compose(a, b, {"t1/tail": "t0/head"}, name="glued_chains")
    assert len(week.factors) == 12
    validate(week)


def test_fanout_of_boundary_keeps_original_port_composable():
    from cffg.core import fan_out

    tree, ports = fan_out(make_state_prior(), "z0", 16)
    validate(tree)
    assert tree.ports["attach"].role is PortRole.GLUE
    assert tuple(p for p in tree.ports if p != "attach") == ports
    combined = compose(tree, make_state_prior(), {ports[-1]: "attach"}, name="coupled")
    validate(combined)
    assert "a/attach" in combined.ports
    assert "a/" + ports[-1] not in combined.ports


def test_compact_scope_and_declared_port_composition_match_flat_wiring(hmm_slice):
    hmm_slice.learnables = {
        "shared": LearnableSpec("shared", "dirichlet", ("A",), ()),
        "private": LearnableSpec("private", "dirichlet", ("B",), ("trans",)),
    }
    hmm_slice.blobs["B"] = replace(hmm_slice.blobs["B"], path="weights/{index}.npy")
    host = make_state_prior()
    host.plates["group/cells"] = PlateSpec(
        hmm_slice, (2, 3), (Chain("tail", "head", axis=1), Broadcast("sense", "all"), Expose()), ("B",)
    )
    original = to_json(host)
    compact = compose(host, host, {"attach": "attach"}, name="pair", scope_b="b/inner")
    flat = expand_plates(host)
    reference = compose(flat, flat, {"attach": "attach"}, name="pair", scope_b="b/inner")
    assert set(compact.plates) == {"a/group/cells", "b/inner/group/cells"}
    assert to_dict(expand_plates(compact)) == to_dict(reference)
    assert "b/inner/group/cells/all" in reference.ports
    assert reference.learnables["b/inner/shared"].sources == ()
    assert reference.learnables["a/group/cells/1/2/private"].sources == ("a/group/cells/1/2/trans",)
    assert reference.blobs["a/group/cells/1/2/B"].path == "weights/1/2.npy"
    assert scoped(scoped(host, "inner"), "b") == scoped(host, "b/inner")
    assert from_json(to_json(compact)) == compact
    document = to_dict(compact)
    assert document["schema_version"] == "1.1"
    renamed = from_dict(document)
    plate = renamed.plates["a/group/cells"]
    renamed.plates["a/group/cells"] = replace(plate, shared_scope="new")
    assert structure_hash(renamed) != structure_hash(compact)
    document["schema_version"] = "1"
    with pytest.raises(ValueError, match="shared plate scope"):
        from_dict(document)
    assert to_dict(host)["schema_version"] == "1"
    assert "shared_scope" not in to_dict(host)["plates"]["group/cells"]
    with pytest.raises(ComposeError, match="expand_plates.*generated plate port"):
        compose(host, host, {"group/cells/0/0/head": "attach"}, name="bad")
    assert to_json(host) == original


@pytest.mark.parametrize("kind, name", [
    ("plates", "p"), ("blobs", "A"), ("blobs", "p/1/B"), ("ports", "p/hub"),
    ("factors", "p/_bcast/hub/eq1"), ("learnables", "shared"),
])
def test_compact_merge_rejects_owner_and_generated_collisions_atomically(hmm_slice, kind, name):
    hmm_slice.learnables["shared"] = LearnableSpec("shared", "dirichlet", ("A",))
    host = Template("grid", plates={
        "p": PlateSpec(hmm_slice, (3,), (Broadcast("sense", "hub"),), ("B",))
    })
    other = make_state_prior()
    if kind == "plates":
        other.plates[name] = host.plates[name]
    elif kind == "blobs":
        other.blobs[name] = BlobRef(name, "overlap.npy")
    elif kind == "learnables":
        other.learnables[name] = LearnableSpec(name, "dirichlet", ("d",))
    elif kind == "ports":
        other.ports[name] = replace(other.ports.pop("attach"), name=name)
    else:
        other.factors[name] = replace(other.factors.pop("init"), name=name)
    before = to_json(host), to_json(other)
    for a, b in ((host, other), (other, host)):
        with pytest.raises(ComposeError, match="clash|already exists"):
            merge(a, b, "bad")
    assert (to_json(host), to_json(other)) == before


def test_disjoint_scoped_merge_does_not_materialize_large_plates(monkeypatch, hmm_slice):
    host = Template("large", plates={"p": PlateSpec(hmm_slice, (1_000_000,))})
    monkeypatch.setattr("cffg.core.plates.expand_plates", lambda t: pytest.fail("grounded compact operand"))
    merged = merge(scoped(host, "left"), scoped(host, "right"), "pair")
    assert all(spec.dims == (1_000_000,) for spec in merged.plates.values())


def test_nested_fragment_fails_before_scoping_or_merging(hmm_slice):
    inner = Template("inner", plates={"cell": PlateSpec(hmm_slice, (1,))})
    outer = Template("outer", plates={"group": PlateSpec(inner, (2,))})
    before = to_json(outer)
    for operation in (lambda: scoped(outer, "a"), lambda: merge(outer, Template("b"), "c")):
        with pytest.raises(ValueError, match="nested plate grounding"):
            operation()
    assert to_json(outer) == before


def test_shared_owner_is_reserved_before_every_copy(hmm_slice):
    cell = make_state_prior()
    cell.blobs["p/0/B"] = BlobRef("p/0/B", "late.npy")
    for name in ("before", "zz"):
        host = Template("host", plates={
            "p": PlateSpec(hmm_slice, (2,), per_copy_blobs=("B",)),
            name: PlateSpec(cell, (1,)),
        })
        with pytest.raises(ComposeError, match="blobs name clash.*p/0/B"):
            expand_plates(scoped(host, "scene"))
