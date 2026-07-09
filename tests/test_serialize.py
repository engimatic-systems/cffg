"""Serialization round-trips and structure-hash semantics."""

import dataclasses

from cffg.core import ConstraintKind, ConstraintSpec, validate
from cffg.core.serialize import (
    from_json,
    load,
    save,
    structure_hash,
    to_dict,
    to_json,
)

from conftest import make_hmm_slice


def test_round_trip_json(hmm_slice):
    hmm_slice.constraints.append(
        ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "obs", (("substituted_role", "out"),))
    )
    back = from_json(to_json(hmm_slice))
    assert to_dict(back) == to_dict(hmm_slice)
    validate(back)


def test_gaussian_domain_round_trips(hmm_slice):
    from cffg.core import VariableSpec

    hmm_slice.variables["g"] = VariableSpec("g", dim=2)  # unused is fine: no validate here
    d = to_dict(hmm_slice)
    assert d["variables"]["g"] == {"dim": 2}
    assert d["variables"]["z_in"] == {"cardinality": 3}  # discrete form unchanged
    back = from_json(to_json(hmm_slice))
    assert back.variables["g"].dim == 2 and back.variables["g"].family == "gaussian"
    assert structure_hash(back) == structure_hash(hmm_slice)


def test_round_trip_file(tmp_path, hmm_slice):
    p = tmp_path / "slice.json"
    save(hmm_slice, p)
    assert to_dict(load(p)) == to_dict(hmm_slice)


def test_canonical_form_is_order_insensitive(hmm_slice):
    shuffled = make_hmm_slice()
    shuffled.variables = dict(reversed(list(shuffled.variables.items())))
    shuffled.factors = dict(reversed(list(shuffled.factors.items())))
    assert to_json(shuffled) == to_json(hmm_slice)


def test_hash_stable_across_blob_relocation(hmm_slice):
    before = structure_hash(hmm_slice)
    moved = make_hmm_slice()
    moved.blobs["A"] = dataclasses.replace(moved.blobs["A"], path="elsewhere/A.safetensors")
    assert structure_hash(moved) == before


def test_hash_sensitive_to_structure(hmm_slice):
    base = structure_hash(hmm_slice)

    recarded = make_hmm_slice()
    recarded.variables["z_in"] = dataclasses.replace(recarded.variables["z_in"], cardinality=9)
    assert structure_hash(recarded) != base

    beaded = make_hmm_slice()
    beaded.constraints.append(
        ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "obs", (("substituted_role", "out"),))
    )
    assert structure_hash(beaded) != base


def test_unsupported_schema_version_rejected(hmm_slice):
    import json

    import pytest

    d = json.loads(to_json(hmm_slice))
    d["schema_version"] = "999"
    with pytest.raises(ValueError, match="schema version"):
        from_json(json.dumps(d))


def test_ef_domain_round_trips(hmm_slice):
    from cffg.core import VariableSpec

    hmm_slice.variables["lam"] = VariableSpec("lam", ef="gamma")  # unused: no validate
    d = to_dict(hmm_slice)
    assert d["variables"]["lam"] == {"ef": "gamma"}
    back = from_json(to_json(hmm_slice))
    assert back.variables["lam"].ef == "gamma" and back.variables["lam"].family == "ef"
    assert structure_hash(back) == structure_hash(hmm_slice)
    # The domain tag is structure: retagging the family must change the hash.
    hmm_slice.variables["lam"] = VariableSpec("lam", ef="beta")
    assert structure_hash(hmm_slice) != structure_hash(back)
