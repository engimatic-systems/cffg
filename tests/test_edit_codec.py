"""The GUI wire format: edit bundles round-trip through JSON, and dry-run
validation reports problems without mutating anything."""

import pytest

from cffg.core import (
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    VariableSpec,
)
from cffg.core.edits import (
    AddConstraint,
    RemoveVariable,
    AddFactor,
    AddVariable,
    DeclarePort,
    EditError,
    RebindBlobPath,
    RemoveFactor,
    ReplaceFactor,
    action_from_dict,
    apply_edits,
    bundle_from_json,
    bundle_to_json,
    check_edits,
)
from cffg.core.serialize import to_dict


def grow_bundle():
    return [
        AddVariable(VariableSpec("u", 4)),
        AddFactor(
            FactorSpec("ctrl", "categorical", edges={"out": "u"}, param_blobs={"statistic": "B"})
        ),
        DeclarePort(PortSpec("control", "u", PortRole.DATA)),
        AddConstraint(ConstraintSpec(ConstraintKind.DELTA, "u")),
        RebindBlobPath("A", "trained/A_v2.safetensors"),
    ]


def test_bundle_round_trips_through_json(hmm_slice):
    bundle = grow_bundle()
    revived = bundle_from_json(bundle_to_json(bundle))
    assert revived == bundle  # frozen dataclasses compare by value
    direct = apply_edits(hmm_slice, bundle)
    via_json = apply_edits(hmm_slice, revived)
    assert to_dict(direct) == to_dict(via_json)


def test_check_edits_reports_without_raising(hmm_slice):
    before = to_dict(hmm_slice)
    problems = check_edits(hmm_slice, [AddVariable(VariableSpec("ghost", 3))])
    assert problems and "unused" in problems[0]
    assert to_dict(hmm_slice) == before  # dry run touched nothing
    assert check_edits(hmm_slice, grow_bundle()) == []


def test_replace_factor_rewires_in_one_action(hmm_slice):
    rewired = apply_edits(
        hmm_slice,
        [
            RemoveFactor("obs"),
            RemoveFactor("eq"),
            ReplaceFactor(
                FactorSpec(
                    "trans", "matrix_cat", edges={"out": "z_out", "in": "z_in"}, param_blobs={"matrix": "B"}
                )
            ),
            # drop the now-orphaned observation branch
        ]
        + [RemoveVariable(v) for v in ("x", "z_obs", "z_mid")],
    )
    assert set(rewired.variables) == {"z_in", "z_out"}


def test_replace_missing_factor_refused(hmm_slice):
    with pytest.raises(EditError, match="use AddFactor"):
        apply_edits(
            hmm_slice,
            [ReplaceFactor(FactorSpec("nope", "categorical", edges={"out": "z_in"}))],
        )


def test_unknown_action_name_refused():
    with pytest.raises(EditError, match="unknown edit action"):
        action_from_dict({"action": "transmogrify", "name": "z"})
