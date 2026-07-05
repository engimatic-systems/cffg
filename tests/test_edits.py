"""Edit bundles are all-or-nothing; failures leave the original untouched."""

import pytest

from cffg.core import (
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    ModelError,
    PortRole,
    PortSpec,
    VariableSpec,
)
from cffg.core.edits import (
    AddConstraint,
    AddFactor,
    AddVariable,
    DeclarePort,
    EditError,
    RebindBlobPath,
    RemoveFactor,
    RemovePort,
    RemoveVariable,
    apply_edits,
)
from cffg.core.serialize import structure_hash, to_dict


def snapshot(t):
    return to_dict(t)


def test_valid_bundle_grows_the_chain(hmm_slice):
    """Extend the slice with a control prior on a new selector edge — the
    kind of coherent multi-part change bundles exist for."""
    grown = apply_edits(
        hmm_slice,
        [
            AddVariable(VariableSpec("u", 4)),
            AddFactor(
                FactorSpec("ctrl", "categorical", edges={"out": "u"}, param_blobs={"statistic": "B"})
            ),
            DeclarePort(PortSpec("control", "u", PortRole.DATA)),
        ],
    )
    assert "ctrl" in grown.factors and "control" in grown.ports
    assert "ctrl" not in hmm_slice.factors  # original untouched


def test_half_valid_bundle_rolls_back(hmm_slice):
    before = snapshot(hmm_slice)
    with pytest.raises(ModelError):
        apply_edits(
            hmm_slice,
            [
                AddVariable(VariableSpec("u", 4)),  # fine alone...
                # ...but no factor/port ever attaches u -> unused variable
            ],
        )
    assert snapshot(hmm_slice) == before


def test_edit_error_rolls_back(hmm_slice):
    before = snapshot(hmm_slice)
    with pytest.raises(EditError, match="does not exist"):
        apply_edits(hmm_slice, [RemoveFactor("obs"), RemoveFactor("obs")])
    assert snapshot(hmm_slice) == before


def test_remove_factor_alone_violates_port_invariant(hmm_slice):
    with pytest.raises(ModelError):
        apply_edits(hmm_slice, [RemoveFactor("obs")])
    # The coherent bundle: the whole observation branch goes, the equality node
    # becomes pointless, and the transition rewires straight to the tail edge.
    pruned = apply_edits(
        hmm_slice,
        [
            RemoveFactor("obs"),
            RemoveFactor("eq"),
            RemoveVariable("x"),  # also drops port "sense"
            RemoveVariable("z_obs"),
            RemoveVariable("z_mid"),
            RemoveFactor("trans"),
            AddFactor(
                FactorSpec(
                    "trans", "matrix_cat", edges={"out": "z_out", "in": "z_in"}, param_blobs={"matrix": "B"}
                )
            ),
        ],
    )
    assert set(pruned.variables) == {"z_in", "z_out"}
    assert set(pruned.ports) == {"head", "tail"}


def test_remove_variable_cascades_ports(hmm_slice):
    with pytest.raises(ModelError):
        # dropping x removes port sense too; obs factor now references a ghost
        apply_edits(hmm_slice, [RemoveVariable("x")])


def test_constraint_bundle_toggles_bfe_ablation(hmm_slice):
    """The one-bead GFE<->BFE flip from Part II §7.1, as a two-action bundle."""
    bead = ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "obs", (("substituted_role", "out"),))
    gfe = apply_edits(hmm_slice, [AddConstraint(bead)])
    assert structure_hash(gfe) != structure_hash(hmm_slice)

    from cffg.core.edits import RemoveConstraint

    bfe = apply_edits(gfe, [RemoveConstraint(bead)])
    assert structure_hash(bfe) == structure_hash(hmm_slice)


def test_rebind_blob_path_preserves_structure_hash(hmm_slice):
    moved = apply_edits(hmm_slice, [RebindBlobPath("A", "trained/A_v2.safetensors")])
    assert structure_hash(moved) == structure_hash(hmm_slice)
    assert moved.blobs["A"].path == "trained/A_v2.safetensors"


def test_remove_port_alone_dangles(hmm_slice):
    with pytest.raises(ModelError):
        apply_edits(hmm_slice, [RemovePort("sense")])
