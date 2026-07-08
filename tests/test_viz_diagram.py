"""Semantic diagram derivation: Template -> CFFG visual algebra (Part I §5).

Executable claims: the default q (joint node marginals + normalisation beads)
is made explicit; constraints/ports/blobs perturb it exactly as the paper's
notation prescribes; GFE composites get dashed boxes; glue ports become
rollout ellipses.
"""

import numpy as np
import pytest

from cffg.core.ir import ConstraintKind, ConstraintSpec, Template, VariableSpec
from cffg.models.rollout import build_rollout
from cffg.viz import BeadShape, NodeEnd, TerminalEnd, TerminalKind, derive


def rollout(**kw):
    K, M, U = 4, 3, 2
    template, _ = build_rollout(
        "roll",
        steps=kw.pop("steps", 2),
        past=kw.pop("past", 0),
        d=np.full(K, 1 / K),
        A=np.full((M, K), 1 / M),
        c=np.full(M, 1 / M),
        e=np.full(U, 1 / U),
        Bs=np.stack([np.eye(K)] * U),
        **kw,
    )
    return template


class TestHmmSlice:
    def test_equality_node_is_default(self, hmm_slice):
        d = derive(hmm_slice)
        eq = d.nodes["eq"]
        assert eq.default_factorisation
        assert len(eq.groups) == 1
        assert eq.groups[0].roles == ("a", "b", "c")
        assert eq.groups[0].shape is BeadShape.CIRCLE

    def test_blob_bound_node_loses_default(self, hmm_slice):
        d = derive(hmm_slice)
        trans = d.nodes["trans"]
        assert trans.pointmass_roles == ("matrix",)
        assert not trans.default_factorisation  # the f_d case, Fig 16
        assert len(trans.groups) == 1
        assert trans.groups[0].roles == ("out", "in")

    def test_data_clamped_role_leaves_q(self, hmm_slice):
        d = derive(hmm_slice)
        obs = d.nodes["obs"]
        assert set(obs.pointmass_roles) == {"out", "matrix"}  # x clamped + A blob
        assert obs.groups[0].roles == ("in",)

    def test_wire_terminals(self, hmm_slice):
        d = derive(hmm_slice)
        x = d.wires["var:x"]
        (term,) = [e for e in x.ends if isinstance(e, TerminalEnd)]
        assert term.kind is TerminalKind.DATA and term.ref == "port:sense"
        assert x.bead is None  # data constraint: pointmass, not a q bead

        for port in ("head", "tail"):
            var = hmm_slice.ports[port].variable
            (term,) = [
                e for e in d.wires[f"var:{var}"].ends if isinstance(e, TerminalEnd)
            ]
            assert term.kind is TerminalKind.ELLIPSIS  # glue: graph continues

        blob = d.wires["blob:trans.matrix"]
        (term,) = [e for e in blob.ends if isinstance(e, TerminalEnd)]
        assert term.kind is TerminalKind.POINTMASS and term.label == "B"
        assert blob.bead is None

    def test_internal_wire_has_default_bead(self, hmm_slice):
        d = derive(hmm_slice)
        z = d.wires["var:z_mid"]
        assert z.bead == ""  # empty normalisation bead
        assert {e.factor for e in z.node_ends()} == {"trans", "eq"}


class TestRollout:
    def test_p_substitution_square_and_composites(self):
        d = derive(rollout(gfe=True))
        for k in (1, 2):
            obs = d.nodes[f"obs{k}"]
            squares = [g for g in obs.groups if g.shape is BeadShape.SQUARE]
            assert [g.roles for g in squares] == [("out",)]
        assert [(c.members, c.x_variable) for c in d.composites] == [
            (("obs1", "goal1"), "x1"),
            (("obs2", "goal2"), "x2"),
        ]

    def test_no_gfe_means_no_squares_no_composites(self):
        d = derive(rollout(gfe=False))
        assert d.composites == []
        assert all(
            g.shape is BeadShape.CIRCLE for n in d.nodes.values() for g in n.groups
        )

    def test_map_policy_delta_bead_on_control_wire(self):
        d = derive(rollout(map_policy=True))
        assert d.wires["var:u1"].bead == "delta"  # Fig 29's MAP readout
        assert d.wires["var:zt1"].bead == ""

    def test_perception_slice_ports(self, viz_gallery):
        viz_gallery(rollout(steps=2, past=1), tag="past slice: act/obs clamps")
        d = derive(rollout(steps=2, past=1))
        for wire_id in ("var:u1", "var:x1"):
            (term,) = [
                e for e in d.wires[wire_id].ends if isinstance(e, TerminalEnd)
            ]
            assert term.kind is TerminalKind.DATA
        # planning slice unchanged
        assert d.composites and d.composites[0].members == ("obs2", "goal2")

    def test_horizon_glue_renders_as_ellipsis(self):
        d = derive(rollout())
        (term,) = [
            e for e in d.wires["var:zt2"].ends if isinstance(e, TerminalEnd)
        ]
        assert term.kind is TerminalKind.ELLIPSIS


class TestEdgeCases:
    def test_mean_field_splits_joint_bead(self, hmm_slice, viz_gallery):
        hmm_slice.constraints.append(
            ConstraintSpec(ConstraintKind.MEAN_FIELD, "eq")
        )
        d = derive(hmm_slice)
        assert [g.roles for g in d.nodes["eq"].groups] == [("a",), ("b",), ("c",)]
        viz_gallery(hmm_slice, level=0, tag="mean-field eq, uncompressed")

    def test_disconnected_variable_rejected(self):
        t = Template(name="bad", variables={"loose": VariableSpec("loose", 2)})
        with pytest.raises(ValueError, match="loose"):
            derive(t)
