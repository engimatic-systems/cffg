"""Compression passes (Part I p.19) as presentation directives.

Each test pins one pass to the figure that illustrates it (Figs 14-19)."""

import numpy as np
import pytest

from cffg.models.rollout import build_rollout
from cffg.viz import derive
from cffg.viz.compress import Placement, present


@pytest.fixture
def hmm(hmm_slice):
    return derive(hmm_slice)


def test_level_bounds(hmm):
    for bad in (-1, 5):
        with pytest.raises(ValueError, match="level"):
            present(hmm, bad)


def test_pass1_empty_wire_beads_vanish(hmm):
    # Fig 15: the empty normalisation bead on the internal z_mid edge goes.
    assert present(hmm, 0).wires["var:z_mid"].bead
    for level in range(1, 5):
        assert not present(hmm, level).wires["var:z_mid"].bead


def test_pass1_keeps_symbol_beads():
    K, U, M = 4, 2, 3
    template, _ = build_rollout(
        "roll", steps=1,
        d=np.full(K, 1 / K), A=np.full((M, K), 1 / M), c=np.full(M, 1 / M),
        e=np.full(U, 1 / U), Bs=np.stack([np.eye(K)] * U), map_policy=True,
    )
    p = present(derive(template), 4)
    assert p.wires["var:u1"].bead and p.wires["var:u1"].symbol == "delta"


def test_pass2_default_nodes_lose_internal_beads(hmm):
    # Fig 16: eq has default factorisation, its joint bead goes; trans has a
    # pointmass parameter edge (the f_d case) so its bead stays.
    assert present(hmm, 1).nodes["eq"].groups
    for level in (2, 3, 4):
        p = present(hmm, level)
        assert p.nodes["eq"].groups == ()
        assert p.nodes["trans"].groups


def test_pass3_default_nodes_lose_internal_edges(hmm):
    # Fig 17: eq's extensions go; trans keeps its internal edges until pass 4.
    assert present(hmm, 2).nodes["eq"].internal_edges
    p3 = present(hmm, 3)
    assert not p3.nodes["eq"].internal_edges
    assert p3.nodes["trans"].internal_edges


def test_pass4_pushes_beads_to_border(hmm):
    # Figs 18-19: surviving beads straddle the border, boxes go compact.
    p3, p4 = present(hmm, 3), present(hmm, 4)
    assert p3.nodes["trans"].placement is Placement.INTERNAL and not p3.compact
    assert p4.nodes["trans"].placement is Placement.BORDER and p4.compact
    assert all(not d.internal_edges for d in p4.nodes.values())


def test_level0_is_uncompressed(hmm):
    p = present(hmm, 0)
    assert all(d.groups or n == "eq" for n, d in p.nodes.items())
    assert p.nodes["eq"].groups  # even default nodes show their joint bead
    assert p.wires["var:z_mid"].bead
    assert not p.wires["blob:trans.matrix"].bead  # pointmass: never a bead
