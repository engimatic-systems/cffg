"""Composition and unrolling: gluing is port bonding, nothing more."""

import pytest

from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
    validate,
)
from cffg.core.compose import ComposeError, compose, merge, unroll
from cffg.core.serialize import structure_hash
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
