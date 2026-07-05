"""Core IR and validation: the port invariant is the load-bearing rule."""

import dataclasses

import pytest

from cffg.core import (
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    FactorType,
    ModelError,
    PortRole,
    PortSpec,
    VariableSpec,
    register_factor_type,
    validate,
)


def test_valid_slice_passes(hmm_slice):
    validate(hmm_slice)


def test_degrees(hmm_slice):
    assert hmm_slice.degree("z_mid") == 2  # trans + eq
    assert hmm_slice.degree("z_in") == 1  # dangling -> port
    assert hmm_slice.degree("x") == 1


def problems_of(template) -> str:
    with pytest.raises(ModelError) as err:
        validate(template)
    return "\n".join(err.value.problems)


def test_dangling_edge_without_port_rejected(hmm_slice):
    del hmm_slice.ports["sense"]
    assert "dangling" in problems_of(hmm_slice)


def test_port_on_fully_connected_edge_rejected(hmm_slice):
    hmm_slice.ports["bad"] = PortSpec("bad", "z_mid", PortRole.GLUE)
    assert "fully connected" in problems_of(hmm_slice)


def test_degree_three_rejected(hmm_slice):
    # Wire the observation factor straight into z_mid: degree 3, needs equality.
    hmm_slice.factors["obs"] = dataclasses.replace(
        hmm_slice.factors["obs"], edges={"out": "x", "in": "z_mid"}
    )
    text = problems_of(hmm_slice)
    assert "degree 3" in text and "equality" in text


def test_unused_variable_rejected(hmm_slice):
    hmm_slice.variables["ghost"] = VariableSpec("ghost", 4)
    assert "unused" in problems_of(hmm_slice)


def test_unknown_variable_reference_rejected(hmm_slice):
    hmm_slice.factors["trans"] = dataclasses.replace(
        hmm_slice.factors["trans"], edges={"out": "z_mid", "in": "nope"}
    )
    assert "unknown variable 'nope'" in problems_of(hmm_slice)


def test_param_role_must_have_exactly_one_source(hmm_slice):
    trans = hmm_slice.factors["trans"]
    hmm_slice.factors["trans"] = dataclasses.replace(
        trans, param_edges={"matrix": "z_out"}  # blob B is still bound too
    )
    assert "exactly one of edge/blob" in problems_of(hmm_slice)


def test_equality_requires_shared_cardinality(hmm_slice):
    hmm_slice.variables["z_obs"] = VariableSpec("z_obs", 7)
    assert "shared cardinality" in problems_of(hmm_slice)


def test_p_substitution_role_checked(hmm_slice):
    hmm_slice.constraints.append(
        ConstraintSpec(
            ConstraintKind.P_SUBSTITUTION, "obs", (("substituted_role", "sideways"),)
        )
    )
    assert "not one of its edges" in problems_of(hmm_slice)


def test_p_substitution_wellformed_accepted(hmm_slice):
    hmm_slice.constraints.append(
        ConstraintSpec(ConstraintKind.P_SUBSTITUTION, "obs", (("substituted_role", "out"),))
    )
    validate(hmm_slice)


def test_all_problems_reported_at_once(hmm_slice):
    del hmm_slice.ports["sense"]
    hmm_slice.variables["ghost"] = VariableSpec("ghost", 4)
    with pytest.raises(ModelError) as err:
        validate(hmm_slice)
    assert len(err.value.problems) == 2


def test_factor_type_registry_rejects_duplicates():
    with pytest.raises(ValueError, match="already registered"):
        register_factor_type(FactorType("categorical", edge_roles=("out",)))


def test_unknown_factor_type_rejected(hmm_slice):
    hmm_slice.factors["woo"] = FactorSpec("woo", "quantum", edges={"out": "z_out"})
    assert "unknown type 'quantum'" in problems_of(hmm_slice)
