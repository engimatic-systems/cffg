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


def test_validation_builds_one_incidence_index(hmm_slice, monkeypatch):
    def repeated_lookup(*_args, **_kwargs):
        raise AssertionError("validation must not rescan the full template per variable")

    monkeypatch.setattr(type(hmm_slice), "degree", repeated_lookup)
    monkeypatch.setattr(type(hmm_slice), "ports_of", repeated_lookup)
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
    assert "shared domain" in problems_of(hmm_slice)


# --- domain families (cff-4e5z) ---------------------------------------------


def test_variable_spec_requires_exactly_one_domain():
    with pytest.raises(ValueError, match="exactly one"):
        VariableSpec("both", cardinality=3, dim=2)
    with pytest.raises(ValueError, match="exactly one"):
        VariableSpec("neither")
    assert VariableSpec("d", 3).family == "discrete"
    assert VariableSpec("g", dim=2).family == "gaussian"
    assert VariableSpec("g", dim=2).domain_key == ("gaussian", 2)


def test_gaussian_factor_rejects_discrete_edge(hmm_slice):
    hmm_slice.variables["y"] = VariableSpec("y", 4)  # discrete, wrong family
    hmm_slice.factors["prior"] = FactorSpec(
        "prior", "gaussian", edges={"out": "y"}, param_blobs={"mean": "A", "cov": "A"}
    )
    hmm_slice.ports["yp"] = PortSpec("yp", "y", PortRole.GLUE)
    assert "needs gaussian edges" in problems_of(hmm_slice)


def test_equality_rejects_mixed_families(hmm_slice):
    hmm_slice.variables["z_obs"] = VariableSpec("z_obs", dim=3)
    assert "shared domain" in problems_of(hmm_slice)


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


# --- EF domains (cff-8s6a) ----------------------------------------------------


def ef_chain(name="ef_chain") -> "Template":
    """gamma prior --lam(ef:gamma)-- poisson_obs --x(ef:poisson): data port."""
    from cffg.core import BlobRef, Template

    return Template(
        name=name,
        variables={
            "lam": VariableSpec("lam", ef="gamma"),
            "x": VariableSpec("x", ef="poisson"),
        },
        factors={
            "prior": FactorSpec(
                "prior", "gamma", edges={"out": "lam"},
                param_blobs={"shape": "a0", "rate": "b0"},
            ),
            "obs": FactorSpec("obs", "poisson_obs", edges={"out": "x", "rate": "lam"}),
        },
        ports={"count": PortSpec("count", "x", PortRole.DATA)},
        blobs={
            "a0": BlobRef("a0", "blobs/a0.npy"),
            "b0": BlobRef("b0", "blobs/b0.npy"),
        },
    )


def test_ef_variable_spec_domain():
    v = VariableSpec("lam", ef="gamma")
    assert v.family == "ef" and v.domain_key == ("ef:gamma", 1)
    with pytest.raises(ValueError, match="exactly one"):
        VariableSpec("both", cardinality=3, ef="gamma")


def test_ef_chain_validates():
    validate(ef_chain())


def test_unknown_ef_family_rejected():
    t = ef_chain()
    t.variables["lam"] = VariableSpec("lam", ef="cauchy")
    text = problems_of(t)
    assert "unknown EF family 'cauchy'" in text


def test_ef_factor_rejects_wrong_edge_family():
    t = ef_chain()
    t.variables["lam"] = VariableSpec("lam", ef="beta")  # rate edge must be gamma
    assert "needs an ef='gamma' edge" in problems_of(t)


def test_ef_factor_rejects_discrete_edge():
    t = ef_chain()
    t.variables["lam"] = VariableSpec("lam", 3)
    assert "needs ef edges" in problems_of(t)


def test_ef_equality_requires_same_family():
    t = ef_chain()
    t.variables["p"] = VariableSpec("p", ef="beta")
    t.factors["eq"] = FactorSpec("eq", "equality", edges={"a": "lam", "b": "p", "c": "p"})
    assert "shared domain" in problems_of(t)
