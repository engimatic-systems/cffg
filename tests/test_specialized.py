"""Specialized ownership and structural drafts precede numeric resolution."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from test_compiler_contracts import assert_same, model
from test_ef_engine import gamma_poisson
from test_gauss_vmp import gamma_gauss_prec
from test_shared_goals import mixed_goals

from cffg import compiler, kernels
from cffg.core import ConstraintKind, ConstraintSpec
from cffg.core.plates import expand_plates
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.core.validate import validate
from cffg.models.rollout import build_rollout
from cffg.specialized import (
    OrdinaryMembership,
    complete_specialized,
    discover_specialized,
)
from cffg.specialized.constraints import resolve_constraint
from cffg.specialized.contracts import ArrayRecipe
from cffg.topology import build_topology


def stages(template):
    template = expand_plates(template)
    validate(template)
    claims = discover_specialized(template, build_topology(template), "sum")
    buckets, _, _, _ = compiler._ordinary_grouping(template, claims, {})
    ordinary = OrdinaryMembership(
        frozenset(
            member
            for (_, _, mode), members in buckets.items()
            if mode == "bp"
            for member in members
        )
    )
    return claims, ordinary, complete_specialized(claims, ordinary)


@pytest.mark.parametrize(
    "name",
    [
        "parameter_edge",
        "gfe",
        "mu3",
        "mixture",
        "hybrid_gfe",
        "recurrence",
        "recurrence_loop",
        "hmm",
        "ef",
        "gaussian",
    ],
)
def test_compiler_reuses_structural_drafts_and_partitions_factor_ownership(
    name, monkeypatch
):
    template, arrays = model(name)
    claims, _, drafts = stages(template)
    monkeypatch.setattr(compiler, "discover_specialized", lambda *args: claims)
    monkeypatch.setattr(compiler, "complete_specialized", lambda *args: drafts)
    arena = compiler.compile_template(template, arrays=arrays)
    for field, arena_field in (
        ("dirichlet", "dirichlet_groups"),
        ("gfe", "gfe_groups"),
        ("transition_gfe", "gfe_tm_groups"),
        ("hybrid_gfe", "hybrid_gfe_groups"),
        ("mixtures", "mix_groups"),
        ("recurrence", "recurrence_groups"),
        ("precision", "prec_groups"),
    ):
        for draft, plan in zip(
            getattr(drafts, field), getattr(arena, arena_field), strict=True
        ):
            assert draft.plan is plan
            assert {d.name for d in draft.arrays.destinations} <= set(arena.log_params)
    assert drafts.goal_parameters is arena.goal_parameter_plans
    claimed = [
        factor
        for owner in drafts.ownership
        for factor in (*owner.executed, *owner.absorbed)
    ]
    ordinary = {
        member
        for group in (*arena.groups, *arena.gauss_groups, *arena.ef_groups)
        if group.type != "equality"
        for member in group.members
    }
    assert len(claimed) == len(set(claimed))
    assert not set(claimed) & ordinary
    assert set(claimed) | ordinary == set(claims.topology.factors)
    assert drafts.ownership_problem is None


def test_completion_has_no_mutable_template_or_array_dependency():
    template, _ = mixed_goals(states=(2, 3, 2))
    claims, ordinary, expected = stages(template)
    for factor in template.factors.values():
        factor.edges.clear()
        factor.param_blobs.clear()
    template.factors.clear()
    template.variables.clear()
    template.blobs.clear()
    template.constraints.clear()
    actual = complete_specialized(claims, ordinary)
    assert_same(actual, expected)
    with pytest.raises(TypeError):
        claims.parameter_layouts["changed"] = None
    with pytest.raises(FrozenInstanceError):
        actual.gfe[0].plan.cards = (99, 99)
    with pytest.raises(TypeError):
        actual.gfe[0].plan.scatter_x[0] = 99
    with pytest.raises(TypeError):
        claims.topology.factors["obs0"].param_blobs["matrix"] = "changed"


def test_fixed_goals_and_mu3_overlays_have_distinct_ownership():
    template, _ = model("gfe")
    claims, _, drafts = stages(template)
    owner = next(o for o in drafts.ownership if o.plan == "gfe:2x2")
    assert owner.executed == ("obs",)
    assert owner.absorbed == ("goal",)
    assert owner.excluded_edges == ("x",)
    assert owner.parameter_owners == ()
    assert drafts.gfe[0].plan.goal_members == ()
    assert claims.excluded_edges == {"x"}

    template, _ = model("mu3")
    _, _, drafts = stages(template)
    owner = next(o for o in drafts.ownership if o.plan == "gfeLM:2x2")
    (overlay,) = drafts.mu3
    assert owner.executed == overlay.members == ("obs0", "obs1")
    assert owner.absorbed == ("goal0", "goal1")
    assert owner.parameter_owners == (overlay.name,)
    assert overlay.name not in {record.plan for record in drafts.ownership}
    assert overlay.evaluator == "cffg.mu3_is.v1"


def test_global_goal_owner_is_not_reintroduced_per_shape_bucket():
    template, _ = mixed_goals(states=(2, 3, 2))
    claims, _, drafts = stages(template)
    (owner,) = drafts.goal_parameters
    assert owner.prior_blob == "Cc"
    assert owner.group_rows == (("gfeC:3x2", (0, 1)), ("gfeC:3x3", (0,)))
    assert claims.excluded_edges == {"x0", "x1", "x2"}
    for record in drafts.ownership:
        if record.plan.startswith("gfeC:"):
            assert record.parameter_owners == ("goal:Cc",)
            assert len(record.parameter_edges) == len(record.executed)
            assert len(record.absorbed) == 2 * len(record.executed)


def test_hybrid_and_dirichlet_exclusions_keep_their_distinct_meaning():
    template, _ = model("hybrid_gfe")
    claims, _, drafts = stages(template)
    (owner,) = (o for o in drafts.ownership if o.plan.startswith("gfeH:"))
    assert owner.executed == ("emission",)
    assert owner.absorbed == ("goal",)
    assert owner.excluded_edges == ("x",)
    assert not drafts.mixtures
    assert claims.excluded_edges == {"x"}
    template, _ = model("parameter_edge")
    claims, _, drafts = stages(template)
    (owner,) = (o for o in drafts.ownership if o.plan.startswith("dirvmp:"))
    assert owner.executed == ("lik",)
    assert owner.absorbed == ("theta_prior",)
    assert owner.parameter_edges == ("theta",)
    assert not claims.excluded_edges


def test_recurrence_completion_uses_ordinary_membership_without_rediscovery():
    template, _ = model("recurrence")
    claims, ordinary, drafts = stages(template)
    assert ordinary.bp_members
    isolated = complete_specialized(claims, OrdinaryMembership(frozenset()))
    assert isolated.recurrence_components != drafts.recurrence_components
    assert all(c.acyclic for c in isolated.recurrence_components)


def test_constraint_payload_is_unresolved_until_family_validation():
    template, arrays = model("constraint")
    _, _, drafts = stages(template)
    (draft,) = drafts.constraints
    assert not hasattr(draft, "k") and not hasattr(draft, "table")
    assert draft.arrays.recipe is ArrayRecipe.CONSTRAINT
    assert [
        (source.factor, source.role, source.blob) for source in draft.arrays.inputs
    ] == [("c", "k", "kk")]
    assert draft.arrays.inputs[0].initializer_shape is None
    assert [(d.kind, d.name) for d in draft.arrays.destinations] == [("payload", "k")]
    plan = resolve_constraint(draft, arrays)
    assert plan.k == 1
    with pytest.raises(ValueError, match="scalar integer"):
        resolve_constraint(draft, {**arrays, "kk": np.array(1.5)})


def test_constraint_incidental_value_error_keeps_its_original_class():
    template, arrays = model("constraint")
    with pytest.raises(ValueError) as error:
        compiler.compile_template(template, arrays={**arrays, "kk": np.array(np.nan)})
    assert type(error.value) is ValueError
    assert str(error.value) == "cannot convert float NaN to integer"


def test_transition_gfe_retains_facing_factors_and_output_entropy():
    template, arrays = build_rollout(
        "transition-ownership",
        steps=2,
        past=0,
        d=np.full(4, 0.25),
        A=np.eye(4),
        c=np.full(4, 0.25),
        e=np.full(2, 0.5),
        Bs=np.full((2, 4, 4), 0.25),
        gfe=False,
        epistemic_b=True,
    )
    claims, _, drafts = stages(template)
    (owner,) = (owner for owner in drafts.ownership if owner.plan.startswith("gfeTM"))
    assert owner.executed == ("tm1", "tm2")
    assert owner.absorbed == owner.excluded_edges == ()
    assert not claims.excluded_edges
    assert not drafts.mu3 and not drafts.goal_parameters and not drafts.dirichlet
    arena = compiler.compile_template(template, arrays=arrays)
    assert {"zm1", "zm2"} <= {
        name for name, edge in arena.edges.items() if edge.interior
    }


def test_precision_requirements_and_specialized_conditioning_rejection():
    from cffg.core import PortRole, PortSpec

    template, arrays = gamma_gauss_prec(1)
    _, _, drafts = stages(template)
    (draft,) = drafts.precision
    assert draft.plan.members == ("gp0",)
    assert draft.arrays.recipe is ArrayRecipe.PRECISION
    assert draft.arrays.inputs[0].initializer_shape == (1,)
    compiler.compile_template(template, arrays=arrays)
    del template.factors["xprior0"]
    template.ports["data"] = PortSpec("data", "x0", PortRole.DATA)
    claims, _, _ = stages(template)
    assert "observed gaussian_prec outputs" in claims.conditioning_problems["data"]
    with pytest.raises(compiler.CompileError, match="observed gaussian_prec outputs"):
        compiler.compile_template(template, arrays=arrays)


def test_invalid_concentration_keeps_orphan_diagnostic_after_array_check():
    template, arrays = model("parameter_edge")
    with pytest.raises(compiler.CompileError) as error:
        compiler.compile_template(
            template, arrays={**arrays, "alpha0": np.zeros((3, 2))}
        )
    assert str(error.value) == (
        f"cannot compile {template.name!r}:\n  - Dirichlet prior 'theta_prior': concentrations must be finite and > 0"
        "\n  - dirichlet factors are parameter priors and must face a supported "
        "mean-field likelihood; unpaired: ['theta_prior']"
    )


@pytest.mark.parametrize(
    "name,detail",
    [
        ("gfe", (("substituted_role", "in"),)),
        ("gfe", (("substituted_role", "out"), ("parameter_message", ["unknown"]))),
        ("hybrid_gfe", (("substituted_role", "out"), ("unknown", {}))),
    ],
)
def test_late_composite_errors_do_not_preempt_array_resolution(
    name, detail, monkeypatch
):
    template, arrays = model(name)
    template.constraints = [
        replace(c, detail=detail) if c.kind is ConstraintKind.P_SUBSTITUTION else c
        for c in template.constraints
    ]

    class ResolutionFailure(Exception):
        pass

    def fail(*args):
        raise ResolutionFailure

    with monkeypatch.context() as patch:
        patch.setattr(compiler, "_resolve_arrays", fail)
        with pytest.raises(ResolutionFailure):
            compiler.compile_template(template, arrays=arrays)
    with pytest.raises(compiler.CompileError) as error:
        compiler.compile_template(template, arrays=arrays)
    if dict(detail).get("parameter_message"):
        assert "['unknown']" in str(error.value)


def test_delta_error_remains_after_numeric_group_validation():
    template, arrays = model("gaussian")
    variable = next(iter(template.variables))
    template.constraints.append(ConstraintSpec(ConstraintKind.DELTA, variable))
    _, _, drafts = stages(template)
    assert "delta constraint on Gaussian edge" in drafts.delta_problem
    bad = {key: value for key, value in arrays.items()}
    mean = next(
        f.param_blobs["mean"] for f in template.factors.values() if f.type == "gaussian"
    )
    bad[mean] = np.zeros(99)
    with pytest.raises(compiler.CompileError) as error:
        compiler.compile_template(template, arrays=bad)
    assert "delta constraint" not in str(error.value)


def test_registered_any_family_keeps_declared_ef_domain_routing():
    kernel = kernels.kernel_for("gamma")
    name = "specialized_test_any_gamma"
    declaration = replace(kernel.factor_type, name=name, family="any")
    kernels.register_kernel(
        replace(kernel, factor_type=declaration, impl_id="cffgtests.any_gamma")
    )
    try:
        template, arrays = gamma_poisson(1)
        template.factors["prior"] = replace(template.factors["prior"], type=name)
        arena = compiler.compile_template(template, arrays=arrays)
        (group,) = (g for g in arena.ef_groups if "prior" in g.members)
        assert group.type == name and group.family == "gamma"
        assert not arena.groups
    finally:
        kernels.unregister_kernel(name)


def test_composite_discovery_does_not_execute_replacement_kernel_shape_callbacks():
    template, arrays = model("gfe")
    original = kernels.kernel_for("matrix_cat")
    calls = []

    def fail_shape(sizes):
        calls.append(sizes)
        raise RuntimeError("shape callback executed")

    parameters = dict(original.binding.parameters)
    parameters["matrix"] = replace(parameters["matrix"], shape=fail_shape)
    replacement = replace(
        original, binding=replace(original.binding, parameters=parameters)
    )
    kernels.unregister_kernel("matrix_cat")
    try:
        kernels.register_kernel(replacement)
        _, _, drafts = stages(template)
        (matrix,) = (
            source for source in drafts.gfe[0].arrays.inputs if source.role == "matrix"
        )
        assert matrix.initializer_shape == (2, 2)
        compiler.compile_template(template, arrays=arrays)
        with pytest.raises(compiler.CompileError, match="no array override"):
            compiler.compile_template(template, arrays={})
        assert calls == []
        with pytest.raises(RuntimeError, match="shape callback executed"):
            compiler._blob_shape(template, "A")
        assert len(calls) == 1
    finally:
        kernels.unregister_kernel("matrix_cat")
        kernels.register_kernel(original)


def test_32_specialized_shapes_preserve_dtype_and_reconstruction_contracts():
    template, arrays = mixed_goals(states=tuple(range(2, 34)))
    claims, _, drafts = stages(template)
    assert len(drafts.gfe) == 32 and len(drafts.goal_parameters) == 1
    arena = compiler.compile_template(template, arrays=arrays)
    reconstructed = from_json(to_json(template))
    assert structure_hash(reconstructed) == structure_hash(template)
    restored_claims, _, restored_drafts = stages(reconstructed)
    assert claims.parameter_layouts == restored_claims.parameter_layouts
    assert claims.excluded_edges == restored_claims.excluded_edges
    assert_same(drafts, restored_drafts)
    other = compiler.compile_template(
        reconstructed,
        arrays={key: value.astype(np.float32) for key, value in arrays.items()},
    )
    for left, right in zip(arena.gfe_groups, other.gfe_groups, strict=True):
        assert_same(left, right)
    assert_same(arena.goal_parameter_plans, other.goal_parameter_plans)
