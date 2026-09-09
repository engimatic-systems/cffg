"""Final assembly joins complete stage values without rediscovering the model."""

from dataclasses import replace

import numpy as np
import pytest
from test_compiler_contracts import IDENTITIES, assert_same, model
from test_registered import stages
from test_shared_goals import mixed_goals

from cffg import compiler, kernels, topology
from cffg.assembly import assemble_arena
from cffg.compiler_errors import CompileError
from cffg.core.plates import expand_plates
from cffg.resolution import sources
from cffg.resolution.resolve import resolve_compile_data
from cffg.specialized import OrdinaryMembership, complete_specialized


def inputs(template, arrays):
    template = expand_plates(template)
    claims, registered = stages(template)
    specialized = complete_specialized(
        claims, OrdinaryMembership(registered.bp_members)
    )
    data = resolve_compile_data(
        template, arrays, None, claims.topology, registered, specialized
    )
    return claims.topology, registered, specialized, data, "sum"


@pytest.mark.parametrize("name", IDENTITIES)
def test_assembly_requires_only_complete_stage_values(name, monkeypatch):
    template, arrays = model(name)
    if arrays is not None:
        arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
    expected = compiler.compile_template(template, arrays=arrays)
    arguments = inputs(template, arrays)

    def forbidden(*args, **kwargs):
        pytest.fail("assembly must not repeat a completed compiler stage")

    monkeypatch.setattr(sources, "resolve_arrays", forbidden)
    monkeypatch.setattr(kernels, "packed_parameters", forbidden)
    monkeypatch.setattr(topology, "prefix_offsets", forbidden)
    template.factors.clear()
    template.variables.clear()
    if arrays is not None:
        for value in arrays.values():
            value[...] = 0
    arena = assemble_arena(*arguments)
    assert_same(arena, expected)
    data = arguments[3]
    assert arena.log_params is data.log_params
    assert arena.message_starts is data.message_starts
    assert arena.g_eta_starts is data.g_eta_starts
    assert arena.g_lam_starts is data.g_lam_starts
    assert arena.g_obs_starts is data.g_obs_starts
    assert arena.member_of is data.member_of


def test_compiler_delegates_final_construction_once(monkeypatch):
    template, arrays = model("hmm")
    assembled = []

    def capture(*args):
        result = assemble_arena(*args)
        assembled.append(result)
        return result

    monkeypatch.setattr(compiler, "assemble_arena", capture)
    arena = compiler.compile_template(template, arrays=arrays)
    assert len(assembled) == 1
    assert arena is assembled[0]


def test_plan_names_can_repeat_across_execution_families():
    template, arrays = model("constraint")
    name = "bp:categorical:2"
    template.factors[name] = replace(template.factors.pop("c"), name=name)
    arena = compiler.compile_template(template, arrays=arrays)
    assert arena.groups[0].name == name
    assert arena.constraint_groups[0].name == name
    assert arena.groups[0].members != arena.constraint_groups[0].members


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_registered_execution_is_rejected(duplicate):
    layout, registered, drafts, data, semiring = inputs(*model("hmm"))
    declarations, plans = registered.discrete, data.groups
    if duplicate:
        declarations, plans = declarations + declarations[:1], plans + plans[:1]
    else:
        declarations, plans = declarations[1:], plans[1:]
    registered = replace(registered, discrete=declarations)
    data = replace(data, groups=plans)
    with pytest.raises(CompileError, match="execution owner"):
        assemble_arena(layout, registered, drafts, data, semiring)


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_specialized_execution_is_rejected(duplicate):
    layout, registered, drafts, data, semiring = inputs(*model("gfe"))
    owners = drafts.ownership
    if duplicate:
        owners = owners + owners[:1]
    else:
        owners = tuple(
            owner for owner in owners if owner.plan != drafts.gfe[0].plan.name
        )
    with pytest.raises(
        CompileError, match="multiple execution owners|no specialized owner"
    ):
        assemble_arena(
            layout, registered, replace(drafts, ownership=owners), data, semiring
        )


@pytest.mark.parametrize("kind", ["exclusion", "parameter", "constraint", "reference"])
def test_objective_declarations_must_match_their_owners(kind):
    name = "parameter_edge" if kind in ("parameter", "reference") else "gfe"
    layout, registered, drafts, data, semiring = inputs(*model(name))
    if kind == "exclusion":
        drafts = replace(drafts, excluded_edges=frozenset())
        match = "entropy exclusions"
    elif kind == "parameter":
        drafts = replace(drafts, parameter_layouts={})
        match = "parameter-edge tags"
    else:
        field = "constraints" if kind == "constraint" else "parameter_owners"
        value = () if kind == "constraint" else ("missing-owner",)
        owners = (replace(drafts.ownership[0], **{field: value}), *drafts.ownership[1:])
        drafts = replace(drafts, ownership=owners)
        match = "constraint declarations|parameter-owner references"
    with pytest.raises(CompileError, match=match):
        assemble_arena(layout, registered, drafts, data, semiring)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "carrier", "references"])
def test_shared_goal_ownership_must_cover_each_original_contribution_once(kind):
    layout, registered, drafts, data, semiring = inputs(*mixed_goals())
    (plan,) = data.goal_parameters
    if kind == "duplicate":
        plans = (plan, plan)
        match = "owned twice|duplicate parameter owner"
    elif kind == "missing":
        plans = ()
        match = "no parameter owner"
    elif kind == "carrier":
        plans = (replace(plan, parameter_edges=tuple(reversed(plan.parameter_edges))),)
        match = "parameter carriers"
    else:
        plans = (plan,)
        owners = tuple(
            replace(owner, parameter_owners=())
            if owner.plan == plan.group_rows[0][0]
            else owner
            for owner in drafts.ownership
        )
        drafts = replace(drafts, ownership=owners)
        match = "parameter-owner references"
    drafts = replace(drafts, goal_parameters=plans)
    data = replace(data, goal_parameters=plans)
    with pytest.raises(CompileError, match=match):
        assemble_arena(layout, registered, drafts, data, semiring)


@pytest.mark.parametrize("invalid_rows", ["range", "shape", "dtype"])
def test_mu3_overlay_is_a_parameter_owner_not_duplicate_factor_execution(invalid_rows):
    arguments = inputs(*model("mu3"))
    arena = assemble_arena(*arguments)
    layout, registered, drafts, data, semiring = arguments
    assert arena.mu3_groups
    (first, *rest) = data.mu3
    rows = {
        "range": np.full(first.rows.shape, 1000),
        "shape": np.zeros((*first.rows.shape, 1), dtype=np.int64),
        "dtype": np.asarray(first.rows, dtype=float),
    }[invalid_rows]
    plans = (replace(first, rows=rows), *rest)
    with pytest.raises(CompileError, match="invalid contributor rows"):
        assemble_arena(
            layout,
            registered,
            replace(drafts, mu3=plans),
            replace(data, mu3=plans),
            semiring,
        )


def test_shared_goal_references_across_groups_are_not_duplicate_ownership():
    arguments = inputs(*mixed_goals())
    arena = assemble_arena(*arguments)
    assert len(arena.gfe_groups) == 2
    assert len(arena.goal_parameter_plans) == 1
    assert arena.goal_parameter_plans is arguments[3].goal_parameters


@pytest.mark.parametrize("kind", ["group", "prefix", "semiring", "overlay"])
def test_mismatched_stage_values_fail_before_arena_construction(kind):
    name = "mu3" if kind == "overlay" else "hmm"
    layout, registered, drafts, data, semiring = inputs(*model(name))
    if kind == "group":
        data = replace(data, groups=data.groups[1:])
        match = "group count"
    elif kind == "prefix":
        data = replace(data, message_starts=np.zeros(1, dtype=np.int64))
        match = "message_starts length"
    elif kind == "semiring":
        semiring = "max"
        match = "semiring disagrees"
    else:
        data = replace(data, mu3=())
        match = "mu3 owners disagree"
    with pytest.raises(CompileError, match=match):
        assemble_arena(layout, registered, drafts, data, semiring)
