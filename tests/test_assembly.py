"""Final assembly joins complete stage values without rediscovering the model."""

from dataclasses import replace

import numpy as np
import pytest
from test_compiler_contracts import IDENTITIES, assert_same, model
from test_registered import stages

from cffg import compiler, kernels, topology
from cffg.assembly import assemble_arena
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
