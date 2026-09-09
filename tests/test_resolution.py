"""Data-resolution contracts, including the final validation checkpoints."""

from dataclasses import fields, replace

import numpy as np
import pytest
from test_compiler_contracts import IDENTITIES, assert_same, model
from test_ef_engine import gamma_poisson
from test_registered import stages

from cffg import compiler, kernels
from cffg._immutable import FrozenArray, FrozenMap
from cffg.compiler import CompileError, _resolve_arrays, compile_template
from cffg.core import (
    BlobRef,
    FactorSpec,
    LearnableSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.plates import expand_plates
from cffg.registered.contracts import EqualityReference
from cffg.resolution.addresses import AddressBuilder, resolve_learnables
from cffg.resolution.resolve import resolve_compile_data
from cffg.specialized import OrdinaryMembership, complete_specialized


def invalid_unused_learnable(template):
    template.blobs["unused"] = BlobRef("unused", "unused.npy")
    template.learnables["bad"] = LearnableSpec("bad", "dirichlet", ("unused",))


@pytest.mark.parametrize(
    "cardinality,error,match",
    [
        (2**63, OverflowError, "too large"),
        (2, CompileError, r"learnable 'bad': no factor reads blobs \['unused'\]"),
    ],
)
def test_prefix_conversion_precedes_learning_validation(cardinality, error, match):
    names = ("a", "b", "c")
    template = Template(
        "prefix-before-learning",
        variables={name: VariableSpec(name, cardinality) for name in names},
        factors={"eq": FactorSpec("eq", "equality", {name: name for name in names})},
        ports={name: PortSpec(name, name, PortRole.GLUE) for name in names},
    )
    invalid_unused_learnable(template)
    with pytest.raises(error, match=match):
        compile_template(template)


@pytest.mark.parametrize("invalid_learning", [False, True])
def test_numeric_publication_follows_learning_validation(invalid_learning, monkeypatch):
    template, arrays = gamma_poisson(1)
    kernel = kernels.kernel_for("gamma")

    def pack(*args):
        return {"eta": np.array([[object(), object()]], dtype=object)}

    monkeypatch.setitem(
        kernels.KERNELS,
        "gamma",
        replace(kernel, binding=replace(kernel.binding, pack_params=pack)),
    )
    if invalid_learning:
        invalid_unused_learnable(template)
        with pytest.raises(CompileError, match="no factor reads blobs"):
            compile_template(template, arrays=arrays)
    else:
        with pytest.raises(
            TypeError, match="published arrays cannot contain Python objects"
        ):
            compile_template(template, arrays=arrays)


@pytest.mark.parametrize("name", IDENTITIES)
def test_declared_addresses_and_learning_match_existing_compiler(name):
    template, arrays = model(name)
    template = expand_plates(template)
    claims, registered = stages(template)
    specialized = complete_specialized(
        claims, OrdinaryMembership(registered.bp_members)
    )
    requirements = [
        draft.arrays
        for drafts in (
            specialized.gfe,
            specialized.transition_gfe,
            registered.gaussian,
            specialized.mixtures,
            specialized.recurrence,
            specialized.precision,
            registered.discrete,
        )
        for draft in drafts
        if not isinstance(draft, EqualityReference)
    ]
    builder = AddressBuilder()
    for requirement in requirements:
        builder.add(template, requirement)
    arena = compile_template(template, arrays=arrays)
    assert_same(builder.member_of, arena.member_of)
    assert builder.writable_members == arena.writable_members
    resolved = _resolve_arrays(template, arrays, None)
    assert_same(
        tuple(resolve_learnables(template, resolved, builder)), arena.learn_plans
    )
    assert [(s.factor, s.role, s.blob) for s in registered.sources] == [
        (factor.name, role, blob)
        for factor in template.factors.values()
        for role, blob in factor.param_blobs.items()
    ]
    data = resolve_compile_data(
        template, arrays, None, claims.topology, registered, specialized
    )
    aliases = {"mu3": "mu3_groups", "goal_parameters": "goal_parameter_plans"}
    for field in fields(data):
        assert_same(
            getattr(data, field.name),
            getattr(arena, aliases.get(field.name, field.name)),
        )


def test_initializer_fallback_is_not_the_composite_packing_shape():
    template, _ = model("gfe")
    claims, registered = stages(template)
    specialized = complete_specialized(
        claims, OrdinaryMembership(registered.bp_members)
    )
    (draft,) = specialized.gfe
    member = draft.plan.members[0]
    packed_input = next(
        s for s in draft.arrays.inputs if s.factor == member and s.role == "matrix"
    )
    source = next(
        s for s in registered.sources if s.factor == member and s.role == "matrix"
    )
    assert packed_input.initializer_shape == (2, 2)
    assert source.shape is not None
    assert source.initializer_fallback is None
    assert (member, "matrix") not in specialized.initializer_fallbacks


@pytest.mark.parametrize("name", ["gaussian", "ef", "learning"])
def test_arena_reuses_resolved_publication_without_mutable_aliases(name, monkeypatch):
    template, arrays = model(name)
    arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
    published = []

    def capture(*args, **kwargs):
        data = resolve_compile_data(*args, **kwargs)
        published.append(data)
        return data

    monkeypatch.setattr(compiler, "resolve_compile_data", capture)
    arena = compile_template(template, arrays=arrays)
    (data,) = published
    aliases = {"mu3": "mu3_groups", "goal_parameters": "goal_parameter_plans"}
    for field in fields(data):
        value = getattr(data, field.name)
        if isinstance(value, (FrozenArray, FrozenMap, tuple)):
            assert value is getattr(arena, aliases.get(field.name, field.name))

    expected = {key: np.asarray(value) for key, value in data.log_params.items()}
    for source in arrays.values():
        source[...] = 0
    for key, value in data.log_params.items():
        detached = np.asarray(value)
        detached[...] = 0
        with pytest.raises(TypeError, match="read-only"):
            value[...] = 0
        np.testing.assert_array_equal(value, expected[key])
        np.testing.assert_array_equal(arena.log_params[key], expected[key])
