"""Registered drafts preserve structural ownership and late callback gates."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from gaussian_kernel_fixture import declaration as gaussian_declaration
from gaussian_kernel_fixture import model as gaussian_model
from test_compiler_contracts import assert_same, model
from test_discrete_kernels import CUSTOM, declaration
from test_discrete_kernels import model as discrete_model
from test_ef_engine import gamma_poisson

from cffg import compiler, kernels
from cffg.array_requirements import ArrayRecipe
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.plates import expand_plates
from cffg.core.serialize import from_json, to_json
from cffg.core.validate import validate
from cffg.registered.capabilities import snapshot_registry
from cffg.registered.contracts import EqualityReference
from cffg.registered.lowering import lower_registered
from cffg.specialized import (
    OrdinaryMembership,
    complete_specialized,
    discover_specialized,
)
from cffg.topology import build_topology


def stages(template, registry=None):
    template = expand_plates(template)
    validate(template)
    topology = build_topology(template)
    registry = snapshot_registry() if registry is None else registry
    claims = discover_specialized(template, topology, "sum", registry)
    drafts = lower_registered(template, topology, claims, "sum", registry)
    return claims, drafts


@pytest.fixture
def external():
    kernel = kernels.register_kernel(declaration())
    yield kernel
    kernels.unregister_kernel(CUSTOM)


@pytest.mark.parametrize("name", ["hmm", "gaussian", "ef", "recurrence", "gfe"])
def test_compiler_consumes_drafts_and_specialized_equality_routes(name, monkeypatch):
    template, arrays = model(name)
    captured = []

    def capture(*args):
        result = lower_registered(*args)
        captured.append((args[2], result))
        return result

    monkeypatch.setattr(compiler, "lower_registered", capture)
    arena = compiler.compile_template(template, arrays=arrays)
    claims, drafts = captured[0]
    specialized = complete_specialized(claims, OrdinaryMembership(drafts.bp_members))
    equality_of = {
        member: plan
        for plan in specialized.equalities.values()
        for member in plan.members
    }
    registered_members = set()
    for pending, plans in (
        (drafts.discrete, arena.groups),
        (drafts.gaussian, arena.gauss_groups),
        (drafts.ef, arena.ef_groups),
    ):
        assert len(pending) == len(plans)
        for draft, plan in zip(pending, plans, strict=True):
            if isinstance(draft, EqualityReference):
                assert_same(equality_of[draft.members[0]], plan)
                continue
            structural = getattr(draft, "plan", draft)
            registered_members.update(structural.members)
            assert structural.name == plan.name
            assert structural.members == plan.members
            assert structural.roles == plan.roles
            assert structural.gather is plan.gather
            assert structural.scatter is plan.scatter
            if hasattr(draft, "plan"):
                assert draft.plan is plan
            assert draft.arrays.recipe is ArrayRecipe.REGISTERED
            assert {d.name for d in draft.arrays.destinations} <= set(arena.log_params)
            assert structural.kernel is kernels.kernel_for(
                template.factors[plan.members[0]].type
            )
    assert registered_members.isdisjoint(claims.consumed)


@pytest.mark.parametrize("observed", [False, True])
def test_gaussian_observation_intent_has_no_assigned_rows(observed):
    kernel = kernels.register_kernel(gaussian_declaration())
    try:
        template, arrays = gaussian_model(observed=observed)
        _, drafts = stages(template)
        (draft,) = drafts.gaussian
        assert not hasattr(draft, "obs_rows")
        assert draft.kernel is kernel
        assert draft.kernel.binding.energy is kernel.binding.energy
        assert draft.kernel.binding.readout is kernel.binding.readout
        if observed:
            intent = draft.observation
            assert (intent.role, intent.role_index) == ("sensor", 1)
            (source,) = intent.sources
            assert (source.member, source.port, source.variable, source.dim) == (
                "joint",
                "sensor",
                "sensor",
                2,
            )
        else:
            assert draft.observation is None
        arena = compiler.compile_template(template, arrays=arrays)
        (plan,) = arena.gauss_groups
        if observed:
            np.testing.assert_array_equal(plan.obs_rows, [0])
            assert arena.gauss_data == {"sensor": (0, 2)}
        else:
            assert plan.obs_rows is None
    finally:
        kernels.unregister_kernel(kernel.factor_type.name)


def test_ef_observation_sources_preserve_member_order_without_rows():
    template, arrays = gamma_poisson(12)
    _, drafts = stages(template)
    observed = next(d for d in drafts.ef if getattr(d, "type", None) == "poisson_obs")
    assert observed.observation.role == "out"
    assert observed.observation.role_index is None
    assert not hasattr(observed, "obs_rows")
    assert tuple(s.member for s in observed.observation.sources) == observed.members
    arena = compiler.compile_template(template, arrays=arrays)
    for row, source in enumerate(observed.observation.sources):
        assert arena.ef_data[source.port] == (row, "poisson")


def test_registry_and_drafts_are_snapshots(external, monkeypatch):
    template, _ = discrete_model()
    registry = snapshot_registry()
    claims, drafts = stages(template, registry)
    interaction = next(d for d in drafts.discrete if d.plan.type == CUSTOM)
    spec = interaction.arrays.inputs[0].shape_requirement
    assert spec is not None
    assert all(
        s.shape_requirement.role_sizes is spec.role_sizes
        for s in interaction.arrays.inputs
    )
    expected_roles = dict(spec.role_sizes)
    replacement = replace(external, binding=replace(external.binding, vmp=None))
    monkeypatch.setitem(kernels.KERNELS, CUSTOM, replacement)
    assert registry.bindings[CUSTOM] is external
    assert interaction.plan.kernel is external
    assert_same(
        drafts, lower_registered(template, claims.topology, claims, "sum", registry)
    )
    template.factors.clear()
    template.variables.clear()
    assert spec.role_sizes == expected_roles
    assert interaction.plan.members == ("interaction",)
    with pytest.raises(TypeError):
        registry.bindings[CUSTOM] = replacement
    with pytest.raises(TypeError):
        spec.role_sizes["spin"] = 900
    with pytest.raises(TypeError):
        interaction.plan.gather[0, 0] = 900
    with pytest.raises(FrozenInstanceError):
        drafts.bp_members = frozenset()


def test_compiler_passes_one_registry_snapshot_to_admission_and_lowering(monkeypatch):
    template, arrays = model("hmm")
    seen = []

    def discover(*args):
        seen.append(args[3])
        return discover_specialized(*args)

    def lower(*args):
        seen.append(args[4])
        return lower_registered(*args)

    monkeypatch.setattr(compiler, "discover_specialized", discover)
    monkeypatch.setattr(compiler, "lower_registered", lower)
    compiler.compile_template(template, arrays=arrays)
    assert len(seen) == 2 and seen[0] is seen[1]


def test_shape_providers_and_packers_do_not_run_during_lowering(external, monkeypatch):
    template, arrays = discrete_model()
    calls = []

    def shape(sizes):
        calls.append(dict(sizes))
        raise ValueError("shape provider reached")

    def pack(*args):
        pytest.fail("packer must not precede shape validation")

    specs = dict(external.binding.parameters)
    specs["coupling"] = replace(specs["coupling"], shape=shape)
    kernel = replace(
        external, binding=replace(external.binding, parameters=specs, pack_params=pack)
    )
    monkeypatch.setitem(kernels.KERNELS, CUSTOM, kernel)
    _, drafts = stages(template)
    interaction = next(d for d in drafts.discrete if d.plan.type == CUSTOM)
    source = next(s for s in interaction.arrays.inputs if s.role == "coupling")
    assert source.initializer_shape is None
    assert source.shape_requirement.parameter is specs["coupling"]
    assert source.shape_requirement.role_sizes == {"spin": 2, "color": 3}
    assert calls == []
    with pytest.raises(compiler.CompileError, match="no array override"):
        compiler.compile_template(template)
    assert calls == []
    with pytest.raises(compiler.CompileError, match="shape provider reached"):
        compiler.compile_template(template, arrays=arrays)
    assert calls == [{"spin": 2, "color": 3}]


def test_parameter_path_error_stays_behind_blob_resolution(external, monkeypatch):
    template, arrays = discrete_model()
    specs = {
        role: replace(spec, group_key=True)
        for role, spec in external.binding.parameters.items()
    }
    kernel = replace(external, binding=replace(external.binding, parameters=specs))
    monkeypatch.setitem(kernels.KERNELS, CUSTOM, kernel)
    _, drafts = stages(template)
    interaction = next(d for d in drafts.discrete if d.plan.type == CUSTOM)
    assert (
        interaction.path_problem
        == "only one packed parameter may occupy the group-name key"
    )
    with pytest.raises(compiler.CompileError, match="no array override"):
        compiler.compile_template(template)
    with pytest.raises(ValueError, match="only one packed parameter") as failure:
        compiler.compile_template(template, arrays=arrays)
    assert type(failure.value) is ValueError


def test_registered_and_specialized_admission_errors_stay_interleaved(
    external, monkeypatch
):
    template, _ = discrete_model(mean_field=True)
    kernel = replace(external, binding=replace(external.binding, vmp=None))
    monkeypatch.setitem(kernels.KERNELS, CUSTOM, kernel)
    template.variables.update({r: VariableSpec(r, 2) for r in ("a", "b", "c")})
    template.ports.update({r: PortSpec(r, r, PortRole.GLUE) for r in ("a", "b", "c")})
    template.factors["eq"] = FactorSpec(
        "eq", "equality", {r: r for r in ("a", "b", "c")}
    )
    template.constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "eq"))
    claims, _ = stages(template)
    expected = (
        (
            "factor 'interaction': kernel 'cffgtests.bilinear_spin_color' does not accept "
            "['mean_field'] beads (conjugate BP is already exact on EF subgraphs)"
        ),
        (
            "factor 'eq': mean-field bead on an equality node is ill-defined "
            "(geometric expectation of a hard constraint)"
        ),
    )
    assert claims.admission_problems == expected
    with pytest.raises(compiler.CompileError) as failure:
        compiler.compile_template(template)
    assert str(
        failure.value
    ) == "cannot compile 'registered_bilinear':\n  - " + "\n  - ".join(expected)


def test_32_registered_groups_keep_numeric_order_under_float32_and_reconstruction():
    template = Template("many_registered_shapes")
    arrays = {}
    for card in range(2, 34):
        name = f"f{card}"
        template.variables[name] = VariableSpec(name, card)
        template.ports[name] = PortSpec(name, name, PortRole.GLUE)
        template.factors[name] = FactorSpec(
            name, "categorical", {"out": name}, param_blobs={"statistic": name}
        )
        template.blobs[name] = BlobRef(name, f"{name}.npy")
        arrays[name] = np.full(card, 1 / card, dtype=np.float32)
    _, drafts = stages(template)
    assert [d.plan.cards for d in drafts.discrete] == [(c,) for c in range(2, 34)]
    _, restored = stages(from_json(to_json(template)))
    assert_same(drafts, restored)
    narrow = compiler.compile_template(template, arrays=arrays)
    wide = compiler.compile_template(
        template, arrays={k: v.astype(np.float64) for k, v in arrays.items()}
    )
    assert_same(narrow, wide)


@pytest.mark.parametrize(
    "name,kind", [("hmm", "matrix_cat"), ("gaussian", "gaussian"), ("ef", "gamma")]
)
def test_registered_packers_keep_the_complete_resolved_mapping(name, kind, monkeypatch):
    template, arrays = model(name)
    template.variables["extra"] = VariableSpec("extra", 2)
    template.ports["extra"] = PortSpec("extra", "extra", PortRole.GLUE)
    template.factors["extra"] = FactorSpec(
        "extra", "categorical", {"out": "extra"}, param_blobs={"statistic": "extra"}
    )
    template.blobs["extra"] = BlobRef("extra", "extra.npy")
    arrays["extra"] = np.array([0.3, 0.7])
    expected = {
        blob
        for factor in template.factors.values()
        for blob in factor.param_blobs.values()
    }
    kernel = kernels.kernel_for(kind)
    calls = []

    def pack(t, members, resolved):
        assert isinstance(members, list)
        assert set(resolved) == expected
        np.testing.assert_array_equal(resolved["extra"], arrays["extra"])
        calls.append(tuple(members))
        return kernel.pack_params(t, members, resolved)

    monkeypatch.setitem(
        kernels.KERNELS,
        kind,
        replace(kernel, binding=replace(kernel.binding, pack_params=pack)),
    )
    compiler.compile_template(template, arrays=arrays)
    assert calls


@pytest.mark.parametrize("gaussian_output", [False, True])
def test_invalid_conditioning_preserves_observed_intent_and_late_error(gaussian_output):
    original = kernels.kernel_for("poisson_obs")
    kernel = replace(
        original,
        factor_type=replace(
            original.factor_type,
            name="pending_observation",
            family="any",
            edge_roles=("rate", "out"),
        ),
        impl_id="cffgtests.pending_observation",
    )
    kernels.register_kernel(kernel)
    try:
        template, arrays = gamma_poisson(1)
        template.factors["obs0"] = replace(
            template.factors["obs0"], type=kernel.factor_type.name
        )
        template.variables["x0"] = (
            VariableSpec("x0", dim=1) if gaussian_output else VariableSpec("x0", 2)
        )
        _, drafts = stages(template)
        observed = next(
            d for d in drafts.ef if getattr(d, "type", None) == kernel.factor_type.name
        )
        assert observed.observation is not None
        assert observed.observation.sources[0].port is None
        expected_problems = (
            (
                "port 'count0': Gaussian DATA port on 'obs0'/'out' requires a registered conditioning role",
            )
            if gaussian_output
            else ()
        )
        assert drafts.conditioning_problems == expected_problems
        with pytest.raises(compiler.CompileError, match="blob 'a0': no array override"):
            compiler.compile_template(template)
        if gaussian_output:
            with pytest.raises(compiler.CompileError) as failure:
                compiler.compile_template(template, arrays=arrays)
            assert (
                str(failure.value)
                == "cannot compile 'gamma_poisson_1':\n  - " + expected_problems[0]
            )
        else:
            with pytest.raises(KeyError) as failure:
                compiler.compile_template(template, arrays=arrays)
            assert failure.value.args == ("x0",)
    finally:
        kernels.unregister_kernel(kernel.factor_type.name)
