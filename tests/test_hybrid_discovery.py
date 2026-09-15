"""D12 graph claims precede ordinary grouping, without numerical execution."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from test_compiler_contracts import assert_same
from test_hybrid_declarations import declared, exposed, update

from cffg import compiler
from cffg.array_requirements import ArrayRecipe
from cffg.core import (
    BlobRef,
    FactorSpec,
    HybridAdapterSpec,
    HybridComponentSpec,
    HybridRegionSpec,
    HybridSeparatorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
    validate,
)
from cffg.core.compose import merge, scoped
from cffg.core.hybrid import map_component
from cffg.registered.capabilities import snapshot_registry
from cffg.registered.contracts import DiscreteDraft
from cffg.registered.lowering import lower_registered
from cffg.specialized import (
    OrdinaryMembership,
    complete_specialized,
    discover_specialized,
    validate_admission,
)
from cffg.specialized import discovery
from cffg.topology import build_topology


def stages(template, registry=None):
    validate(template)
    topology = build_topology(template)
    claims = discover_specialized(template, topology, "sum", registry)
    validate_admission(template.name, claims)
    registered = lower_registered(template, topology, claims, "sum", claims.registry)
    drafts = complete_specialized(claims, OrdinaryMembership(registered.bp_members))
    return claims, registered, drafts


def ordinary_members(registered):
    return {member for draft in (*registered.discrete, *registered.gaussian, *registered.ef)
            for member in (draft.plan if isinstance(draft, DiscreteDraft) else draft).members}


def assert_adapter_slots(claims, component):
    declared_adapters = {a.name: a for a in component.plan.declaration.adapters}
    for adapter in component.plan.adapters:
        variable = declared_adapters[adapter.name].variable
        edge = claims.topology.edges[variable]
        assert adapter.gather == (edge.slot0, edge.slot1)
        inside, = [end for end in claims.topology.endpoints[variable]
                   if end[0] in component.ownership.executed]
        assert adapter.scatter == claims.topology.slot_of[inside]
        assert variable not in claims.excluded_edges
        for outside in claims.topology.endpoints[variable]:
            if outside != inside:
                assert claims.topology.opposite[outside] == adapter.scatter


def test_graph_claims_partition_ordinary_factors_and_entropy_edges():
    claims, registered, drafts = stages(declared())
    component, = drafts.hybrid_components
    assert component.plan.declaration == declared().hybrid_components["association"]
    assert claims.consumed == {"r", "d", "pz", "px", "eq"}
    assert not claims.recurrence and not claims.mixtures and not claims.equalities
    assert not drafts.recurrence_components and not drafts.mixtures and not drafts.equalities
    assert ordinary_members(registered) == {"observe"}
    assert claims.excluded_edges == {"z0", "z1", "xr", "xd", "xp"}
    assert component.ownership == drafts.ownership[0]
    assert set(component.ownership.executed) == claims.consumed
    assert not component.ownership.parameter_owners
    assert not component.ownership.parameter_edges
    assert drafts.ownership_problem is None
    assert {o.factor for o in component.objectives if o.kind == "region"} == {"r", "d", "pz", "px"}
    assert {o.name for o in component.objectives if o.kind == "separator"} == {"left", "middle", "right"}
    assert {o.name for o in component.objectives if o.kind == "adapter"} == {"readout"}
    copies = {s.name: s for s in component.plan.separators}
    assert copies["left"].state_variables == copies["middle"].state_variables == ("xd", "xp", "xr")
    assert copies["middle"].regime_variables == copies["right"].regime_variables == ("z1",)
    for region in component.plan.regions:
        ports = next(r.ports for r in component.plan.declaration.regions if r.name == region.name)
        assert tuple(component.plan.separators[i].name for i in region.separator_indices) == ports
    assert component.plan.coordinate_requirements is None
    assert component.plan.scratch_requirements is None
    assert component.plan.schedule_stage == "deferred"
    assert_adapter_slots(claims, component)


def test_sources_keep_original_roles_order_shapes_and_parameter_conventions():
    template = declared()
    template.factors = dict(reversed(tuple(template.factors.items())))
    for name, factor in tuple(template.factors.items()):
        template.factors[name] = replace(factor, param_blobs=dict(reversed(tuple(factor.param_blobs.items()))))
    claims, registered, drafts = stages(template)
    expected = [(f.name, role, blob) for f in template.factors.values() for role, blob in f.param_blobs.items()]
    assert [(s.factor, s.role, s.blob) for s in registered.sources] == expected
    component, = drafts.hybrid_components
    inputs = [source for arrays in component.arrays for source in arrays.inputs]
    assert [(s.factor, s.role, s.blob) for s in inputs] == [s for s in expected if s[0] in claims.consumed]
    source_of = {(s.factor, s.role): s for s in registered.sources}
    for source in inputs:
        assert_same(source.shape_requirement, source_of[source.factor, source.role].shape)
        assert source.initializer_shape == claims.initializer_shapes[source.factor, source.role]
    shapes = {(s.factor, s.role): s.initializer_shape for s in inputs}
    assert shapes["r", "weights"] == (3, 2, 2)
    assert shapes["r", "offsets"] == (3, 2)
    assert shapes["d", "matrix"] == (3, 2, 2)
    assert shapes["d", "cov"] == (3, 2, 2)
    requirements = {a.members[0].member: a for a in component.arrays}
    assert requirements["r"].recipe is ArrayRecipe.RECURRENCE
    assert requirements["d"].recipe is ArrayRecipe.MIXTURE
    assert requirements["pz"].recipe is requirements["px"].recipe is ArrayRecipe.REGISTERED
    assert requirements["pz"].members[0].runtime["statistic"].transform == "probability_to_log"
    assert requirements["pz"].members[0].runtime["statistic"].probability_axis == 0
    assert requirements["pz"].members[0].runtime["statistic"].key == requirements["pz"].owner
    for factor in ("px", "d"):
        assert requirements[factor].members[0].runtime["cov"].transform == "covariance_to_precision"
        assert requirements[factor].members[0].runtime["cov"].key.endswith(":W")
    for arrays in component.arrays:
        member, = arrays.members
        assert member.index == 0
        assert not member.writable and not member.statistic_source
        assert set(member.runtime) == set(template.factors[member.member].param_blobs)
        assert {p.key for p in member.runtime.values()} == {d.name for d in arrays.destinations}


def test_reordering_and_arbitrary_renaming_do_not_choose_membership():
    original = declared()
    claims, _, drafts = stages(original)
    reordered = declared()
    reordered.factors = dict(reversed(tuple(reordered.factors.items())))
    spec = reordered.hybrid_components["association"]
    update(reordered, regions=tuple(reversed(spec.regions)), separators=tuple(reversed(spec.separators)))
    reordered_claims, _, reordered_drafts = stages(reordered)
    assert reordered_claims.consumed == claims.consumed
    assert_same(reordered_drafts.hybrid_components[0].plan.regions, drafts.hybrid_components[0].plan.regions)
    assert_same(reordered_drafts.hybrid_components[0].plan.separators, drafts.hybrid_components[0].plan.separators)

    names = set(original.variables) | set(original.factors) | set(original.blobs) | set(original.ports)
    names |= {spec.name, *(s.name for s in spec.separators), *(r.name for r in spec.regions), *(a.name for a in spec.adapters)}
    mapping = {name: f"opaque_{i:03}" for i, name in enumerate(reversed(sorted(names)))}
    rename = mapping.__getitem__
    renamed = Template("opaque")
    renamed.variables = {rename(n): replace(v, name=rename(n)) for n, v in original.variables.items()}
    renamed.factors = {
        rename(n): replace(f, name=rename(n), edges={r: rename(v) for r, v in f.edges.items()},
                           param_blobs={r: rename(b) for r, b in f.param_blobs.items()})
        for n, f in original.factors.items()
    }
    renamed.blobs = {rename(n): replace(b, name=rename(n)) for n, b in original.blobs.items()}
    renamed.ports = {rename(n): replace(p, name=rename(n), variable=rename(p.variable)) for n, p in original.ports.items()}
    renamed.hybrid_components = {rename(spec.name): map_component(spec, rename)}
    actual, registered, actual_drafts = stages(renamed)
    assert actual.consumed == set(map(rename, claims.consumed))
    assert actual.excluded_edges == set(map(rename, claims.excluded_edges))
    assert ordinary_members(registered) == {rename("observe")}
    assert_adapter_slots(actual, actual_drafts.hybrid_components[0])


def test_multiple_components_and_both_ordinary_adapter_domains():
    left = exposed()
    left.variables["prior_z"] = VariableSpec("prior_z", cardinality=3)
    left.factors["ordinary"] = FactorSpec("ordinary", "matrix_cat", {"in": "prior_z", "out": "z0"}, param_blobs={"matrix": "probabilities"})
    left.blobs["probabilities"] = BlobRef("probabilities", "absent.npy")
    left.ports["label"] = replace(left.ports["label"], variable="prior_z")
    template = merge(scoped(left, "a"), scoped(declared(), "b"), "pair")
    claims, registered, drafts = stages(template)
    assert len(claims.hybrid_components) == len(drafts.hybrid_components) == 2
    assert ordinary_members(registered) == {"a/ordinary", "a/observe", "b/observe"}
    owners = [member for component in drafts.hybrid_components for member in component.ownership.executed]
    assert len(owners) == len(set(owners)) == len(claims.consumed)
    for component in drafts.hybrid_components:
        assert_adapter_slots(claims, component)
    assert {a.domain for a in drafts.hybrid_components[0].plan.adapters} == {"categorical", "gaussian"}
    assert not {"a/z0", "a/xp", "a/y", "b/y"} & claims.excluded_edges


def test_selected_gaussian_unary_preserves_mixture_roles():
    template = exposed()
    spec = template.hybrid_components["association"]
    template.factors["site"] = FactorSpec("site", "gaussian_mixture", {"out": "xp", "selector": "z0"}, param_blobs={"mean": "site_m", "cov": "site_V"})
    template.blobs.update({n: BlobRef(n, "absent.npy") for n in ("site_m", "site_V")})
    del template.ports["state"], template.ports["label"]
    update(template, regions=(*spec.regions, HybridRegionSpec("site_owner", "unary", "site", ("left",))), adapters=spec.adapters[:1])
    claims, _, drafts = stages(template)
    assert not claims.mixtures
    arrays, = [a for a in drafts.hybrid_components[0].arrays if a.members[0].member == "site"]
    assert arrays.recipe is ArrayRecipe.MIXTURE
    assert [(s.role, s.initializer_shape) for s in arrays.inputs] == [("mean", (3, 2)), ("cov", (3, 2, 2))]


def branched(branches=2, cardinality=3, dim=2):
    """Several R/H/D pairs share one Y; equality trees keep FFG degree two."""
    template = Template("branched")
    separators, regions, adapters, equalities = [], [], [], []
    z_leaves, x_leaves = ["z_prior"], ["x_prior"]

    def variable(name, *, gaussian=False):
        template.variables[name] = VariableSpec(name, dim=dim) if gaussian else VariableSpec(name, cardinality=cardinality)
        return name

    def factor(name, kind, edges, roles):
        blobs = {role: f"{name}_{role}" for role in roles}
        template.factors[name] = FactorSpec(name, kind, edges, param_blobs=blobs)
        template.blobs.update({b: BlobRef(b, "absent.npy") for b in blobs.values()})

    factor("prior_z", "categorical", {"out": variable("z_prior")}, ("statistic",))
    factor("prior_x", "gaussian", {"out": variable("x_prior", gaussian=True)}, ("mean", "cov"))
    support = (0, cardinality - 1)
    separators.append(HybridSeparatorSpec("shared", "z_prior", "x_prior", support))
    regions.extend((HybridRegionSpec("prior_z", "unary", "prior_z", ("shared",)),
                    HybridRegionSpec("prior_x", "unary", "prior_x", ("shared",))))
    for i in range(branches):
        zi, zj = variable(f"zi{i}"), variable(f"zj{i}")
        xr, xd, y, obs = (variable(f"{n}{i}", gaussian=True) for n in ("xr", "xd", "y", "obs"))
        z_leaves.append(zi)
        x_leaves.extend((xr, xd))
        factor(f"r{i}", "recurrence_link", {"in": zi, "out": zj, "drive": xr}, ("weights", "offsets"))
        factor(f"d{i}", "gaussian_mixture_linear", {"in": xd, "out": y, "selector": zj}, ("matrix", "offset", "cov"))
        factor(f"observe{i}", "gaussian_linear", {"in": y, "out": obs}, ("matrix", "offset", "cov"))
        template.ports[f"sensor{i}"] = PortSpec(f"sensor{i}", obs, PortRole.DATA)
        separators.extend((HybridSeparatorSpec(f"h{i}", zj, xr, support), HybridSeparatorSpec(f"next{i}", zj, y, support)))
        regions.extend((HybridRegionSpec(f"switch{i}", "recurrence", f"r{i}", ("shared", f"h{i}")),
                        HybridRegionSpec(f"channel{i}", "dynamics", f"d{i}", (f"h{i}", f"next{i}"))))
        adapters.append(HybridAdapterSpec(f"readout{i}", f"next{i}", y))

    for prefix, leaves in (("z", z_leaves), ("x", x_leaves)):
        remaining = list(leaves)
        index = 0
        while len(remaining) > 3:
            bridge = variable(f"{prefix}_bridge{index}", gaussian=prefix == "x")
            name = f"{prefix}_eq{index}"
            factor(name, "equality", dict(zip(("a", "b", "c"), (*remaining[:2], bridge))), ())
            equalities.append(name)
            remaining = [bridge, *remaining[2:]]
            index += 1
        name = f"{prefix}_eq{index}"
        factor(name, "equality", dict(zip(("a", "b", "c"), remaining)), ())
        equalities.append(name)
    template.hybrid_components["branches"] = HybridComponentSpec("branches", separators=separators, regions=regions, adapters=adapters, equalities=equalities)
    return template


@pytest.mark.parametrize("branches,cardinality,dim", [(2, 3, 2), (40, 3, 2), (2, 90, 2), (2, 3, 80)])
def test_general_incidence_sparse_labels_and_independent_structural_scale(branches, cardinality, dim):
    claims, registered, drafts = stages(branched(branches, cardinality, dim))
    component, = drafts.hybrid_components
    plan = component.plan
    assert len(plan.regions) == branches * 2 + 2
    assert len(plan.separators) == branches * 2 + 1
    assert len(plan.adapters) == branches
    assert ordinary_members(registered) == {f"observe{i}" for i in range(branches)}
    assert not drafts.recurrence_components
    assert all(s.cardinality == cardinality and s.dim == dim for s in plan.separators)
    assert all(s.active_support == (0, cardinality - 1) for s in plan.declaration.separators)
    assert sum(r.separator_indices.size for r in plan.regions) == 4 * branches + 2
    shared = next(i for i, s in enumerate(plan.separators) if s.name == "shared")
    assert sum(shared in r.separator_indices for r in plan.regions) == branches + 2
    assert plan.coordinate_requirements is plan.scratch_requirements is None
    weights = next(s for a in component.arrays for s in a.inputs if s.factor == "r0" and s.role == "weights")
    assert weights.initializer_shape == (cardinality, cardinality - 1, dim)
    assert_adapter_slots(claims, component)


@pytest.mark.parametrize("defect,diagnostic", [
    ("ownership", "association.*overlaps objective owner"),
    ("boundary", "association.*undeclared ordinary boundary"),
    ("ports", "association.*source/pair-selected"),
    ("support", "association.*inconsistent fixed active support"),
    ("association", "association.*association cut"),
])
def test_direct_stage_rejects_malformed_graph_declarations_by_name(defect, diagnostic):
    template = declared()
    spec = template.hybrid_components["association"]
    if defect == "ownership":
        update(template, regions=(*spec.regions, replace(spec.regions[0], name="overlap")))
    elif defect == "boundary":
        update(template, adapters=())
    elif defect == "ports":
        update(template, regions=(replace(spec.regions[0], ports=tuple(reversed(spec.regions[0].ports))), *spec.regions[1:]))
    elif defect == "support":
        update(template, separators=(*spec.separators[:2], replace(spec.separators[2], active_support=(0,))))
    else:
        template = exposed()
        template.factors["outside"] = FactorSpec("outside", "gaussian_mixture", {"out": "xp", "selector": "z0"}, param_blobs={"mean": "m", "cov": "V"})
        del template.ports["state"], template.ports["label"]
    topology = build_topology(template)
    claims = discover_specialized(template, topology, "sum")
    assert not claims.hybrid_components
    with pytest.raises(compiler.CompileError, match=diagnostic):
        validate_admission(template.name, claims)


def test_snapshots_are_immutable_and_do_not_resolve_or_execute(monkeypatch):
    import equinox as eqx
    import jax
    from cffg import Engine, inits
    from cffg.hybrid_joint import Coordinate
    from cffg.lifecycle import StateLifecycle
    from cffg.resolution import sources

    def forbidden(*args, **kwargs):
        pytest.fail("structural discovery called numerical resolution or execution")

    registry = snapshot_registry()
    bindings = dict(registry.bindings)
    for kind in ("categorical", "gaussian", "gaussian_linear"):
        kernel = bindings[kind]
        parameters = {role: replace(parameter, shape=forbidden) for role, parameter in kernel.binding.parameters.items()}
        bindings[kind] = replace(kernel, binding=replace(kernel.binding, parameters=parameters, pack_params=forbidden))
    registry = replace(registry, bindings=bindings)
    for target, name in ((np, "load"), (inits, "from_spec"), (sources, "resolve_arrays"),
                         (Coordinate, "from_sites"), (Engine, "__init__"), (StateLifecycle, "init_state"),
                         (eqx, "filter_jit"), (jax, "jit"), (jax, "grad"), (jax, "value_and_grad"),
                         (jax, "jacfwd"), (jax, "jacrev"), (jax, "eval_shape")):
        monkeypatch.setattr(target, name, forbidden)
    template = declared()
    claims, registered, drafts = stages(template, registry)
    template.factors.clear()
    template.variables.clear()
    template.blobs.clear()
    template.hybrid_components.clear()
    actual = complete_specialized(claims, OrdinaryMembership(registered.bp_members))
    assert_same(actual, drafts)
    component, = actual.hybrid_components
    with pytest.raises(FrozenInstanceError):
        component.plan = None
    with pytest.raises(TypeError):
        component.plan.regions[0].separator_indices[0] = 99
    with pytest.raises(TypeError):
        component.arrays[0].members[0].runtime["changed"] = None


def test_undeclared_models_skip_d12_discovery_and_public_guard_stays_closed(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("D12 discovery was invoked")

    monkeypatch.setattr(discovery, "discover_hybrid_components", forbidden)
    template = declared()
    with pytest.raises(compiler.CompileError, match="association.*hybrid-regime-ec-v1"):
        compiler.compile_template(template)
    template.hybrid_components.clear()
    claims, _, drafts = stages(template)
    assert not claims.hybrid_components and not drafts.hybrid_components
    assert claims.recurrence and claims.mixtures and claims.equalities
