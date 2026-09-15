"""General D12 metadata preserves admitted scopes without running a solver."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from test_hybrid_declarations import declared, exposed, update
from test_hybrid_discovery import branched, one_label_unary, stages

from cffg.core import BlobRef, FactorSpec, HybridComponentSpec, HybridRegionSpec, HybridSeparatorSpec, Template, VariableSpec
from cffg.core.compose import merge, scoped
from cffg.evaluator_contract import describe_evaluator
from cffg.incidence_contract import chain_applicability, from_draft


def contract(template):
    _, _, drafts = stages(template)
    return tuple(from_draft(draft) for draft in drafts.hybrid_components)


def graph(links, *, cardinality=3, dim=2, support=(0, 2)):
    """Actual admitted graph, with a selected Gaussian unary at every Y."""
    template = Template("directed")
    copies, regions, separators, equalities = {}, [], [], []

    def copy(node, domain):
        group = copies.setdefault((node, domain), [])
        name = f"{domain}_{node}_{len(group)}"
        group.append(name)
        template.variables[name] = VariableSpec(name, cardinality=cardinality) if domain == "z" else VariableSpec(name, dim=dim)
        return name

    def factor(name, kind, edges, roles):
        blobs = {role: f"{name}_{role}" for role in roles}
        template.factors[name] = FactorSpec(name, kind, edges, param_blobs=blobs)
        template.blobs.update({b: BlobRef(b, "absent.npy") for b in blobs.values()})

    for node in sorted({node for link in links for node in link}):
        z, x = copy(node, "z"), copy(node, "x")
        separators.append(HybridSeparatorSpec(f"y{node}", z, x, support))
        factor(f"s{node}", "gaussian_mixture", {"selector": z, "out": x}, ("mean", "cov"))
        regions.append(HybridRegionSpec(f"u{node}", "unary", f"s{node}", (f"y{node}",)))
    for i, (left, right) in enumerate(links):
        zi, zj, xr = copy(left, "z"), copy(right, "z"), copy(left, "x")
        factor(f"r{i}", "recurrence_link", {"in": zi, "out": zj, "drive": xr}, ("weights", "offsets"))
        factor(f"d{i}", "gaussian_mixture_linear", {"selector": copy(right, "z"), "in": copy(left, "x"), "out": copy(right, "x")},
               ("matrix", "offset", "cov"))
        separators.append(HybridSeparatorSpec(f"h{i}", zj, xr, support))
        regions.extend((HybridRegionSpec(f"r{i}", "recurrence", f"r{i}", (f"y{left}", f"h{i}")),
                        HybridRegionSpec(f"d{i}", "dynamics", f"d{i}", (f"h{i}", f"y{right}"))))
    for (node, domain), leaves in copies.items():
        # Each equality has three ends; the last two copies can share an edge.
        remaining = list(leaves)
        while len(remaining) > 2:
            name = f"eq_{node}_{domain}_{len(remaining)}"
            if len(remaining) == 3:
                edges = remaining
                remaining = []
            else:
                bridge = f"{name}_bridge"
                template.variables[bridge] = replace(template.variables[remaining[0]], name=bridge)
                edges = [*remaining[:2], bridge]
                remaining = [bridge, *remaining[2:]]
            factor(name, "equality", dict(zip(("a", "b", "c"), edges)), ())
            equalities.append(name)
        if len(remaining) == 2:
            first, second = remaining
            template.factors = {name: replace(f, edges={role: first if v == second else v for role, v in f.edges.items()})
                                for name, f in template.factors.items()}
            separators = [replace(s, regime=first if s.regime == second else s.regime,
                                  state=first if s.state == second else s.state) for s in separators]
            del template.variables[second]
    template.hybrid_components["general"] = HybridComponentSpec("general", separators=separators, regions=regions, equalities=equalities)
    return template


def intermediate_unary():
    template = declared()
    spec = template.hybrid_components["association"]
    for name, prototype in (("xsite", "xp"), ("xbridge", "xp"), ("zd", "z1"), ("zsite", "z1")):
        template.variables[name] = replace(template.variables[prototype], name=name)
    template.factors["eq"] = replace(template.factors["eq"], edges={"a": "xr", "b": "xd", "c": "xbridge"})
    template.factors["eqx"] = FactorSpec("eqx", "equality", {"a": "xbridge", "b": "xp", "c": "xsite"})
    template.factors["eqz"] = FactorSpec("eqz", "equality", {"a": "z1", "b": "zd", "c": "zsite"})
    template.factors["d"] = replace(template.factors["d"], edges={**template.factors["d"].edges, "selector": "zd"})
    template.factors["hsite"] = FactorSpec("hsite", "gaussian_mixture", {"selector": "zsite", "out": "xsite"},
                                          param_blobs={"mean": "hmean", "cov": "hcov"})
    template.blobs.update({name: BlobRef(name, "absent.npy") for name in ("hmean", "hcov")})
    update(template, regions=(*spec.regions, HybridRegionSpec("h_unary", "unary", "hsite", ("middle",))),
           equalities=(*spec.equalities, "eqx", "eqz"))
    return template


@pytest.mark.parametrize("branches", [2, 3])
def test_actual_outward_branches_keep_every_port_and_degree_correction(branches):
    value, = contract(branched(branches, 3, 2))
    shared, = [s for s in value.separators if s.name == "shared"]
    assert len(shared.ports) == branches + 2
    objective, = [o for o in value.objectives if o.kind == "separator" and o.owner == "shared"]
    assert (objective.port_count, objective.entropy_coefficient) == (branches + 2, branches + 1)
    assert len(value.ports) == 5 * branches + 2
    assert len(value.auxiliaries) == branches
    result = chain_applicability(value)
    assert result.internal_mapping is None
    assert "directed_branching" in {r.code for r in result.general_requirements}
    # Two outward edges are an undirected path, but cannot be silently reversed.
    assert len([r for r in value.regions if r.kind == "recurrence"]) == branches


@pytest.mark.parametrize("cardinality", [1, 5])
def test_unary_only_total_cardinality_does_not_invent_transition(cardinality):
    template = one_label_unary()
    template.variables["z"] = replace(template.variables["z"], cardinality=cardinality)
    value, = contract(template)
    assert not value.auxiliaries
    assert all(p.coefficients.labels == (0,) and p.coefficients.constant_count == 0 for p in value.ports)
    result = chain_applicability(value)
    assert result.internal_mapping is None
    codes = {r.code for r in result.general_requirements}
    assert "no_transitions" in codes
    assert ("total_cardinality" in codes) == (cardinality == 1)
    boundary = describe_evaluator(value, dtype="float32")
    assert boundary.coordinate_partitions[1].size == 0
    assert boundary.coordinate_partitions[1].constraint == "nonnegative"
    assert boundary.coupled_coordinate_size is boundary.coupled_residual_size is None


def test_intermediate_selected_unary_is_not_moved_or_declared_invalid():
    value, = contract(intermediate_unary())
    unary, = [r for r in value.regions if r.name == "h_unary"]
    assert value.ports[unary.ports[0]].separator == "middle"
    assert tuple(s.role for s in unary.parameters.inputs) == ("mean", "cov")
    result = chain_applicability(value)
    assert result.internal_mapping is None
    requirement, = [r for r in result.general_requirements if r.code == "intermediate_inputs"]
    assert requirement.owners == ("middle", "h_unary")
    assert any(o.owner == "h_unary" and o.entropy_coefficient == -1 for o in value.objectives)


def test_path_order_is_directed_and_original_unary_owners_are_retained():
    value, = contract(graph(((7, 2), (2, 9), (9, 1))))
    result = chain_applicability(value)
    assert not result.general_requirements and not result.exterior_obligations
    mapping = result.internal_mapping
    assert mapping.shape == (4, 3, 2)
    assert tuple(s.separator for s in mapping.sites) == ("y7", "y2", "y9", "y1")
    assert tuple(s.unary_owners for s in mapping.sites) == (("u7",), ("u2",), ("u9",), ("u1",))
    assert tuple(t.recurrence for t in mapping.transitions) == ("r0", "r1", "r2")
    assert {r.code for r in result.unresolved_inputs} == {"resolved_parameters", "fixed_support_origins_dtype", "numerical_qualification"}
    boundary = describe_evaluator(value, dtype="float64")
    assert boundary.coupled_coordinate_size == value.coordinate_size
    assert boundary.coupled_residual_size == boundary.internal_residual_size


@pytest.mark.parametrize("links", [((0, 1), (1, 0)), ((0, 1), (2, 3))])
def test_cycles_and_independent_paths_cannot_be_concatenated(links):
    value, = contract(graph(links))
    result = chain_applicability(value)
    assert result.internal_mapping is None
    assert "not_one_directed_path" in {r.code for r in result.general_requirements}


def test_components_and_both_adapter_domains_keep_slot_and_exterior_ownership():
    values = contract(merge(scoped(exposed(), "a"), scoped(declared(), "b"), "pair"))
    assert tuple(v.name for v in values) == ("a/association", "b/association")
    assert {a.domain for a in values[0].adapters} == {"categorical", "gaussian"}
    for value in values:
        result = chain_applicability(value)
        assert result.internal_mapping is not None
        assert len(result.exterior_obligations) == len(value.adapters)
        assert {name for s in result.internal_mapping.sites for name in s.held_adapter_inputs} == {a.name for a in value.adapters}
        for adapter in value.adapters:
            assert adapter.incoming_slot != adapter.outgoing_slot
            assert {adapter.incoming_slot, adapter.outgoing_slot} == set(adapter.gather)
            assert adapter.ordinary_entropy_increment == 1
            assert adapter.ordinary_entropy_coefficient is None
            objective, = [o for o in value.objectives if o.owner == adapter.name]
            assert (objective.energy, objective.entropy_coefficient, objective.port_count) == ("zero", -1, 2)
        boundary = describe_evaluator(value, dtype="float32")
        assert boundary.coupled_coordinate_size is boundary.coupled_residual_size is None
        assert "exterior_stationarity" in boundary.certificates
        assert {r.name for r in boundary.inputs if r.owners[0] in {a.name for a in value.adapters}} == {
            "ordinary_incoming_cavity", "ordinary_port_belief", "ordinary_outgoing_message"}


def test_disconnected_shapes_remain_admitted_and_cannot_share_a_chain_batch():
    template = merge(scoped(graph(((0, 1),)), "a"),
                     scoped(graph(((0, 1),), cardinality=4, dim=3, support=(1, 3)), "b"), "mixed")
    parts = tuple(template.hybrid_components.values())
    template.hybrid_components = {"mixed": HybridComponentSpec(
        "mixed", separators=tuple(s for p in parts for s in p.separators),
        regions=tuple(r for p in parts for r in p.regions), equalities=tuple(e for p in parts for e in p.equalities))}
    value, = contract(template)
    assert {(s.cardinality, s.dim) for s in value.separators} == {(3, 2), (4, 3)}
    result = chain_applicability(value)
    assert result.internal_mapping is None
    assert "heterogeneous_shapes" in {r.code for r in result.general_requirements}


@pytest.mark.parametrize("support", [(4,), (6, 2)])
def test_support_keeps_nonzero_gauge_label_and_only_at_risk_sticks(support):
    value, = contract(graph(((0, 1),), cardinality=9, support=support))
    assert all(p.coefficients.gauge_label == min(support) for p in value.ports)
    assert all(p.coefficients.labels == tuple(sorted(support)) for p in value.ports)
    auxiliary, = value.auxiliaries
    assert auxiliary.source_labels == tuple(sorted(support))
    assert auxiliary.sticks == tuple(range(max(support) + 1))
    assert chain_applicability(value).internal_mapping.shape == (2, 9, 2)


@pytest.mark.parametrize("branches,cardinality,dim", [(3, 3, 2), (40, 3, 2), (3, 90, 2), (3, 3, 80)])
def test_sparse_gauges_and_auxiliaries_scale_without_dense_buffers(branches, cardinality, dim):
    value, = contract(branched(branches, cardinality, dim))
    per_port = 1 + 2 * (dim + dim * (dim + 1) // 2)
    assert value.message_size == len(value.ports) * per_port
    assert tuple(p.offset for p in value.ports) == tuple(range(0, value.message_size, per_port))
    for port in value.ports:
        assert port.coefficients.labels == (0, cardinality - 1)
        assert port.coefficients.gauge_label == 0
    for i, auxiliary in enumerate(value.auxiliaries):
        assert auxiliary.source_labels == (0, cardinality - 1)
        assert auxiliary.sticks == tuple(range(cardinality - 1))
        assert auxiliary.size == 2 * (cardinality - 1)
        assert auxiliary.offset == value.message_size + i * auxiliary.size
    assert all(s.origin_shape == (cardinality, dim) for s in value.separators)
    f32, f64 = (describe_evaluator(value, dtype=d) for d in ("float32", "float64"))
    assert f64.inputs[0].logical_bytes == 2 * f32.inputs[0].logical_bytes == 8 * value.coordinate_size
    assert f32.accepted_state_requirements is f32.scratch_requirements is None
    assert next(r for r in f32.outputs if r.name == "regional_beliefs").logical_bytes is None
    assert value.plan.coordinate_requirements is value.plan.scratch_requirements is None


def test_reordering_and_renaming_preserve_incidence_and_parameter_roles():
    template = graph(((7, 2), (2, 9)))
    original, = contract(template)
    spec = template.hybrid_components["general"]
    template.hybrid_components["general"] = replace(spec, regions=tuple(reversed(spec.regions)),
                                                   separators=tuple(reversed(spec.separators)))
    template.factors = dict(reversed(tuple(template.factors.items())))
    reordered, = contract(template)
    assert reordered.ports == original.ports
    assert reordered.auxiliaries == original.auxiliaries
    assert chain_applicability(reordered) == chain_applicability(original)
    renamed, = contract(scoped(template, "opaque:owner%"))
    assert renamed.coordinate_size == original.coordinate_size
    assert [p.position for p in renamed.ports] == [p.position for p in original.ports]
    assert [tuple(s.role for s in r.parameters.inputs) for r in renamed.regions] == [tuple(s.role for s in r.parameters.inputs) for r in original.regions]
    assert tuple(s.separator for s in chain_applicability(renamed).internal_mapping.sites) == (
        "opaque:owner%/y7", "opaque:owner%/y2", "opaque:owner%/y9")


def test_malformed_contracts_are_named_and_empty_support_is_not_a_layout():
    _, _, drafts = stages(declared())
    draft, = drafts.hybrid_components
    for broken, match in ((replace(draft, arrays=()), "parameter inputs"),
                          (replace(draft, objectives=()), "objective ownership"),
                          (replace(draft, ownership=replace(draft.ownership, executed=(*draft.ownership.executed, draft.ownership.executed[0]))),
                           "execution ownership"),
                          (replace(draft, arrays=(*draft.arrays, draft.arrays[0])), "duplicated or unowned")):
        with pytest.raises(ValueError, match=f"association.*{match}"):
            from_draft(broken)
    spec = draft.plan.declaration
    with pytest.raises(ValueError, match="association.*left.*invalid fixed label support"):
        replace(draft.plan, declaration=replace(spec, separators=(replace(spec.separators[0], active_support=()), *spec.separators[1:])))
    with pytest.raises(ValueError, match="association.*explicit float32 or float64"):
        describe_evaluator(from_draft(draft), dtype=None)
    mismatched = tuple(replace(s, dim=3) if s.name == "right" else s for s in draft.plan.separators)
    with pytest.raises(ValueError, match="association.*channel.*dimensions disagree"):
        from_draft(replace(draft, plan=replace(draft.plan, separators=mismatched)))
    inconsistent = replace(spec, separators=(spec.separators[0], replace(spec.separators[1], active_support=(0,)), spec.separators[2]))
    with pytest.raises(ValueError, match="association.*shared regime copies/support/dimensions disagree"):
        from_draft(replace(draft, plan=replace(draft.plan, declaration=inconsistent)))


def test_metadata_never_resolves_calls_shapes_or_runs_numerics(monkeypatch):
    import jax
    from cffg import hybrid_joint, hybrid_joint_solver, hybrid_joint_storage, hybrid_separator, inits
    from cffg.resolution import sources

    _, _, drafts = stages(declared())
    draft, = drafts.hybrid_components

    def forbidden(*args, **kwargs):
        pytest.fail("metadata invoked array resolution, a shape callback or numerical execution")

    for target, name in ((np, "load"), (inits, "from_spec"), (sources, "resolve_arrays"),
                         (hybrid_joint.Coordinate, "from_sites"), (hybrid_joint, "evaluate"),
                         (hybrid_joint, "linearize"), (hybrid_joint, "initialize"),
                         (hybrid_joint_solver, "solve"), (hybrid_joint_storage, "buffer_values"),
                         (hybrid_separator, "evaluate"), (jax, "jit"), (jax, "grad"), (jax, "eval_shape")):
        monkeypatch.setattr(target, name, forbidden)
    arrays = []
    for requirement in draft.arrays:
        inputs = []
        for source in requirement.inputs:
            shape = source.shape_requirement
            if shape is not None:
                shape = replace(shape, parameter=replace(shape.parameter, shape=forbidden))
            inputs.append(replace(source, shape_requirement=shape))
        arrays.append(replace(requirement, inputs=tuple(inputs)))
    value = from_draft(replace(draft, arrays=tuple(arrays)))
    boundary = describe_evaluator(value, dtype="float32")
    assert chain_applicability(value).internal_mapping is not None
    assert {o.code for o in boundary.obligations} >= {"source_inventory", "storage", "policy_binding", "initialization", "derivative_actions"}
    with pytest.raises(FrozenInstanceError):
        value.ports[0].offset = 1
    with pytest.raises(TypeError):
        value.regions[0].parameters.members[0].runtime["new"] = None
    with pytest.raises(FrozenInstanceError):
        boundary.dtype = "float64"


@pytest.mark.parametrize("factor", ["r", "d", "pz", "px", "hsite"])
@pytest.mark.parametrize("defect", [
    "empty_sources", "partial_sources", "partial_runtime", "matching_omission",
    "empty_destinations", "partial_destinations", "payload_destination", "undeclared_key",
])
def test_complete_parameter_roles_and_addresses_are_required(factor, defect):
    _, _, drafts = stages(intermediate_unary())
    draft, = drafts.hybrid_components
    requirement, = [a for a in draft.arrays if a.members[0].member == factor]
    member, = requirement.members
    first = requirement.inputs[0]
    if defect == "empty_sources":
        broken = replace(requirement, inputs=())
    elif defect == "partial_sources":
        broken = replace(requirement, inputs=requirement.inputs[1:])
    elif defect in ("partial_runtime", "matching_omission"):
        runtime = {role: value for role, value in member.runtime.items() if role != first.role}
        broken = replace(requirement, members=(replace(member, runtime=runtime),))
        if defect == "matching_omission":
            broken = replace(broken, inputs=requirement.inputs[1:])
    elif defect == "empty_destinations":
        broken = replace(requirement, destinations=())
    elif defect == "partial_destinations":
        broken = replace(requirement, destinations=requirement.destinations[1:])
    elif defect == "payload_destination":
        broken = replace(requirement, destinations=(replace(requirement.destinations[0], kind="payload"), *requirement.destinations[1:]))
    else:
        runtime = {**member.runtime, first.role: replace(member.runtime[first.role], key="undeclared")}
        broken = replace(requirement, members=(replace(member, runtime=runtime),))
    arrays = tuple(broken if a is requirement else a for a in draft.arrays)
    with pytest.raises(ValueError, match=f"hybrid component 'association': factor '{factor}':"):
        from_draft(replace(draft, arrays=arrays))


@pytest.mark.parametrize("factor", ["r", "d", "px", "hsite"])
def test_two_runtime_roles_cannot_alias_one_parameter_destination(factor):
    _, _, drafts = stages(intermediate_unary())
    draft, = drafts.hybrid_components
    requirement, = [a for a in draft.arrays if a.members[0].member == factor]
    member, = requirement.members
    first_key = next(iter(member.runtime.values())).key
    runtime = {role: replace(value, key=first_key) for role, value in member.runtime.items()}
    broken = replace(requirement, members=(replace(member, runtime=runtime),))
    arrays = tuple(broken if a is requirement else a for a in draft.arrays)
    with pytest.raises(ValueError, match=f"association.*factor '{factor}'.*one-to-one"):
        from_draft(replace(draft, arrays=arrays))
